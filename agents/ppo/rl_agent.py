
import os
import copy
from collections import deque
import numpy as np
import wandb
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#from collections import deque

from dm_control import viewer
import utils.logger as logger
import utils.trajectoryBuffer as trajBuff
import utils.rl_utils as rl_utils
from agents.ppo.core import PPOActor, PPOCritic


class PPOAgent:
    def __init__(self, env, args, seed):
        self._env = env
        self._logger = None
        """argument to self value"""
        self.render = args.render
        self.img = None

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.log_dir = args.log_dir
        self.log_interval = args.log_interval

        self.model_dir = args.model_dir
        self.save_interval = args.save_interval

        self.max_iter = args.max_iter
        self.batch_size = args.batch_size
        self.model_update_num = args.model_update_num
        self.total_sample_size = args.total_sample_size

        self.gamma = args.gamma
        self.lamda = args.lamda
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.clip_param = args.clip_param

        self.state_dim = sum([v.shape[0] for k, v in self._env.observation_spec().items()])
        self.action_dim = self._env.action_spec().shape[0]
        self.goal_dim = self._env._task.reference_data.get_goal(0).shape[1]

        print("State Size : ",self.state_dim)
        print("Action Size: ",self.action_dim)
        print("Goal Size: ",self.goal_dim)

        self.dev = None
        if args.gpu:
            self.dev = torch.device("cuda:0")
        else:
            self.dev = torch.device("cpu")

        self._actor = PPOActor(self.state_dim * 3 + self.action_dim * 3 + self.goal_dim*4, self.action_dim, args).to(self.dev)
        self._critic = PPOCritic(self.state_dim * 3 + self.action_dim * 3 + self.goal_dim*4, args).to(self.dev)
        print("Input Size: ",self.state_dim * 3 + self.action_dim * 3 + self.goal_dim*4)
        self.actor_optim = optim.Adam(self._actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self._critic.parameters(), lr=self.critic_lr)

        self.history = None
        self.global_episode = 0

    def test_interact(self, model_path, random=False):
        if random:
            def random_policy(time_step):
                del time_step  # Unused.
                return np.random.uniform(low=self._env.action_spec().minimum,
                                         high=self._env.action_spec().maximum,
                                         size=self._env.action_spec().shape)
            viewer.launch(self._env, policy=random_policy)
        else:
            self._actor.load_state_dict(torch.load(model_path))
            def source_policy(time_step):
                s = None
                for k, v in time_step.observation.items():
                    if s is None:
                        s = v
                    else:
                        s = np.hstack([s, v])
                j_vel = time_step.observation["joint_velocity"]
                s_3d = np.reshape(s, [1, self.state_dim])
                mu, std = self._actor(torch.Tensor(s_3d).to(self.dev))
                p_out = self._actor.get_action(mu, std)
                pd_torque = self._env._task.PD_torque(p_out,j_vel)
                return pd_torque
            viewer.launch(self._env, policy=source_policy)
    def train(self):
        log_file = os.path.join(self.log_dir,"log.txt")
        self._logger = logger.Logger()
        self._logger.configure_output_file(log_file)

        start_time = time.time()
        total_samples = 0
        for iter in range(self.max_iter):
            self.history = trajBuff.Trajectory()
            sample_num, avg_train_reward, avg_train_return, avg_steps = self._rollout()
            #print(len(self.history.states))
            total_samples += sample_num
            wall_time = time.time() - start_time
            wall_time /= 60 * 60 # store time in hours
            if (iter+1)%self.log_interval==0:
                self._logger.log_tabular("Iteration", iter+1)
                self._logger.log_tabular("Wall_Time", wall_time)
                self._logger.log_tabular("Samples", total_samples)
                self._logger.log_tabular("Train_Return", avg_train_return)
                self._logger.log_tabular("Train_Paths", avg_steps)

                wandb.log({"Iteration": iter+1,
                           "Wall_Time": wall_time,
                           "Samples": total_samples,
                           "Avg_reward_iter": avg_train_reward,
                           "Avg_Return_iter": avg_train_return,
                           "Avg_ep_len_iter": avg_steps})

            if (iter+1)%self.save_interval==0:
                self.save_model(iter, self.model_dir)
            self._update(iter)


    def _update(self, iter):
        """update network parameters"""
        states = self.history.states
        actions = self.history.actions
        rewards = self.history.rewards
        masks = self.history.masks

        actions = torch.Tensor(actions).squeeze(0)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)

        old_values = self._critic(torch.Tensor(states).to(self.dev))

        rewards2go, advantages, td_target = rl_utils.get_rl_ele(masks, rewards, old_values, self)

        mu, std = self._actor(torch.Tensor(states).to(self.dev))
        old_policy_log = self._actor.get_log_prob(actions.to(self.dev), mu, std).squeeze(1)
        mse = torch.nn.MSELoss()

        num_sample = len(rewards)

        arr = np.arange(num_sample)
        num = 0
        total_actor_loss = 0
        total_critic_loss = 0
        for _ in range(self.model_update_num):
            np.random.shuffle(arr)
            for i in range(num_sample//self.batch_size):
                mini_batch_index = arr[self.batch_size*i : self.batch_size*(i+1)]
                mini_batch_index = torch.LongTensor(mini_batch_index).to(self.dev)

                states_samples = torch.Tensor(states)[mini_batch_index].to(self.dev)
                actions_samples = torch.Tensor(actions)[mini_batch_index].to(self.dev)
                advantages_samples = advantages.unsqueeze(1)[mini_batch_index].to(self.dev)
                rewards2go_samples = rewards2go.unsqueeze(1)[mini_batch_index].to(self.dev)
                td_target_samples = td_target.unsqueeze(1)[mini_batch_index].to(self.dev)

                old_values_samples = old_values[mini_batch_index].detach()

                new_values_samples = self._critic(states_samples).squeeze(1)

                """Monte-Carlo Estimate(unbias high variance)"""
                critic_loss = mse(new_values_samples, rewards2go_samples)

                """TD, Bellman Error(bias low variance)"""
                #critic_loss = mse(new_values_samples, td_target_samples)
                #print(new_values_samples.shape, rewards2go_samples.shape)
                #Surrogate Loss

                actor_loss, ratio = rl_utils.surrogate_loss(self._actor, old_policy_log.detach(),
                                   advantages_samples, states_samples,  actions_samples,
                                   mini_batch_index)
                #print(actor_loss.shape)
                ratio_clipped = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
                actor_loss = -torch.min(actor_loss,ratio_clipped*advantages_samples).mean()
                num += 1
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

                # update actor & critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

        if (iter+1)%self.log_interval==0:
            self._logger.log_tabular("Actor_loss", total_actor_loss/num)
            self._logger.log_tabular("Critic_loss", total_critic_loss/num)
            self._logger.print_tabular()
            self._logger.dump_tabular()
            wandb.log({"Actor_loss": total_actor_loss/num,
                       "Critic_loss": total_critic_loss/num})



    def _rollout(self):
        """rollout utill sample num is larger thatn max samples per iter"""
        sample_num = 0
        episode = 0
        avg_train_return = 0
        avg_steps = 0
        max_frame = self._env._task.max_frame

        while sample_num < self.total_sample_size:
            steps = 0
            total_reward_per_ep = 0
            time_step = self._env.reset()
            s, _ , __ = self.history.covert_time_step_data(time_step)
            pd_torque = 0
            s_3d = np.reshape(s, [1, self.state_dim])
            s_3d_dummy = np.zeros_like(s_3d)
            a_3d_dummy = np.zeros([1,self.action_dim])
            input = deque(maxlen=6)
            for i in range(2):
                input.append(a_3d_dummy)
                input.append(s_3d_dummy)
            input.append(a_3d_dummy)
            input.append(s_3d)

            while not time_step.last():
                tic = time.time()
                #print(input)
                input_tensor = np.hstack([x for x in input])

                cur_frame = self._env._task.num_frame
                goal_tensor = np.hstack([self._env._task.reference_data.get_goal((cur_frame + 1) % max_frame),
                                         self._env._task.reference_data.get_goal((cur_frame + 2) % max_frame),
                                         self._env._task.reference_data.get_goal((cur_frame + 10) % max_frame),
                                         self._env._task.reference_data.get_goal((cur_frame + 30) % max_frame)])

                input_tensor = np.hstack([input_tensor, goal_tensor])
                #print("steps",steps)
                #print(s_3d.shape, goal_tensor.shape, input_tensor.shape)
                mu, std = self._actor(torch.Tensor(input_tensor).to(self.dev))
                p_out = self._actor.get_action(mu, std)
                input.append(p_out)
                pd_torque = self._env._task.PD_torque(p_out, self._env._physics)

                time_step = self._env.step(pd_torque)
                #print(pd_torque.max(),pd_torque.min())
                s_, r , m = self.history.covert_time_step_data(time_step)

                self.history.store_history(p_out, input_tensor, r, m)
                s = s_
                s_3d = np.reshape(s, [1, self.state_dim])
                input.append(s_3d)
                total_reward_per_ep += r

                if self.render:
                    self._render(tic, steps)

                steps += 1



            episode += 1
            self.global_episode += 1

            wandb.log({"episode":self.global_episode,
                       "Ep_total_reward": total_reward_per_ep,
                       "Ep_Avg_reward": total_reward_per_ep / steps,
                       "Ep_len": steps})

            sample_num = self.history.get_trajLength()

        avg_steps = sample_num / episode
        sum_reward_iter = self.history.calc_return()
        avg_train_return = sum_reward_iter / episode
        avg_train_reward = sum_reward_iter / steps
        return sample_num, avg_train_reward, avg_train_return, avg_steps

    def _render(self, tic, steps):
        max_frame = 90

        width = 640
        height = 480
        video = np.zeros((1000, height, 2 * width, 3), dtype=np.uint8)
        video[steps] = np.hstack([self._env.physics.render(height, width, camera_id=0),
                                 self._env.physics.render(height, width, camera_id=1)])

        if steps==0:
            self.img = plt.imshow(video[steps])
        else:
            self.img.set_data(video[steps])
        toc = time.time()
        clock_dt = toc-tic
        plt.pause(max(0.01, 0.03 - clock_dt))  # Need min display time > 0.0.
        plt.draw()

    def save_model(self, iter, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

        ckpt_path_a = dir + str(iter)+'th_model_a.pth.tar'
        ckpt_path_c = dir + str(iter)+'th_model_c.pth.tar'
        torch.save(self._actor.state_dict(), ckpt_path_a)
        torch.save(self._critic.state_dict(), ckpt_path_c)
