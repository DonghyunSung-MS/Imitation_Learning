import copy
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

#from collections import deque

import utills.logger as logger
import utiils.trajectoryBuffer as trajBuff
from agents.ppo.core import PPOActor, PPOCritic


class PPOAgent:
    def __init__(self, env, args):
        self._env = env
        self._logger = None
        """argument to self value"""
        self.log_dir = args.log_dir
        self.log_interval = args.log_interval
        self.model_dir = args.model_dir


        self.max_iter = args.max_iter
        self.batch_size = args.batch_size
        self.total_sample_size = args.total_sample_size

        self.gamma = args.gamma
        self.lamda = args.lamda
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.clip_param = args.clip_param


        self.state_dim = sum([v.shape[0] for k, v in self._env.observation_spec().items()])
        self.action_dim = self._env.action_spec().shape[0]
        self._actor = PPOActor(sself.state_dim, self.action_dim, args)
        self._critic = PPOCritic(self.state_dim, 1, args)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def train(self):
        log_file = os.path.join(sefl.log_dir,"log.txt")
        self._logger = logger.Logger()
        self._logger.configure_output_file(log_file)

        model_file = os.path.join(self.model_dir,"model.pth.tar")

        start_time = time.time()

        for iter in range(args.max_iter_num):
            history = trajBuff.Trajectory()
            self._rollout(history)
            wall_time = time.time() - start_time
            wall_time /= 60 * 60 # store time in hours
            if iter%self.log_interval==0:
                self._logger.log_tabular("Iteration", iter)
                self._logger.log_tabular("Wall_Time", wall_time)
                self._logger.log_tabular("Samples", total_samples)
                self._logger.log_tabular("Train_Return", avg_train_return)
                self._logger.log_tabular("Train_Paths", total_train_path_count)
                self._logger.log_tabular("Test_Return", test_return)
                self._logger.log_tabular("Test_Paths", test_path_count)
            self._update(----)

    def _update(self):
        """update network parameters"""
    def _rollout(self, history):
        """rollout utill sample num is larger thatn max samples per iter"""
        sample_num = 0
        while sample_num < self.total_sample_size:
            time_step = env.reset()
            s, _ , __ = history.covert_time_step_data(time_step)
            s_3d = np.reshape(s, [-1, self.input_dim])
            while not time_step.last():
                mu, std = self.actor(torch.Tensor(s_3d))
                action = self._actor.get_action(mu, std)
                time_step = self._env.step(action)
                s_, r , m = history.covert_time_step_data(time_step)
                history.store_history(action, s, r, m)
                s = s_
                s_3d = np.reshape(s, [-1, self.input_dim])

            sample_num = history.get_trajLength()
