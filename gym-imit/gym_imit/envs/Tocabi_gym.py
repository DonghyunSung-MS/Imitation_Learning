import os
import numpy as np
from math import atan2
from gym.envs.mujoco import mujoco_env
from gym import utils
import json
from math import exp
from pyquaternion import Quaternion
import mujoco_py

def cubic(time,time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):
    x_t = x_0
    if (time < time_0):
        x_t = x_0
    elif (time > time_f):
        x_t = x_f
    else :
        elapsed_time = time - time_0
        total_time = time_f - time_0
        total_time2 = total_time * total_time
        total_time3 = total_time2 * total_time
        total_x    = x_f - x_0

        x_t = x_0 + x_dot_0 * elapsed_time \
            + (3 * total_x / total_time2 \
            - 2 * x_dot_0 / total_time \
            - x_dot_f / total_time) \
            * elapsed_time * elapsed_time \
            + (-2 * total_x / total_time3 + \
            (x_dot_0 + x_dot_f) / total_time2) \
            * elapsed_time * elapsed_time * elapsed_time
    return x_t

def cubicDot(time, time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):

    if (time < time_0):
        x_t = x_dot_0
    elif (time > time_f):
        x_t = x_dot_f
    else:
        elapsed_time = time - time_0
        total_time = time_f - time_0
        total_time2 = total_time * total_time
        total_time3 = total_time2 * total_time
        total_x    = x_f - x_0

        x_t = x_dot_0 \
            + 2*(3 * total_x / total_time2 \
            - 2 * x_dot_0 / total_time \
            - x_dot_f / total_time) \
            * elapsed_time \
            + 3*(-2 * total_x / total_time3 +  \
            (x_dot_0 + x_dot_f) / total_time2) \
            * elapsed_time * elapsed_time

    return x_t

CollisionCheckBodyList = ["base_link",\
            "R_HipRoll_Link", "R_HipCenter_Link", "R_Thigh_Link", "R_Knee_Link",\
            "L_HipRoll_Link", "L_HipCenter_Link", "L_Thigh_Link", "L_Knee_Link",\
            "Waist1_Link", "Waist2_Link", "Upperbody_Link", \
            "R_Shoulder1_Link", "R_Shoulder2_Link", "R_Shoulder3_Link", "R_Armlink_Link", "R_Elbow_Link", "R_Forearm_Link", "R_Wrist1_Link", "R_Wrist2_Link",\
            "L_Shoulder1_Link", "L_Shoulder2_Link", "L_Shoulder3_Link", "L_Armlink_Link", "L_Elbow_Link", "L_Forearm_Link", "L_Wrist1_Link","L_Wrist2_Link"]

ObsBodyList = ["R_Thigh_Link", "R_Knee_Link","R_AnkleCenter_Link", \
            "L_Thigh_Link", "L_Knee_Link", "L_AnkleCenter_Link", \
            "Waist1_Link", "Upperbody_Link", \
            "R_Shoulder1_Link", "R_Armlink_Link", "R_Forearm_Link", "R_Wrist1_Link", "R_Foot_Link",\
            "L_Shoulder1_Link", "L_Armlink_Link", "L_Forearm_Link", "L_Wrist1_Link", "L_Foot_Link"]

Kp = np.asarray([400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, \
                400, 400, 400,\
                400, 400, 400, 400, 400, 400, 400, 400])

Kd = np.asarray([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, \
                10, 10, 10,\
                10, 10, 10, 10, 10, 10, 10, 10])

class DYROSRedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, frameskip=100):
        #xml_path = os.path.join(os.getcwd(), 'assets','dyros_red','mujoco_model','dyros_red_robot_RL.xml')
        xml_path = "/home/donghyun/RL_study/self_project/Imitation_Learning/gym-imit/gym_imit/assets/dyros_red/mujoco_model/dyros_red_robot_RL.xml"
        #xml_path = os.path.join(os.path.dirname(__file__), "assets", "dyros_red/mujoco_model/dyros_red_robot_RL.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, frameskip)
        #mujoco_env.MujocoEnv.__init__(self, 'dyros_red.xml', frameskip)
        utils.EzPickle.__init__(self)
        for id in CollisionCheckBodyList:
            self.collision_check_id.append(self.model.body_name2id(id))
        print("Collision Check ID", self.collision_check_id)

    def _get_obs(self):
        mocap_cycle_dt = 0.033332
        mocap_cycle_period = self.mocap_data_num* mocap_cycle_dt
        phase = np.array((self.init_mocap_data_idx + self.time % mocap_cycle_period / mocap_cycle_dt) % self.mocap_data_num / self.mocap_data_num)

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        body_pos = np.zeros((len(ObsBodyList),3))
        body_quat = np.zeros((len(ObsBodyList),4))
        body_vel = np.zeros((len(ObsBodyList),3))
        body_angvel = np.zeros((len(ObsBodyList),3))

        basequat = Quaternion(self.sim.data.get_body_xquat("base_link"))
        basequat_conj = basequat.conjugate
        basepos = self.get_body_com("base_link")
        basevel = self.sim.data.get_body_xvelp("base_link")
        baseangvel = self.sim.data.get_body_xvelr("base_link")

        com_pos = basequat_conj.rotate(self.sim.data.subtree_com[0] - basepos)
        com_vel = np.asarray([0.0, 0.0, 0.0])
        for i in range(self.model.body_mass.size):
            com_vel += (self.model.body_mass[i] * self.sim.data.cvel[i,0:3])/self.model.body_subtreemass[0]

        for idx, body_name in enumerate(ObsBodyList):
            body_pos[idx] = basequat_conj.rotate(self.get_body_com(body_name) - basepos)
            body_quat[idx] = (basequat_conj * Quaternion(self.sim.data.get_body_xquat(body_name))).elements
            body_vel[idx] = basequat_conj.rotate(self.sim.data.get_body_xvelp(body_name))
            body_angvel[idx] = basequat_conj.rotate(self.sim.data.get_body_xvelr(body_name))

        return np.concatenate([phase.flatten(),
                            # qpos[7:].flatten(),
                            # qvel[6:].flatten(),
                            # com_pos.flatten(),
                            # com_vel.flatten(),
                            basepos[2].flatten(),
                            basequat.elements.flatten(),
                            basevel.flatten(),
                            baseangvel.flatten(),
                            body_pos.flatten(),
                            body_quat.flatten(),
                            body_vel.flatten(),
                            body_angvel.flatten()])

    def step(self, a):
        mocap_cycle_dt = 0.033332
        mocap_cycle_period = self.mocap_data_num* mocap_cycle_dt

        target_vel = (a - self.sim.data.qpos[7:])/self.dt
        for i in range(self.frame_skip):
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
            torque = 400*(a - qpos[7:]) + 40*(- qvel[6:])
            self.do_simulation(torque,1)

        self.time += self.dt

        # kp=900#4900
        # kv=60#140
        # action_size = len(self.action_space.sample())
        # virtual_action_size = action_size + 6
        # target_qvel = (a - self.sim.data.qpos[7:])/ self.dt

        # for timestep in range(self.frame_skip):
        #     qpos = self.sim.data.qpos
        #     qvel = self.sim.data.qvel
        #     MNN_vector = np.zeros(virtual_action_size**2)
        #     mujoco_py.cymj._mj_fullM(self.model, MNN_vector, self.sim.data.qM)
        #     M = MNN_vector.reshape((virtual_action_size, virtual_action_size))
        #     torque = np.matmul(M[6:,6:], kp*(a - qpos[7:]) + kv* (- qvel[6:])) + self.sim.data.qfrc_bias[6:]
        #     self.do_simulation(torque, 1)


        local_time = self.time % mocap_cycle_period
        local_time_plus_init = (local_time + self.init_mocap_data_idx*mocap_cycle_dt) % mocap_cycle_period
        self.mocap_data_idx = (self.init_mocap_data_idx + int(local_time / mocap_cycle_dt)) % self.mocap_data_num
        next_idx = self.mocap_data_idx + 1

        if (self.mocap_data_idx_pre != self.mocap_data_idx) and (self.mocap_data_idx == self.init_mocap_data_idx):
            self.cycle_init_root_pos[0] = self.sim.data.qpos[0]
            self.cycle_init_root_pos[1] = self.sim.data.qpos[1]

        self.mocap_data_idx_pre = np.copy(self.mocap_data_idx)

        target_data_qpos = np.zeros_like(a)
        target_data_qvel = np.zeros_like(a)
        Tar_EE_COM = np.zeros((4,3))
        target_data_body_delta = np.zeros(3)
        target_data_body_vel = np.zeros(3)
        target_com = np.zeros(3)

        for i in range(a.size):
            target_data_qpos[i] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,i+8], self.mocap_data[next_idx,i+8], 0.0, 0.0)
            target_data_qvel[i] =  (self.mocap_data[next_idx,i+8] -  self.mocap_data[self.mocap_data_idx,i+8]) / mocap_cycle_dt


        if(self.mocap_data_idx >= self.init_mocap_data_idx):
            target_data_body_delta[0] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,1] - self.mocap_data[self.init_mocap_data_idx,1], self.mocap_data[next_idx,1]-self.mocap_data[self.init_mocap_data_idx,1], 0.0, 0.0)
            target_data_body_delta[1] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,3] - self.mocap_data[self.init_mocap_data_idx,3], self.mocap_data[next_idx,3]-self.mocap_data[self.init_mocap_data_idx,3], 0.0, 0.0)
            target_data_body_delta[2] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,2] - self.mocap_data[self.init_mocap_data_idx,2], self.mocap_data[next_idx,2]-self.mocap_data[self.init_mocap_data_idx,2], 0.0, 0.0)
        else:
            target_data_body_delta[0] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_num,1] + self.mocap_data[self.mocap_data_idx,1] - self.mocap_data[self.init_mocap_data_idx,1], self.mocap_data[self.mocap_data_num,1] + self.mocap_data[next_idx,1] - self.mocap_data[self.init_mocap_data_idx,1], 0.0, 0.0)
            target_data_body_delta[1] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,3] - self.mocap_data[self.init_mocap_data_idx,3], self.mocap_data[next_idx,3] - self.mocap_data[self.init_mocap_data_idx,3], 0.0, 0.0)
            target_data_body_delta[2] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,2] - self.mocap_data[self.init_mocap_data_idx,2], self.mocap_data[next_idx,2] - self.mocap_data[self.init_mocap_data_idx,2], 0.0, 0.0)

        target_data_body_vel[0] = (self.mocap_data[next_idx,1] - self.mocap_data[self.mocap_data_idx,1])/mocap_cycle_dt
        target_data_body_vel[1] = (self.mocap_data[next_idx,3] - self.mocap_data[self.mocap_data_idx,3])/mocap_cycle_dt
        target_data_body_vel[2] = (self.mocap_data[next_idx,2] - self.mocap_data[self.mocap_data_idx,2])/mocap_cycle_dt

        for ee_idx in range(4):
            for cartesian_idx in range(3):
                data_type = 8 + a.size + 3*ee_idx + cartesian_idx
                Tar_EE_COM[ee_idx,cartesian_idx] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,data_type], self.mocap_data[next_idx,data_type] , 0.0, 0.0)


        # for i in range(3):
        #     data_type = 8 + a.size + Tar_EE_COM.size + i
        #     target_com[i] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,data_type], self.mocap_data[next_idx,data_type] , 0.0, 0.0)


        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        com_pos =self.sim.data.subtree_com[0]

        basequat = Quaternion(self.sim.data.get_body_xquat("base_link"))
        basequat_conj = basequat.conjugate
        basepos = self.get_body_com("base_link")
        EE_CoM = np.concatenate((basequat_conj.rotate(self.get_body_com("R_AnkleCenter_Link") - basepos), \
                basequat_conj.rotate(self.get_body_com("L_AnkleCenter_Link") - basepos), \
                basequat_conj.rotate(self.get_body_com("R_Wrist1_Link") - basepos), \
                basequat_conj.rotate(self.get_body_com("L_Wrist1_Link") - basepos)))
        basequat_desired = Quaternion(self.mocap_data[self.mocap_data_idx,4:8])
        basequat = Quaternion(qpos[3:7])
        baseQuatError = (basequat_desired*basequat.conjugate).angle

        Tar_Body = self.cycle_init_root_pos+target_data_body_delta
        Tar_COM = Tar_Body + target_com

        # self.set_state(
        #     np.concatenate((Tar_Body, basequat_desired.elements, target_data_qpos)),
        #     self.init_qvel + np.concatenate((target_data_body_vel, np.zeros(3), target_data_qvel)),
        # )
        # self.sim.step()


        # for i in range(self.frame_skip):
        #     qpos = self.sim.data.qpos
        #     qvel = self.sim.data.qvel
        #     torque = 400*(target_data_qpos - qpos[7:]) + 40*(target_data_qvel- qvel[6:])
        #     self.do_simulation(torque,1)

        mimic_qpos_reward = 0.55 * exp(-2.0*(np.linalg.norm(target_data_qpos - qpos.flat[7:])**2))
        mimic_qvel_reward = 0.05 * exp(-0.1*(np.linalg.norm(target_data_qvel - qvel.flat[6:])**2))
        mimic_ee_reward = 0.1 * exp(-40*(np.linalg.norm(EE_CoM - Tar_EE_COM.flatten())**2))
        mimic_body_reward = 0.2 * exp(-10*(np.linalg.norm(Tar_Body - qpos.flat[0:3])**2 + 0.5*baseQuatError**2))
        mimic_body_vel_reward = 0.1*exp(-10*(np.linalg.norm(target_data_body_vel - qvel.flat[0:3])**2)) #
        reward = mimic_qpos_reward + mimic_qvel_reward + mimic_ee_reward + mimic_body_reward + mimic_body_vel_reward


        done_by_contact = False
        if self.done_init is False:
            done_by_contact = False
            self.done_init = True
        else:
            for i in range(self.sim.data.ncon):
                if (self.sim.data.contact[i].geom1 == 0 and  any(self.model.geom_bodyid[self.sim.data.contact[i].geom2] == collisioncheckid for collisioncheckid in self.collision_check_id)) or \
                    (self.sim.data.contact[i].geom2 == 0 and any(self.model.geom_bodyid[self.sim.data.contact[i].geom1] == collisioncheckid for collisioncheckid in self.collision_check_id)):
                    done_by_contact = True
                    break

        if not done_by_contact:
            self.epi_len += 1
            self.epi_reward += reward
            return self._get_obs(), reward, done_by_contact, dict(specific_reward=dict(mimic_qpos_reward=mimic_qpos_reward, mimic_qvel_reward=mimic_qvel_reward, mimic_ee_reward= mimic_ee_reward, mimic_body_reward=mimic_body_reward, mimic_body_vel_reward=mimic_body_vel_reward))
        else:
            mimic_qpos_reward = 0.0
            mimic_qvel_reward = 0.0
            mimic_ee_reward = 0.0
            mimic_body_reward = 0.0
            mimic_body_vel_reward = 0.0
            reward = 0.0
            return_epi_len = self.epi_len
            return_epi_reward = self.epi_reward
            return self._get_obs(), reward, done_by_contact, dict(episode=dict(r=return_epi_reward, l=return_epi_len), specific_reward=dict(mimic_qpos_reward=mimic_qpos_reward, mimic_qvel_reward=mimic_qvel_reward, mimic_ee_reward= mimic_ee_reward, mimic_body_reward=mimic_body_reward,mimic_body_vel_reward=mimic_body_vel_reward))



    def reset_model(self):
        self.time = 0.0
        self.epi_len = 0
        self.epi_reward = 0
        self.init_mocap_data_idx = np.random.randint(low=0, high=self.mocap_data_num-1)
        self.mocap_data_idx_pre = np.copy(self.init_mocap_data_idx)
        next_idx = self.init_mocap_data_idx + 1
        mocap_cycle_dt = 0.033332
        quat_desired = np.zeros(4)
        quat_desired =self.mocap_data[self.init_mocap_data_idx,4:8]
        self.cycle_init_root_pos = self.sim.data.qpos[0:3].copy()

        q_desired = self.mocap_data[self.init_mocap_data_idx,8:8+len(self.action_space.sample())]
        qvel_desired = (self.mocap_data[next_idx,8:8+len(self.action_space.sample())] - self.mocap_data[self.init_mocap_data_idx,8:8+len(self.action_space.sample())]) / mocap_cycle_dt
        target_data_body_vel = (self.mocap_data[next_idx,1:4] - self.mocap_data[self.init_mocap_data_idx,1:4]) / mocap_cycle_dt

        self.set_state(
            self.init_qpos + np.concatenate((np.zeros(3), -self.init_qpos[3:7] + quat_desired, q_desired)),
            self.init_qvel + np.concatenate((target_data_body_vel, np.zeros(3),qvel_desired)),
        )
        # + np.concatenate(((self.mocap_data[next_idx,1:4] - self.mocap_data[self.init_mocap_data_idx,1:4])/mocap_cycle_dt, np.zeros(3+len(self.action_space.sample()))))
        #
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
