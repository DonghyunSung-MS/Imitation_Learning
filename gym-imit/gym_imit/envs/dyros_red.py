import os
import mujoco_py
import time

import numpy as np
import math
from gym.envs.mujoco import mujoco_env
from gym import utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    #print(mass)
    xpos =sim.data.xipos
    #print(xpos.shape)
    return (np.sum(mass * xpos, 0) / np.sum(mass))

#ref: openai/gym
class RedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mj_path, _ = mujoco_py.utils.discover_mujoco()
        xml_path = os.path.join(mj_path, 'model', 'dyros_red_robot.xml')
        #xml_path = "humanoid.xml"
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        com_pos = mass_center(self.model, self.sim)
        #print(com_pos)
        return np.concatenate([data.qpos.flat,
                               data.qvel.flat,
                               com_pos])
    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        #print(pos_before)

        # reward setting

        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum() #torque
        reward = 0.4*math.exp(-(lin_vel_cost[0]**2 + lin_vel_cost[1]**2)) - 0.1*math.exp(-quad_ctrl_cost) + 0.7*math.exp(-alive_bonus)
        qpos = self.sim.data.qpos

        #termination condition

        done = bool(pos_after[2] < 0.5)
        return self._get_obs(),reward, done, dict(reward_linvel=lin_vel_cost,
                                                  reward_quadctrl=-quad_ctrl_cost,
                                                  reward_alive=alive_bonus
                                                  )

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
#register test
