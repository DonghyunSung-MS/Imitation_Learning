import os
import mujoco_py
import time

import numpy as np
import math
from dm_control import mjcf
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
        #print(os.getcwd())
        self.source_xml_path = os.path.join(os.path.dirname(__file__),
                                "../assets/dyros_tocabi.xml")
        mujoco_env.MujocoEnv.__init__(self, self.source_xml_path, 5)
        utils.EzPickle.__init__(self)

    def domain_randomizer(self, xml_name, out_dir,
                          bounds={
                                    "mass":[0.8, 1.2],
                                    "inertia":[0.8, 1.2],
                                    "friction":[0.8, 1.2],
                                    "damping":[0.8, 1.2]
                                 }):
        """
        Change global joint damping and friction
        Assume that hardware shares same actuator
        """
        mjcf_model = mjcf.from_path(self.source_xml_path)
        mjcf_model.default.joint.damping *= np.random.uniform(bounds["damping"][0],
                                                              bounds["damping"][1])

        mjcf_model.default.joint.frictionloss *= np.random.uniform(bounds["friction"][0],
                                                                   bounds["friction"][1])

        #Assume mass and inertia of all boides change sameway
        bodies = mjcf_model.worldbody.find_all('body')
        mass_rand = np.random.uniform(bounds["mass"][0],
                                      bounds["mass"][1])
        inertia_rand= np.random.uniform(bounds["inertia"][0],
                                        bounds["inertia"][1])
        for body in bodies:
            body.inertial.mass *= mass_rand
            body.inertial.fullinertia*= inertia_rand


        mjcf.export_with_assets(mjcf_model, out_dir=out_dir, out_file_name = xml_name)
        mujoco_env.MujocoEnv.__init__(self, out_dir+xml_name, 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        com_pos = mass_center(self.model, self.sim)
        #print(com_pos)
        return np.concatenate([data.qpos.flat
                               #data.qvel.flat,
                               #com_pos
                               ])
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
    def get_body_name(self):
        mjcf_model = mjcf.from_path(self.source_xml_path)
        bodies = mjcf_model.worldbody.find_all('body')
        body_name_list = []
        for body in bodies:
            body_name_list.append(body.name)
        return body_name_list

    def get_act_name(self):
        mjcf_model = mjcf.from_path(self.source_xml_path)
        motors = mjcf_model.actuator.motor
        act_name_list = []
        for motor in motors:
            act_name_list.append(motor.name)
        return act_name_list

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        )
        return self._get_obs()

    def reset_fix(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
#register test
