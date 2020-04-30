
from dm_control.suite.utils import parse_amc
from dm_control.mujoco import math as mjmath

import collections
import numpy as np

class ReferenceData:
    def __init__(self, env, filename, max_num_frames):
        self.converted = parse_amc.convert(filename,
                                  env.physics, env.control_timestep())
        self.max_frame = min(max_num_frames, self.converted.qpos.shape[1] - 1)

        self.data_per_frame = []
        '''
        Data contains refrence joint angle, torso orientation quaternion,
        extremities position in torso(egocentric) frame, com position, joint velocity
        '''
        for i in range(self.max_frame):
            qpos = self.converted.qpos[:, i]
            tmp = collections.OrderedDict()
            tmp['initialize_episode'] = qpos
            tmp['joint_angles'] = env.physics.joint_angles()
            tmp['joint_velocity'] = self.converted.qvel[6:, i].copy()

            with env.physics.reset_context():
                env.physics.data.qpos[:] = qpos

            ori = env.physics.torso_vertical_orientation()
            tmp['torso_orientation_quat'] = mjmath.euler2quat(ori[0], ori[1], ori[2])
            tmp['extremities'] = env.physics.extremities()
            tmp['com_position'] = env.physics.center_of_mass_position()
            tmp["com_linear_velocity"] = env.physics.center_of_mass_velocity()
            tmp["com_angular_velocity"] = env.physics.center_of_mass_angvel()
            self.data_per_frame.append(tmp)

    def __call__(self, num_frame):
        return self.data_per_frame[num_frame]

    def get_goal(self, num_frame):
        tmp = np.hstack([self.data_per_frame[num_frame]['joint_angles'],
                         self.data_per_frame[num_frame]['torso_orientation_quat']])
        return np.reshape(tmp,[1,-1])


    def get_max_frame(self):
        return self.max_frame
