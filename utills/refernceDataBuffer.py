
from dm_control.suite.utils import parse_amc
from dm_control.mujoco import math as mjmath

import collections

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
            tmp['joint_velocity'] = self.converted.qvel[7:, i].copy()

            with env.physics.reset_context():
                env.physics.data.qpos[:] = qpos

            ori = env.physics.torso_vertical_orientation()
            tmp['torso_orientation_quat'] = mjmath.euler2quat(ori[0], ori[1], ori[2])
            tmp['extremities'] = env.physics.extremities()
            tmp['com_position'] = env.physics.center_of_mass_position()

            self.data_per_frame.append(tmp)

    def __call__(self, num_frame):
        return self.data_per_frame[num_frame]

    def get_max_frame(self):
        return self.max_frame
