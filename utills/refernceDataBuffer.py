
from dm_control.suite.utils import parse_amc
from dm_control.mujoco import math as mjmath

import collections
import numpy as np
import wandb

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def getphi(x_before, x_cur):
    x_before = np.reshape(x_before,[3,3])
    x_cur = np.reshape(x_cur,[3,3])
    s1_b = np.reshape(x_before[:][0],[3])
    s2_b = np.reshape(x_before[:][1],[3])
    s3_b = np.reshape(x_before[:][2],[3])

    s1_b_skew = skew(s1_b)
    s2_b_skew = skew(s2_b)
    s3_b_skew = skew(s3_b)

    s1_c = np.reshape(x_cur[:][0],[3,1])
    s2_c = np.reshape(x_cur[:][1],[3,1])
    s3_c = np.reshape(x_cur[:][2],[3,1])
    #print(s1_b_skew.shape)
    #print(s1_c.shape)
    result = -(np.matmul(s1_b_skew, s1_c) + np.matmul(s2_b_skew, s2_c) + np.matmul(s3_b_skew, s3_c))/2
    result = np.reshape(result,[3])
    return result


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
        dt = env.control_timestep()
        print("dt is ",dt)
        for i in range(self.max_frame):
            qpos = self.converted.qpos[:, i]
            tmp = collections.OrderedDict()
            tmp['initialize_episode'] = qpos
            tmp['joint_angles'] = env.physics.joint_angles()
            tmp['joint_velocity'] = self.converted.qvel[6:, i].copy()

            with env.physics.reset_context():
                env.physics.data.qpos[:] = qpos
            tmp['torso_orientation_quat'] = env.physics.com_orientation()
            tmp['torso_orientation_mat'] = env.physics.com_orientation_mat()
            tmp['extremities'] = env.physics.extremities()
            tmp['com_position'] = env.physics.center_of_mass_position()

            if i>0:
                tmp["com_linear_velocity"] = (tmp['com_position'] - self.data_per_frame[i-1]['com_position']) / dt
                tmp["com_angular_velocity"] = getphi(self.data_per_frame[i-1]['torso_orientation_mat'] ,env.physics.com_orientation_mat()) / dt

            else:
                tmp["com_linear_velocity"] = np.zeros(3)
                tmp["com_angular_velocity"] = np.zeros(3)
            self.data_per_frame.append(tmp)
            #print(tmp["com_linear_velocity"])
            #print(tmp["com_angular_velocity"])
            #print()
            #wandb.log(tmp)
        self.data_per_frame[0]['com_linear_velocity'] = self.data_per_frame[1]['com_linear_velocity']
        self.data_per_frame[0]['com_angular_velocity'] = self.data_per_frame[1]['com_angular_velocity']

    def __call__(self, num_frame):
        return self.data_per_frame[num_frame]

    def get_goal(self, num_frame):
        tmp = np.hstack([self.data_per_frame[num_frame]['joint_angles'],
                         self.data_per_frame[num_frame]['torso_orientation_quat']])
        return np.reshape(tmp,[1,-1])


    def get_max_frame(self):
        return self.max_frame
