import gym
import time
import numpy as np

import utils.BVH2MJDATA

env = gym.make("gym_imit:HumanoidSource-v0")
frametime, qposes, qvels =utils.BVH2MJDATA.load("./animations/LocomotionFlat02_000.bvh", order='zyx')
env.reset()
obs_size = env.observation_space.shape[0]
env.set_state(np.array([0,0,1]+[0]*(obs_size-3)), np.zeros(obs_size-1))

frame_num = qposes.shape[0]

print("observation size : ", obs_size)
print("qposes size : ", qposes.shape)

for i in range(frame_num):
    s = time.time()
    env.set_state(qposes[i, :], qvels[i, :])
    env.render()
    while time.time() - s < frametime:
        continue
