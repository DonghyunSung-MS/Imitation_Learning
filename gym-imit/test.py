import gym
env = gym.make("gym_imit:DyrosRed-v1")
obs = env.reset()
action_size = env.action_space.shape[0]
print(env.action_space)
import numpy as np
while True:
    random_action = np.random.randn(action_size)*500
    #print(random_action)
    env.step(random_action)

    env.render()
