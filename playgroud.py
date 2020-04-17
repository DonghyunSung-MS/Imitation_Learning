from dm_control import viewer
import time
from dm_control import suite
import matplotlib.pyplot as plt
import numpy as np
'''
#Load one task:
env = suite.load(domain_name="humanoid_CMU", task_name="stand")
#viewer.launch(env) #interactive viwer

max_frame = 90

width = 640
height = 480
video = np.zeros((90, height, 2 * width, 3), dtype=np.uint8)

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()
while not time_step.last():
  for i in range(max_frame):
    action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
    time_step = env.step(action)
    video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])
    print(time_step.observation['extremities'].shape)

  tic = time.time()
  for i in range(max_frame):
    if i==0:
        img = plt.imshow(video[i])
    else:
        img.set_data(video[i])
    toc = time.time()
    clock_dt = toc-tic
    plt.pause(max(0.01, 0.03 - clock_dt))  # Need min display time > 0.0.
    plt.draw()
  #plt.waitforbuttonpress()
'''
'''
Test
'''
import sys, os
#sys.path.append(os.pardir)
#print(os.getcwd())
from tasks.humanoid_CMU import humanoid_CMU_imitation
args = dict()
args['filename'] = '/home/donghyun/RL_study/Imitation_Learning/motionData/humanoid_CMU/subject7_walk1.amc'
args['max_num_frames'] = 90
#env = humanoid_CMU_imitation.walk()
#env._task.set_referencedata(env, args['filename'], args['max_num_frames'])
env = suite.load(domain_name="hopper",task_name="stand")
#print(env._task.reference_data(0))

max_frame = 90

width = 640
height = 480
video = np.zeros((1000, height, 2 * width, 3), dtype=np.uint8)

action_spec = env.action_spec()
observation_spec = sum([v.shape[0] for k,v in env.observation_spec().items()])
print(action_spec)
print(observation_spec)
'''
i=0
while True:
    time_step = env.reset()
    while not time_step.last():
        tic = time.time()
        action = np.random.uniform(action_spec.minimum,
                                 action_spec.maximum,
                                 size=action_spec.shape)
        time_step = env.step(action)
        print(time_step)
        state = None
        for k,v in time_step.observation.items():
            if state is None:
                state = v
            else:
                state = np.hstack([state, v])
        print(state)

        video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                              env.physics.render(height, width, camera_id=1)])
        if i==0:
            img = plt.imshow(video[i])
        else:
            img.set_data(video[i])
        toc = time.time()
        clock_dt = toc-tic
        plt.pause(max(0.01, 0.03 - clock_dt))  # Need min display time > 0.0.
        plt.draw()
        i=i+1
'''
