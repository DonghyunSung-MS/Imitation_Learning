## DYROS Tocabi RL Environment

#### How to install
```bash
$ pip install -e . #in the root directory where README.md(gym-imit) located
```

### How to use
* Roll out

```python
import gym
env = gym.make("gym_imit:DyrosRed-v1")
obs = env.reset()
action_size = env.action_space.shape[0]
print(env.action_space)
import numpy as np
while True:
    random_action = np.random.randn(action_size)*500
    next_state, reward, done , info = env.step(random_action)#or your policy
    env.render()

```

* Randomization sample code in test.py

```python
import gym
env = gym.make("gym_imit:DyrosRed-v1")
env.domain_randomizer(xml_name, _OUT_DIR,bounds={
                                                   "mass":[0.8, 1.2],
                                                   "inertia":[0.8, 1.2],
                                                   "friction":[0.8, 1.2],
                                                   "damping":[0.8, 1.2]})
```
you can customize your randomization strategy.

#### Support
__Tocabi(Humanoid) Environment.__
* Model free - on going
* Imitation learning - on going
* Simple domain randomization by making new xml file.
 * dm_control(deepmind control suite) mjcf module
__Source Character Environment.__
