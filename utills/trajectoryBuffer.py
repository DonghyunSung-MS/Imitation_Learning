import numpy as np


class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.clear()
        
    def store_history(self, action, state, reward, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)

    def clear(self):
        for key, vals in vars(self).items():
            if type(vals) is list:
                vals.clear()

    def covert_time_step_data(self, time_step):
        """convert time_step data(DeepMind style) to OpenAI Gym style"""
        tmp_state = None
        for k, v in time_step.observation.items():
            if tmp_state is None:
                tmp_state = v
            else:
                tmp_state = np.hstack([tmp_state, v])
        tmp_reward = time_step.reward
        tmp_mask = 1.0 if not time_step.last() else 0.0

        return tmp_state, tmp_reward, tmp_mask

    def calc_return(self):
        return np.sum(self.rewards)
    def get_trajLength(self):
        return len(self.rewards)

"""test
history = Trajectory()
history.states.append(1)
print(history.states)
history = Trajectory()
print(history.states)
"""
