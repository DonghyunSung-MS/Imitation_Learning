import argparse
import numpy as np

from dm_control import suite
from dm_control import mjcf
from dm_control import viewer

from imit_configs import IMIT_CONFIGS
import tasks.humanoid_CMU.humanoid_CMU_imitation as cmu_imit

#mjcf_model = mjcf.from_path("/home/donghyun/RL_study/Imitation_Learning/assets/dyros_red/mujoco_model/dyros_red_robot.xml")
mjcf_model = mjcf.from_path("/home/donghyun/Env_Package/dm_control/dm_control/suite/humanoid_CMU.xml")
#print(type(mjcf_model))
#print(mjcf_model.default.joint.damping)
#mjcf_model.default.joint.damping = 2
#print(mjcf_model.default.joint.damping)




parser = argparse.ArgumentParser()
parser.add_argument("--env",type=str,default="sub7_walk1")
args = parser.parse_args()

configs = IMIT_CONFIGS[args.env] #presetting prameters for each enviroment.

env = cmu_imit.walk()
env._task.set_referencedata(env, configs.filename, configs.max_num_frames)

print(env.action_spec())
print(mjcf_model.default.motor.ctrlrange)
mjcf_model.default.motor.ctrlrange = [-2, 2] #change ctrlrange
print(mjcf_model.default.motor.ctrlrange)
env._physics = env._physics.domain_randomizers(mjcf_model)
print(env.action_spec())
action_spec = env.action_spec()
'''
# Define a uniform random policy.
def random_policy(time_step):
  del time_step  # Unused.
  return np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)
viewer.launch(env, policy=random_policy)
'''
