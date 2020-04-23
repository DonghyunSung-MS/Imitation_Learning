# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Humanoid_CMU Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math


from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.mujoco import math as mjmath

from utills.refernceDataBuffer import ReferenceData

import numpy as np

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = 0.02

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('humanoid_CMU.xml'), common.ASSETS


@SUITE.add()
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Stand task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = HumanoidCMUImitation(move_speed=1, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the humanoid_CMU domain."""

  def thorax_upright(self):
    """Returns projection from y-axes of thorax to the z-axes of world."""
    return self.named.data.xmat['thorax', 'zy']

  def head_height(self):
    """Returns the height of the head."""
    return self.named.data.xpos['head', 'z']

  def center_of_mass_position(self):
    """Returns position of the center-of-mass."""
    return self.named.data.subtree_com['thorax']

  def center_of_mass_velocity(self):
    """Returns the velocity of the center-of-mass."""
    return self.named.data.sensordata['thorax_subtreelinvel'].copy()

  def torso_vertical_orientation(self):
    """Returns the z-projection of the thorax orientation matrix."""
    return self.named.data.xmat['thorax', ['zx', 'zy', 'zz']]

  def joint_angles(self):
    """Returns the state without global orientation or position."""
    return self.data.qpos[7:].copy()  # Skip the 7 DoFs of the free root joint.

  def extremities(self):
    """Returns end effector positions in egocentric frame."""
    torso_frame = self.named.data.xmat['thorax'].reshape(3, 3)
    torso_pos = self.named.data.xpos['thorax']
    positions = []
    for side in ('l', 'r'):
      for limb in ('hand', 'foot'):
        torso_to_limb = self.named.data.xpos[side + limb] - torso_pos
        positions.append(torso_to_limb.dot(torso_frame))
    return np.hstack(positions)


class HumanoidCMUImitation(base.Task):
  """A task for the CMU Humanoid."""

  def __init__(self, move_speed, random=None):
    """Initializes an instance of `Humanoid_CMU`.

    Args:
      move_speed: A float. If this value is zero, reward is given simply for
        standing up. Otherwise this specifies a target horizontal velocity for
        the walking task.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._move_speed = move_speed
    super(HumanoidCMUImitation, self).__init__(random=random)

    self.reference_data = None
    self.num_frame = 0.0
    self.max_frame = None

  def initialize_episode(self, physics):
    """Sets a random collision-free configuration at the start of each episode.

    Args:
      physics: An instance of `Physics`.
    """
    penetrating = True
    while penetrating:
      self.num_frame = np.random.randint(self.reference_data.max_frame)
      #print(self.num_frame)
      tmp = self.reference_data(self.num_frame)
      physics.data.qpos[:] = tmp['initialize_episode']
      #randomizers.randomize_limited_and_rotational_joints(physics, self.random)
      # Check for collisions.

      physics.after_reset()
      penetrating = False
    super(HumanoidCMUImitation, self).initialize_episode(physics)

  def get_observation(self, physics):
    """Returns a set of egocentric features."""
    #print(physics.named.data.qvel.shape)
    #print(physics.named.data.qpos.shape)
    obs = collections.OrderedDict()
    obs['phase'] = np.array([self.num_frame/self.max_frame])
    obs['joint_angles'] = physics.joint_angles()
    obs['joint_velocity'] = physics.named.data.qvel[6:].copy()
    #obs['head_height'] = physics.head_height()
    #obs['extremities'] = physics.extremities()
    ori = physics.torso_vertical_orientation()
    obs['torso_orientation_quat'] = mjmath.euler2quat(ori[0], ori[1], ori[2])

    #obs['com_velocity'] = physics.center_of_mass_velocity()
    #obs['velocity'] = physics.velocity()
    return obs
  def PD_torque(self, p_out, j_vel):
      #print(j_vel.shape, p_out.shape, self.reference_data(self.num_frame)["joint_angles"].shape)
      kp = 0.1
      kv = 0.1
      return kp*(self.reference_data(self.num_frame)["joint_angles"] - p_out) - kv*j_vel

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    """Custumizing reward for imitation"""
    '''
    standing = rewards.tolerance(physics.head_height(),
                                 bounds=(_STAND_HEIGHT, float('inf')),
                                 margin=_STAND_HEIGHT/4)
    upright = rewards.tolerance(physics.thorax_upright(),
                                bounds=(0.9, float('inf')), sigmoid='linear',
                                margin=1.9, value_at_margin=0)
    stand_reward = standing * upright
    small_control = rewards.tolerance(physics.control(), margin=1,
                                      value_at_margin=0,
                                      sigmoid='quadratic').mean()
    small_control = (4 + small_control) / 5
    if self._move_speed == 0:
      horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
      dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
      return small_control * stand_reward * dont_move
    else:
      com_velocity = np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]])
      move = rewards.tolerance(com_velocity,
                               bounds=(self._move_speed, float('inf')),
                               margin=self._move_speed, value_at_margin=0,
                               sigmoid='linear')
      move = (5*move + 1) / 6
      '''
    """ imitation  reward"""


    def reward_normalize(a, b, weight):
        return math.exp(weight *np.sum(np.square(a, b)))
    # cyclic motion
    self.num_frame += 1
    if self.num_frame >=self.max_frame:
        self.num_frame = 0

    #Joint angle and orientation reward
    ref_pos = np.hstack([self.reference_data(self.num_frame)['joint_angles'],
                         self.reference_data(self.num_frame)['torso_orientation_quat']])

    ori = physics.torso_vertical_orientation()
    cur_pos = np.hstack([physics.joint_angles(),
                         mjmath.euler2quat(ori[0], ori[1], ori[2])])
    imit_pos_reward = reward_normalize(ref_pos, cur_pos, -2.0)

    #Joint velocity reward
    ref_jvel = self.reference_data(self.num_frame)['joint_velocity']
    cur_jvel = physics.data.qvel[6:].copy()
    imit_vel_reward = reward_normalize(ref_jvel, cur_jvel, -0.1)

    #End effector(extremities) reward
    ref_ee = self.reference_data(self.num_frame)['extremities']
    cur_ee = physics.extremities()
    imit_ee_reward = reward_normalize(ref_ee, cur_ee, -40.0)

    #Com position reward
    ref_com = self.reference_data(self.num_frame)['com_position']
    cur_com = physics.center_of_mass_position()
    imit_com_reward = reward_normalize(ref_com, cur_com, -10.0)

    total_imit_reward = 0.65 * imit_pos_reward + \
                         0.1 * imit_vel_reward + \
                         0.15 * imit_ee_reward + \
                         0.1 * imit_com_reward

    """ Goal reward"""
    #TODO
    total_goal_reward = 0
    return total_imit_reward + total_goal_reward

  def get_termination(self, physics):
    #early termination condition
    height = physics.center_of_mass_position()[2]
    return 1.0 if height<0.6 or height>1.8 else None


  def set_referencedata(self, env, filename, max_num_frames):
      self.reference_data = ReferenceData(env, filename, max_num_frames)
      self.max_frame = self.reference_data.get_max_frame()
