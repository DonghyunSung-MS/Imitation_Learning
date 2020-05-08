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

"""Demonstration of amc parsing for CMU mocap database.

To run the demo, supply a path to a `.amc` file:

    python mocap_demo --filename='path/to/mocap.amc'

CMU motion capture clips are available at mocap.cs.cmu.edu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import wandb
# Internal dependencies.

from absl import app
from absl import flags

from tasks.humanoid_CMU import humanoid_CMU_imitation
from utils import parse_amc

import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', None, 'amc file to be converted.')
flags.DEFINE_integer('max_num_frames', 277,
                     'Maximum number of frames for plotting/playback')


def main(unused_argv):
    env = humanoid_CMU_imitation.walk()

    # Parse and convert specified clip.
    converted = parse_amc.convert(FLAGS.filename,
                                env.physics, env.control_timestep())

    table = parse_amc.parse(FLAGS.filename)
    table = np.array(table).transpose()
    print(table[1][:].shape)
    print(converted.qpos[0][:].shape)
    print(FLAGS.max_num_frames)
    max_frame = FLAGS.max_num_frames
    print(max_frame)
    frame_list = np.array([x+1 for x in range(max_frame)])
    resample_frame = np.array([x+1 for x in range(converted.qpos.shape[1])])
    '''
    fig, axes = plt.subplots(nrows=2, ncols=6)
    for i in range(6):
        if i<3:
            axes[i//6][i%6].plot(frame_list/120.0, table[i][:]*0.056444)
        else:
            axes[i//6][i%6].plot(frame_list/120.0, table[i][:])
        axes[i//6][i%6].set_title(str(i)+"root")
    for i in range(6):
        i = i+6
        axes[i//6][i%6].plot(resample_frame*0.002, converted.qpos[i-6][:])
        axes[i//6][i%6].set_title("resample"+str(i-6)+"root")
    plt.show()
    '''
    #print(converted.qpos)
    width = 480
    height = 480
    video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

    for i in range(converted.qpos.shape[1] - 1):
        p_i = converted.qpos[:, i]

        with env.physics.reset_context():
            env.physics.data.qpos[:] = p_i

        #print("root_pos : ",env.physics.center_of_mass_position())
        #print("root_vel : ",env.physics.center_of_mass_velocity())
        #print("root_ori : ",env.physics.com_orientation())
        #print("root_avl : ",env.physics.center_of_mass_angvel())
        """
        wandb.log({"frame": i,
                   "root_pos_x" : env.physics.center_of_mass_position()[0],
                   "root_pos_y" : env.physics.center_of_mass_position()[1],
                   "root_pos_z" : env.physics.center_of_mass_position()[2],
                   "root_ori_x" : env.physics.com_orientation()[0],
                   "root_ori_y" : env.physics.com_orientation()[1],
                   "root_ori_z" : env.physics.com_orientation()[2]})
        """
        video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                              env.physics.render(height, width, camera_id=1)])

    tic = time.time()
    i=0
    while True:
        if i == 0:
          img = plt.imshow(video[i])
        elif i==max_frame-1:
          i=0
        else:
          img.set_data(video[i])
        toc = time.time()
        clock_dt = toc - tic
        tic = time.time()
        i=i+1
        # Real-time playback not always possible as clock_dt > .03
        plt.pause(0.002)  # Need min display time > 0.0.

        plt.draw()
        #plt.waitforbuttonpress()



if __name__ == '__main__':
    #wandb.init(project="imitation-learning-walk")
    flags.mark_flag_as_required('filename')
    app.run(main)
