import re
import numpy as np
from .Quaternions import Quaternions
from .Pivots import Pivots

"""
BVH Parser and Quaternions is from Daniel Holden's animation utils
(http://theorangeduck.com/page/phase-functioned-neural-networks-character-control)
"""

channelmap = {
    'Xrotation' : 'x',
    'Yrotation' : 'y',
    'Zrotation' : 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x' : 0,
    'y' : 1,
    'z' : 2,
}

SCALE_FACTOR = 0.05644

Mocap2MJFrame = np.array([[0., 0., 1.],
                          [1., 0., 0.],
                          [0., 1., 0.]])

def load(filename, start=None, end=None, order=None, world=False):
    """
    Reads a BVH file and constructs an animation

    Parameters
    ----------
    filename: str
        File to be opened

    start : int
        Optional Starting Frame

    end : int
        Optional Ending Frame

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    world : bool
        If set to true euler angles are applied
        together in world space rather than local
        space

    Returns
    -------

    (animation, joint_names, frametime)
        Tuple of loaded animation and joint names
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = Quaternions.id(0)
    offsets = np.array([]).reshape((0,3))
    parents = np.array([], dtype=int)

    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        """ Modified line read to handle mixamo data """
#        rmatch = re.match(r"ROOT (\w+)", line)
        rmatch = re.match(r"ROOT (\w+:?\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets    = np.append(offsets,    np.array([[0,0,0]]),   axis=0)
            orients.qs = np.append(orients.qs, np.array([[1,0,0,0]]), axis=0)
            parents    = np.append(parents, active)
            active = (len(parents)-1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site: end_site = False
            else: active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2+channelis:2+channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        """ Modified line read to handle mixamo data """
#        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        jmatch = re.match("\s*JOINT\s+(\w+:?\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets    = np.append(offsets,    np.array([[0,0,0]]),   axis=0)
            orients.qs = np.append(orients.qs, np.array([[1,0,0,0]]), axis=0)
            parents    = np.append(parents, active)
            active = (len(parents)-1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start)-1
            else:
                fnum = int(fmatch.group(1))
            jnum = len(parents)
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end-1):
            i += 1
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if   channels == 3:
                positions[fi,0:1] = data_block[0:3]
                rotations[fi, : ] = data_block[3: ].reshape(N,3)
            elif channels == 6:
                data_block = data_block.reshape(N,6)
                positions[fi,:] = data_block[:,0:3]
                rotations[fi,:] = data_block[:,3:6]
            elif channels == 9:
                positions[fi,0] = data_block[0:3]
                data_block = data_block[3:].reshape(N-1,9)
                rotations[fi,1:] = data_block[:,3:6]
                positions[fi,1:] += data_block[:,0:3] * data_block[:,6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = Quaternions.from_euler(np.radians(rotations), order=order, world=False)
    frame_num = positions.shape[0]
    """Rotating Frame & Scaling to make Mujoco friendly data"""
    #qposes
    root_positions = positions[:,0,:] #(frame_num, 3)
    root_positions = SCALE_FACTOR * np.matmul(root_positions, Mocap2MJFrame.T)

    root_orientations = rotations[:,:1] #(frame_num, 1, 3)
    angles, axis = root_orientations.angle_axis()
    axis = np.matmul(axis, Mocap2MJFrame.T)
    root_quat = Quaternions.from_angle_axis(angles, axis)
    root_orientations = root_quat.qs.reshape(frame_num, -1)

    joints_quat = rotations[:,1:] #(frame_num, 30, 3)
    angles, axis = joints_quat.angle_axis()
    axis = np.matmul(axis, Mocap2MJFrame.T)
    joints_quat = Quaternions.from_angle_axis(angles, axis)

    joints_euler = joints_quat.euler()
    joints_euler = joints_euler.reshape(frame_num, -1)

    qposes = np.hstack((root_positions, root_orientations, joints_euler))

    #qvels
    root_linvel = np.zeros_like(root_positions)
    root_linvel[1:,:] = (root_positions[1:, :] - root_positions[0:-1, :])/frametime

    angle, axis = Quaternions(root_quat[1:]* -root_quat[:-1]).angle_axis()
    root_angvel = angle/frametime*axis.reshape(-1,3)
    root_angvel = np.vstack((np.zeros((1,3)), root_angvel))

    angle, axis = Quaternions(joints_quat[1:]*-joints_quat[:-1]).angle_axis()
    joints_angvel = angle.reshape(frame_num-1,-1,1)/frametime * axis
    joints_angvel = np.vstack((np.zeros((1,30,3)), joints_angvel))

    qvels = np.hstack((root_linvel, root_angvel, joints_angvel.reshape(frame_num, -1)))
    #print(qposes.shape, qvels.shape)
    return frametime, qposes, qvels



def get_phi(Rmat_prev, Rmat_current):
    """Rmat (3,3)"""
    s1_hat = skew_mat(Rmat_prev[:, 0])
    s2_hat = skew_mat(Rmat_prev[:, 1])
    s3_hat = skew_mat(Rmat_prev[:, 2])

    prev = np.hstack((s1_hat, s2_hat, s3_hat))

    s1d = Rmat_current[:, 0]
    s2d = Rmat_current[:, 1]
    s3d = Rmat_current[:, 2]

    current = np.vstack((s1d, s2d, s3d))

    return -0.5*np.matmul(prev, current)

def skew_mat(vector3d):
    if vector3d.shape != (1,3):
        TypeError("vector shoud be (1,3) numpy array")
    else:
        return np.array([[0, -vector3d[0, 2], vector3d[0, 1]],
                         [vector3d[0, 2], 0, -vector3d[0]],
                         [-vector3d[0,1], vector3d[0, 0], 0]])
