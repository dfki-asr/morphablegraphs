#!/usr/bin/env python
#
# Copyright 2019 DFKI GmbH.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.
import os
import collections
import numpy as np
import scipy as sp
from copy import copy
from transformations import quaternion_matrix, quaternion_from_matrix
from anim_utils.animation_data.motion_distance import convert_quat_frame_to_point_cloud

LEN_Q = 4
LEN_P = 3
BSPLINE_DEGREE = 3


def get_data_analysis_folder(elementary_action,
                             motion_primitive,
                             repo_dir):
    '''
    get data analysis folder, repo_dir is the local path for data svn repository, e.g.: C:\repo
    :param elementary_action (str):
    :param motion_primitive (str):
    :param repo_dir (str):
    :return:
    '''
    mocap_data_analysis_folder = os.path.join(repo_dir, r'data\1 - MoCap\7 - Mocap analysis')
    assert os.path.exists(mocap_data_analysis_folder), ('Please configure path to mocap analysis folder!')
    elementary_action_folder = os.path.join(mocap_data_analysis_folder, 'elementary_action_' + elementary_action)
    if not os.path.exists(elementary_action_folder):
        os.mkdir(elementary_action_folder)
    motion_primitive_folder = os.path.join(elementary_action_folder, motion_primitive)
    if not os.path.exists(motion_primitive_folder):
        os.mkdir(motion_primitive_folder)
    return motion_primitive_folder


def get_aligned_data_folder(elementary_action,
                            motion_primitive,
                            repo_dir=None):
    if repo_dir is None:
        repo_dir = r'C:\repo'
    assert os.path.exists(repo_dir), ('Please configure morphablegraph repository directory!')
    data_folder = 'data'
    mocap_folder = '1 - Mocap'
    alignment_folder = '4 - Alignment'
    elementary_action_folder = 'elementary_action_' + elementary_action
    return os.sep.join([repo_dir,
                        data_folder,
                        mocap_folder,
                        alignment_folder,
                        elementary_action_folder,
                        motion_primitive])



def _convert_pose_to_point_cloud(skeleton, pose, normalize=False):
    _f = pose[:]
    if normalize:
        _f[:3] = [0, 0, 0]
        _f[3:7] = [0, 0, 0, 1]
    point_cloud = convert_quat_frame_to_point_cloud(skeleton, _f)
    return point_cloud


def convert_poses_to_point_clouds(skeleton, motions, normalize=False):
    point_clouds = []
    for m in motions:
        point_cloud = []
        for f in m:
            p = _convert_pose_to_point_cloud(skeleton, f, normalize)
            point_cloud.append(p)
        point_clouds.append(point_cloud)
    return point_clouds


def get_max_translation(motions):
    max_x = 0
    max_y = 0
    max_z = 0
    for k, m in motions.items():
        tmp = np.asarray(m)
        max_x_i = np.max(np.abs(tmp[:, 0]))
        max_y_i = np.max(np.abs(tmp[:, 1]))
        max_z_i = np.max(np.abs(tmp[:, 2]))
        if max_x < max_x_i:
            max_x = max_x_i
        if max_y < max_y_i:
            max_y = max_y_i
        if max_z < max_z_i:
            max_z = max_z_i
    return np.array([max_x, max_y, max_z])


def normalize_root_translation(motions):
    """ Scale all root channels in the given frames.
    It scales the root channel by taking its absolute maximum
    (max_x, max_y, max_z) and divide all values by the maximum,
    scaling all positions between -1 and 1
    """
    scale_vec = get_max_translation(motions)
    if np.any(scale_vec==0):
        return motions, np.array([1,1,1])
    scaled_motions = collections.OrderedDict()
    for key, frames in motions.items():
        frames = np.array(frames)
        frames[:, :3] /= scale_vec
        scaled_motions[key] = frames
    return scaled_motions, scale_vec


def scale_root_translation_in_fpca_data(mean, eigen_vectors, scale_vec, n_coeffs, n_dims):
    root_columns = []
    for coeff_idx in range(n_coeffs):
        coeff_start = coeff_idx * n_dims
        root_columns += np.arange(coeff_start, coeff_start + 3).tolist()

    indices_range = list(range(len(root_columns)))
    x_indices = [root_columns[i] for i in indices_range if i % 3 == 0]
    y_indices = [root_columns[i] for i in indices_range if i % 3 == 1]
    z_indices = [root_columns[i] for i in indices_range if i % 3 == 2]
    eigen_vectors[:, x_indices] *= scale_vec[0]
    eigen_vectors[:, y_indices] *= scale_vec[1]
    eigen_vectors[:, z_indices] *= scale_vec[2]
    mean[x_indices] *= scale_vec[0]
    mean[y_indices] *= scale_vec[1]
    mean[z_indices] *= scale_vec[2]
    return mean, eigen_vectors


def rotate_frames(frames, q):
    new_frames = []
    m = quaternion_matrix(q)
    for f in frames:
        new_frame = copy(f)
        new_frame[:3] = np.dot(m[:3, :3], f[:3])
        #new_frame[3:7] = quaternion_multiply(q, f[3:7])
        new_frame[3:7] = quaternion_from_matrix(np.dot(m, quaternion_matrix(f[3:7])))
        new_frames.append(new_frame)
    return np.array(new_frames)


def align_quaternion_frames(skeleton, motions):
    """align quaternions for blending
    src: http://physicsforgames.blogspot.de/2010/02/quaternions.html
    """
    ref_frame = None
    new_motions = collections.OrderedDict()
    for key, m in motions.items():
        new_frames = []
        for frame in m:
            if ref_frame is None:
                ref_frame = frame
            else:
                offset = 3
                for joint in skeleton.animated_joints:
                    q = frame[offset:offset + 4]
                    ref_q = ref_frame[offset:offset + 4]
                    dot = np.dot(ref_q, q)
                    if dot < 0:
                        frame[offset:offset + 4] = -q
                    offset += 4
            new_frames.append(frame)
        new_motions[key] = np.array(new_frames)
    return new_motions


def get_cubic_b_spline_knots(n_basis, n_canonical_frames):
    """ create cubic bspline knot list, the order of the spline is 4
    :param n_basis: number of knots
    :param n_canonical_frames: length of discrete samples
    :return:
    """
    n_orders = 4
    knots = np.zeros(n_orders + n_basis)
    # there are two padding at the beginning and at the end
    knots[3: -3] = np.linspace(0, n_canonical_frames-1, n_basis-2)
    knots[-3:] = n_canonical_frames - 1
    return knots


def gen_gaussian_eigen(covars):
    covars = np.asarray(covars)
    eigen = np.empty(covars.shape)
    for i, covar in enumerate(covars):
        s, U = sp.linalg.eigh(covar)
        s.clip(0, out=s)
        np.sqrt(s, out=s)
        eigen[i] = U * s
        eigen[i] = np.transpose(eigen[i])
    return eigen
