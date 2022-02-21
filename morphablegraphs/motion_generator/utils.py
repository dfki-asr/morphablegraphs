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
import numpy as np
from transformations import quaternion_multiply, quaternion_inverse
from anim_utils.animation_data.motion_concatenation import get_orientation_vector_from_matrix, get_rotation_angle, quaternion_about_axis
from anim_utils.animation_data.utils import quaternion_from_vector_to_vector


def normalize(v):
    return v/np.linalg.norm(v)




def add_frames(skeleton, a, b):
    """ returns c = a + b"""
    c = np.zeros(len(a))
    c[:3] = a[:3] + b[:3]
    for idx, j in enumerate(skeleton.animated_joints):
        o = idx * 4 + 3
        q_a = a[o:o + 4]
        q_b = b[o:o + 4]
        #print(q_a,q_b)
        q_prod = quaternion_multiply(q_a, q_b)
        c[o:o + 4] = q_prod / np.linalg.norm(q_prod)
    return c


def substract_frames(skeleton, a, b):
    """ returns c = a - b"""
    c = np.zeros(len(a))
    c[:3] = a[:3] - b[:3]
    for idx, j in enumerate(skeleton.animated_joints):
        o = idx*4 + 3
        q_a = a[o:o+4]
        q_b = b[o:o+4]
        q_delta = get_quaternion_delta(q_a, q_b)
        c[o:o+4] = q_delta / np.linalg.norm(q_delta)
    return c


def get_quaternion_delta(a, b):
    return quaternion_multiply(quaternion_inverse(b), a)


REF_VECTOR = [0,0,1]


def get_root_delta_angle(skeleton, node_name, frames, target_dir, ref_vector=REF_VECTOR):
    """returns the delta quaternion to align the root orientation with the target direction"""
    m = skeleton.nodes[node_name].get_global_matrix(frames[-1])
    dir_vec = get_orientation_vector_from_matrix(m[:3, :3], ref_vector)
    target_dir = [target_dir[0], target_dir[2]]
    angle = get_rotation_angle(dir_vec, target_dir)
    return  angle


def get_root_delta_q(skeleton, node_name, frames, target_dir, ref_vector=REF_VECTOR):
    """returns the delta quaternion to align the root orientation with the target direction"""
    m = skeleton.nodes[node_name].get_global_matrix(frames[-1])
    dir_vec = np.dot(m[:3, :3], ref_vector)
    dir_vec = normalize(dir_vec)
    q = quaternion_from_vector_to_vector(dir_vec, target_dir)
    return q


def generate_smoothing_factors(window, n_frames):
    """ Generate curve of smoothing factors
    """
    w = float(window)
    smoothing_factors = []
    for f in range(n_frames):
        f = float(f)
        value = 0.0
        if f <= w:
            value = 1 - (f/w)
        smoothing_factors.append(value)
    return np.array(smoothing_factors)


def smooth_quaternion_frames2(prev_frame, frames, window=20, include_root=True):
    """ Smooth quaternion frames given discontinuity frame

    Parameters
    ----------
    prev_frame : frame
    \tA quaternion frame
    frames: list
    \tA list of quaternion frames
    window : (optional) int, default is 20
    The smoothing window
    include_root:  (optional) bool, default is False
    \tSets whether or not smoothing is applied on the x and z dimensions of the root translation
    Returns
    -------
    None.
    """
    n_joints = int((len(frames[0]) - 3) / 4)
    # smooth quaternion
    n_frames = len(frames)
    for i in range(n_joints):
        for j in range(n_frames):
            start = 3 + i * 4
            end = 3 + (i + 1) * 4
            q1 = np.array(prev_frame[start: end])
            q2 = np.array(frames[j][start:end])
            if np.dot(q1, q2) < 0:
                frames[j][start:end] = -frames[j][start:end]

    smoothing_factors = generate_smoothing_factors(window, n_frames)
    #print("smooth", smoothing_factors)
    dofs = list(range(len(frames[0])))[3:]
    if include_root:
        dofs = [0,1,2] + dofs
    else:
        dofs = [1] + dofs
    new_frames = np.array(frames)
    for dof_idx in dofs:
        curve = np.array(frames[:, dof_idx])  # extract dof curve
        magnitude = prev_frame[dof_idx] - curve[0]
        new_frames[:, dof_idx] = curve + (magnitude * smoothing_factors)
    return new_frames


def get_trajectory_end_direction(control_points):
    b = np.array([control_points[-1][0], 0, control_points[-1][2]])
    a = np.array([control_points[-2][0], 0, control_points[-2][2]])
    return normalize(b-a)
