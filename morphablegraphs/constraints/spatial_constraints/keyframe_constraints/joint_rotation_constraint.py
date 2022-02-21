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
from transformations import euler_matrix, quaternion_matrix
from .keyframe_constraint_base import KeyframeConstraintBase
LEN_ROOT_POSITION = 3
LEN_QUAT = 4


class JointRotationConstraint(KeyframeConstraintBase):
    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(JointRotationConstraint, self).__init__(constraint_desc,
                                                      precision,
                                                      weight_factor)
        self.skeleton = skeleton
        self.joint_name = constraint_desc['joint_name']
        self.rotation_type = constraint_desc['rotation_type']
        self.rotation_constraint = constraint_desc['rotation_constraint']
        self.frame_idx = constraint_desc['frame_index']
        if self.rotation_type == "euler":
            rad_angles = list(map(np.deg2rad, self.rotation_constraint))
            self.constraint_rotmat = euler_matrix(rad_angles[0],
                                                  rad_angles[1],
                                                  rad_angles[2],
                                                  axes='rxyz')
        elif self.rotation_type == "quaternion":
            quat = np.asarray(self.rotation_constraint)
            quat /= np.linalg.norm(quat)
            self.constraint_rotmat = quaternion_matrix(quat)
        else:
            raise ValueError('Unknown rotation type!')

    def evaluate_motion_sample(self, aligned_quat_frames):
        """
        Extract the rotation angle of given joint at certain frame, to compare with
        constrained rotation matrix

        """
        return self.evaluate_frame(aligned_quat_frames[self.frame_idx])

    def evaluate_motion_spline(self, aligned_spline):
        return self.evaluate_frame(aligned_spline.evaluate(self.frame_idx))

    def evaluate_frame(self, frame):
        joint_idx = list(self.skeleton.node_name_frame_map.keys()).index(self.joint_name)
        quat_value = frame[LEN_ROOT_POSITION + joint_idx*LEN_QUAT :
                     LEN_ROOT_POSITION + (joint_idx + 1) * LEN_QUAT]
        quat_value = np.asarray(quat_value)
        quat_value /= np.linalg.norm(quat_value)
        rotmat = quaternion_matrix(quat_value)
        diff_mat = self.constraint_rotmat - rotmat
        tmp = np.ravel(diff_mat)
        error = np.linalg.norm(tmp)
        return error

    def get_residual_vector(self, aligned_quat_frames):
        return [self.evaluate_frame(aligned_quat_frames[self.frame_idx])]

    def get_residual_vector_spline(self, aligned_spline):
        return [self.evaluate_frame(aligned_spline.evaluate(self.frame_idx))]

    def get_length_of_residual_vector(self):
        return 1