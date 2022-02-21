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
from anim_utils.animation_data.utils import calculate_weighted_frame_distance_quat, quat_distance 
from .keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE
LEN_QUAT = 4
LEN_ROOT_POS = 3
LEN_QUAT_FRAME = 79

class PoseConstraintQuatFrame(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(PoseConstraintQuatFrame, self).__init__(constraint_desc, precision, weight_factor)
        self.skeleton = skeleton
        self.pose_constraint = constraint_desc["frame_constraint"]
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE
        assert len(self.pose_constraint) == LEN_QUAT_FRAME, ("pose_constraint is not quaternion frame")
        return

    def evaluate_motion_sample(self, aligned_quat_frames):
        weights = self.skeleton.get_joint_weights()
        error = calculate_weighted_frame_distance_quat(self.pose_constraint,
                                                       aligned_quat_frames[0],
                                                       weights)
        return error

    def evaluate_motion_spline(self, aligned_spline):
        weights = self.skeleton.get_joint_weights()
        error = calculate_weighted_frame_distance_quat(self.pose_constraint,
                                                       aligned_spline.evaluate(0),
                                                       weights)
        return error

    def get_residual_vector_spline(self, aligned_spline):
        return self.get_residual_vector_frame(aligned_spline.evaluate(0))

    def get_residual_vector(self, aligned_quat_frames):
        return self.get_residual_vector_frame(aligned_quat_frames[0])

    def get_residual_vector_frame(self, frame):
        weights = self.skeleton.get_joint_weights()
        residual_vector = []
        quat_frame_a = self.pose_constraint
        quat_frame_b = frame
        for i in range(len(weights) - 1):
            quat1 = quat_frame_a[(i+1)*LEN_QUAT+LEN_ROOT_POS: (i+2)*LEN_QUAT+LEN_ROOT_POS]
            quat2 = quat_frame_b[(i+1)*LEN_QUAT+LEN_ROOT_POS: (i+2)*LEN_QUAT+LEN_ROOT_POS]
            tmp = quat_distance(quat1, quat2)*weights[i]
            residual_vector.append(tmp)
        return residual_vector

    def get_length_of_residual_vector(self):
        return LEN_QUAT_FRAME