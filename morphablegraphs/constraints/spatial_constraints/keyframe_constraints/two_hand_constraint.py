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
from .keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION


LEN_TWO_HAND_CONSTRAINT_SET = 3


class TwoHandConstraintSet(KeyframeConstraintBase):
    """calculates the sum of three features:
    the center of both hands, the global orientation of the line between both hands and the distance between both hands

    """
    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(TwoHandConstraintSet, self).__init__(constraint_desc, precision, weight_factor)
        self.skeleton = skeleton
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION
        self.skeleton = skeleton
        self.positions = constraint_desc["positions"]
        self.orientations = constraint_desc["orientations"]
        self.joint_names = constraint_desc["joint"]
        self.target_direction = np.array(self.positions[1]) - np.array(self.positions[0])
        self.target_delta = np.linalg.norm(self.target_direction)
        self.target_center = self.positions[0] + 0.5 * self.target_direction
        self.target_direction /= self.target_delta

        self.n_canonical_frames = constraint_desc["n_canonical_frames"]

    def _get_global_hand_positions(self, frame):
        left_hand_position = self.skeleton.nodes[self.joint_names[0]].get_global_position(frame)
        right_hand_position = self.skeleton.nodes[self.joint_names[1]].get_global_position(frame)
        return left_hand_position, right_hand_position

    def evaluate_motion_spline(self, aligned_spline):
        return sum(self.get_residual_vector_frame(aligned_spline.evaluate(self.canonical_keyframe)))

    def evaluate_motion_sample(self, aligned_quat_frames):
        return sum(self.get_residual_vector_frame(aligned_quat_frames[self.canonical_keyframe]))

    def get_residual_vector(self, aligned_quat_frames):
        return self.get_residual_vector_frame(aligned_quat_frames[self.canonical_keyframe])

    def get_residual_vector_spline(self, aligned_spline):
        return self.get_residual_vector_frame(aligned_spline.evaluate(self.canonical_keyframe))

    def get_residual_vector_frame(self, frame):
        left_hand_position, right_hand_position = self._get_global_hand_positions(frame)

        delta_vector = right_hand_position - left_hand_position
        #print "delta vector", delta_vector
        residual_vector = [0.0, 0.0, 0.0]
        #get distance to center
        residual_vector[0] = np.linalg.norm(self.target_center - (left_hand_position + 0.5 * delta_vector)) #*100.0
        residual_vector[1] = np.linalg.norm(self.positions[0] - left_hand_position)
        residual_vector[2] = np.linalg.norm(self.positions[1] - right_hand_position)

        #print "position", residual_vector[0]
        #get difference to distance between hands
        delta = np.linalg.norm(delta_vector)
        #residual_vector[1] = abs(self.target_delta - delta)
        #print "difference", residual_vector[1]
        #get difference of global orientation
        direction = delta_vector/delta

        #residual_vector[2] = abs(self.target_direction[0] - direction[0]) + \
        #                     abs(self.target_direction[1] - direction[1]) + \
        #                     abs(self.target_direction[2] - direction[2])
        #residual_vector[2] *= 10.0
        #print "direction", residual_vector[2]
        #print self.target_direction, direction
        return residual_vector

    def get_length_of_residual_vector(self):
        return LEN_TWO_HAND_CONSTRAINT_SET
