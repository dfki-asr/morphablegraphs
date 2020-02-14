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
# -*- coding: utf-8 -*-
import numpy as np
from anim_utils.animation_data.motion_concatenation import get_global_node_orientation_vector
from .keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D
from math import acos, degrees


class Direction2DConstraint(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(Direction2DConstraint, self).__init__(constraint_desc, precision, weight_factor)
        self.skeleton = skeleton
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D
        self.direction_constraint = constraint_desc["dir_vector"]
        self.target_dir = np.array([self.direction_constraint[0], self.direction_constraint[2]])
        self.target_dir /= np.linalg.norm(self.target_dir)
        self.target_dir_len = np.linalg.norm(self.target_dir)

    def evaluate_motion_spline(self, aligned_spline):
        frame = aligned_spline.evaluate(self.canonical_keyframe)
        motion_dir = get_global_node_orientation_vector(self.skeleton, self.skeleton.aligning_root_node, frame, self.skeleton.aligning_root_dir)
        magnitude = self.target_dir_len * np.linalg.norm(motion_dir)
        cos_angle = np.dot(self.target_dir, motion_dir)/magnitude
        #print self.target_dir, motion_dir
        cos_angle = min(1,max(cos_angle,-1))
        angle = acos(cos_angle)
        error = abs(degrees(angle))
        #print "angle", error
        return error

    def evaluate_motion_sample(self, aligned_quat_frames):
        frame = aligned_quat_frames[self.canonical_keyframe]
        motion_dir = get_global_node_orientation_vector(self.skeleton, self.skeleton.aligning_root_node, frame, self.skeleton.aligning_root_dir)
        magnitude = self.target_dir_len * np.linalg.norm(motion_dir)
        cos_angle = np.dot(self.target_dir, motion_dir)/ magnitude
        cos_angle = min(1,max(cos_angle,-1))
        angle = acos(cos_angle)
        error = abs(degrees(angle))
        return error

    def get_residual_vector_spline(self, aligned_spline):
        return [self.evaluate_motion_spline(aligned_spline)]

    def get_residual_vector(self, aligned_quat_frames):
        return [self.evaluate_motion_sample(aligned_quat_frames)]

    def get_length_of_residual_vector(self):
        return 1