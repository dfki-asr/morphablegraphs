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
"""
Created on Mon Aug 03 19:01:21 2015

@author: Erik herrmann
"""

import numpy as np
from .keyframe_constraint_base import KeyframeConstraintBase
SPATIAL_CONSTRAINT_TYPE_LOCAL_TRAJECTORY = "local_trajectory"


class LocalTrajectoryConstraint(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(LocalTrajectoryConstraint, self).__init__(constraint_desc, precision, weight_factor)
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_LOCAL_TRAJECTORY
        self.trajectory = constraint_desc["trajectory"]
        self.start_t = constraint_desc["start_t"]
        self.n_canonical_frames = constraint_desc["n_canonical_frames"]
        self.joint_name = constraint_desc["joint_name"]
        self.skeleton = skeleton

    def get_positions_from_spline(self, aligned_spline):
        positions = []
        for idx in range(self.n_canonical_frames):
            frame = aligned_spline.evaluate(idx)
            positions.append(self.skeleton.nodes[self.joint_name].get_global_position(frame))
        return positions

    def get_positions_from_frames(self, aligned_quat_frames):
        positions = []
        for frame in aligned_quat_frames:
            positions.append(self.skeleton.nodes[self.joint_name].get_global_position(frame))
        return positions

    def evaluate_motion_spline(self, aligned_spline):
        positions = self.get_positions_from_spline(aligned_spline)
        return sum(self.get_spline_distance(positions))


    def evaluate_motion_sample(self, aligned_quat_frames):
        positions = self.get_positions_from_frames(aligned_quat_frames)
        return sum(self.get_spline_distance(positions))

    def get_spline_distance(self, positions):
        errors = []
        last_p = None
        current_t = self.start_t
        delta = np.zeros(2)
        for p in positions:
            if last_p is not None:
                delta_t = np.linalg.norm(last_p - p)
                current_t += delta_t
            target = self.trajectory.query_point_by_absolute_arc_length(current_t)
            delta[0] = target[0] - p[0]
            delta[1] = target[2] - p[2]
            error = np.dot(delta, delta)
            #print "eval spline",current_t, delta, target, p
            last_p = p
            errors.append(error)
        return errors


    def get_residual_vector_spline(self, aligned_spline):
        positions = self.get_positions_from_spline(aligned_spline)
        return self.get_spline_distance(positions)

    def get_residual_vector(self, aligned_quat_frames):
        positions = self.get_positions_from_frames(aligned_quat_frames)
        return self.get_spline_distance(positions)

    def get_length_of_residual_vector(self):
        return self.n_canonical_frames