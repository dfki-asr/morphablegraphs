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
from .spatial_constraint_base import SpatialConstraintBase
from anim_utils.animation_data.utils import get_cartesian_coordinates_from_quaternion


class DiscreteTrajectoryConstraint(SpatialConstraintBase):
    def __init__(self, joint_name,  skeleton, precision, weight_factor=1.0):
        SpatialConstraintBase.__init__(self, precision, weight_factor)
        self.joint_name = joint_name
        self.skeleton = skeleton
        self.point_list = None
        self._n_canonical_frames = 0
        self.unconstrained_indices = None
        self.range_start = None
        self.range_end = None

    def set_active_range(self, range_start, range_end):
        self.range_start = range_start
        self.range_end = range_end

    def set_number_of_canonical_frames(self, n_canonical_frames):
        self._n_canonical_frames = n_canonical_frames

    def init_from_trajectory(self, trajectory, aligned_quat_frames, min_arc_length=0.0):
        """ Use sample of frames to create a list of reference points from the trajectory
        :param trajectory: ParameterizedSpline
        :param aligned_quat_frames: list of quaternion frames.
        :param min_arc_length: float
        :return:
        """
        self.point_list = []
        last_joint_position = None
        arc_length = min_arc_length
        for frame in aligned_quat_frames:
            if last_joint_position is not None:
                arc_length += np.linalg.norm(joint_position-last_joint_position)
            joint_position = np.asarray(self.skeleton.get_cartesian_coordinates_from_quaternion(self.joint_name, frame))
            self.point_list.append(trajectory.query_point_by_absolute_arc_length(arc_length))
            last_joint_position = joint_position
        self._n_canonical_frames = len(self.point_list)
        self.unconstrained_indices = trajectory.unconstrained_indices

    def evaluate_motion_sample(self, aligned_quat_frames):
        """  Calculate sum of distances between discrete frames and samples with corresponding arc length from the trajectory
             unconstrained indices are ignored
        :param aligned_quat_frames: list of quaternion frames.
        :return: error
        """
        return np.average(self.get_residual_vector(aligned_quat_frames))

    def get_residual_vector(self, aligned_quat_frames):
        """ Calculate distances between discrete frames and samples with corresponding arc length from the trajectory
             unconstrained indices are ignored
        :return: the residual vector
        """
        errors = []
        index = 0
        for frame in aligned_quat_frames:
            if index < self._n_canonical_frames:
                joint_position = np.asarray(self.skeleton.get_cartesian_coordinates_from_quaternion(self.joint_name, frame))
                target = self.point_list[index]
                target[self.unconstrained_indices] = 0
                joint_position[self.unconstrained_indices] = 0
                errors.append(np.linalg.norm(joint_position-target))
                index += 1
            else:
                errors.append(0.0)
        return np.array(errors)

    def get_length_of_residual_vector(self):
        return self._n_canonical_frames
