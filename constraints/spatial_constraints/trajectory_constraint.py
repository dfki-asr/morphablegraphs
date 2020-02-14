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
from .splines.annotated_spline import AnnotatedSpline
from .spatial_constraint_base import SpatialConstraintBase
from .discrete_trajectory_constraint import DiscreteTrajectoryConstraint
from . import SPATIAL_CONSTRAINT_TYPE_TRAJECTORY

TRAJECTORY_DIM = 3  # spline in cartesian space


class TrajectoryConstraint(AnnotatedSpline, SpatialConstraintBase):
    def __init__(self, joint_name, control_points, orientations, spline_type, min_arc_length, unconstrained_indices, skeleton, precision, weight_factor=1.0,
                 closest_point_search_accuracy=0.001, closest_point_search_max_iterations=5000, granularity=1000):
        AnnotatedSpline.__init__(self, control_points, orientations, spline_type, granularity=granularity,
                                     closest_point_search_accuracy=closest_point_search_accuracy,
                                     closest_point_search_max_iterations=closest_point_search_max_iterations)
        SpatialConstraintBase.__init__(self, precision, weight_factor)
        self.semantic_annotation = None
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_TRAJECTORY
        self.joint_name = joint_name
        self.skeleton = skeleton
        self.min_arc_length = min_arc_length
        self.n_canonical_frames = 0
        self.arc_length = 0.0  # will store the full arc length after evaluation
        self.unconstrained_indices = unconstrained_indices
        self.range_start = None
        self.range_end = None
        self.is_collision_avoidance_constraint = False

    def create_discrete_trajectory(self, aligned_quat_frames):
        discrete_trajectory_constraint = DiscreteTrajectoryConstraint(self.joint_name, self.skeleton, self.precision, self.weight_factor)
        discrete_trajectory_constraint.init_from_trajectory(self, aligned_quat_frames, self.min_arc_length)
        return discrete_trajectory_constraint

    def set_active_range(self, new_range_start, new_range_end):
        #print("set range", new_range_start, new_range_end)
        self.range_start = new_range_start
        self.range_end = new_range_end

    def set_number_of_canonical_frames(self, n_canonical_frames):
        self.n_canonical_frames = n_canonical_frames

    def set_min_arc_length_from_previous_frames(self, previous_frames):
        """ Sets the minimum arc length of the constraint as the approximate arc length of the position of the joint
            in the last frame of the previous frames.
        :param previous_frames: list of quaternion frames.
        """
        if previous_frames is not None and len(previous_frames) > 0:
            point = self.skeleton.get_cartesian_coordinates_from_quaternion(self.joint_name, previous_frames[-1])
            closest_point, distance = self.find_closest_point(point, self.min_arc_length, -1)
            if closest_point is not None:
                self.min_arc_length = self.get_absolute_arc_length_of_point(closest_point)[0]
            else:
                self.min_arc_length = self.full_arc_length
        else:
            self.min_arc_length = 0.0

    def evaluate_motion_sample(self, aligned_quat_frames):
        """
        :param aligned_quat_frames: list of quaternion frames.
        :return: average error
        """
        error = np.average(self.get_residual_vector(aligned_quat_frames))
        return error

    def evaluate_motion_spline(self, aligned_spline):
        return self.evaluate_motion_sample(aligned_spline.get_motion_vector())

    def get_residual_vector_spline(self, aligned_spline):
        return self.get_residual_vector(aligned_spline.get_motion_vector())

    def get_residual_vector(self, aligned_quat_frames):
        """ Calculate distances between discrete frames and samples with corresponding arc length from the trajectory
             unconstrained indices are ignored
        :return: the residual vector
        """
        self._n_canonical_frames = len(aligned_quat_frames)
        errors = np.empty(self._n_canonical_frames)
        index = 0
        min_u = self.min_arc_length/self.full_arc_length
        for frame in aligned_quat_frames:
            if index < self._n_canonical_frames:
                joint_position = np.asarray(self.skeleton.get_cartesian_coordinates_from_quaternion(self.joint_name, frame))
                #target = self.point_list[index]
                target, u = self.find_closest_point_fast(joint_position, min_u)
                #target[self.unconstrained_indices] = 0
                #joint_position[self.unconstrained_indices] = 0
                #print joint_position, target
                errors[index] = np.linalg.norm(joint_position-target)
                min_u = u# *self.source.full_arc_length
            else:
                errors[index] = 1000
            index += 1
        #print errors
        return errors

    def get_residual_vector2(self, aligned_quat_frames):
        """ Calculate distances between discrete frames and samples with corresponding arc length from the trajectory
             unconstrained indices are ignored
        :return: the residual vector
        """
        self.arc_length = self.min_arc_length
        #if self.arc_length is None:
        #    return np.zeros(len(aligned_quat_frames))

        last_joint_position = None
        errors = np.empty(len(aligned_quat_frames))
        index = 0
        for frame in aligned_quat_frames:
            joint_position = np.asarray(self.skeleton.get_cartesian_coordinates_from_quaternion(self.joint_name, frame))
            if last_joint_position is not None:
                self.arc_length += np.linalg.norm(joint_position - last_joint_position)
            if self.range_start is None or self.is_active(self.arc_length):
                target = self.query_point_by_absolute_arc_length(self.arc_length)
                last_joint_position = joint_position
                #target[self.unconstrained_indices] = 0
                #joint_position[self.unconstrained_indices] = 0
                errors[index] = np.linalg.norm(joint_position-target)
            else:
                errors[index] = 1000
            index += 1
        errors.fill(np.average(errors))
        print(errors)
        return errors

    def get_length_of_residual_vector(self):
        return self.n_canonical_frames

    def is_active(self, arc_length):
        return self.range_start is not None and self.range_start <= arc_length <= self.range_end