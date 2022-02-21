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
from . import *
import numpy as np
from .spatial_constraint_base import SpatialConstraintBase
from copy import copy


class TrajectorySetConstraint(SpatialConstraintBase):
    def __init__(self, joint_trajectories, joint_names, skeleton, precision, weight_factor):
        SpatialConstraintBase.__init__(self, precision, weight_factor)
        self.skeleton = skeleton
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_TRAJECTORY_SET
        self.joint_trajectories = joint_trajectories
        self.joint_names = joint_names
        self.joint_arc_lengths = np.zeros(len(self.joint_trajectories))
        self.semantic_annotation = dict()
        self.semantic_annotation["generated"] = True
        self.n_canonical_frames = 0

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

    def set_number_of_canonical_frames(self, n_canonical_frames):
        self.n_canonical_frames = n_canonical_frames

    def set_min_arc_length_from_previous_frames(self, previous_frames):
        """ Sets the minimum arc length of the constraint as the approximate arc length of the position of the joint
            in the last frame of the previous frames.
        :param previous_frames: list of quaternion frames.
        """

        if previous_frames is not None and len(previous_frames) > 0:
            joint_positions = self._extract_joint_positions_from_frame(previous_frames[-1])
            closest_points = [joint_trajectory.find_closest_point(point, min_arc_length, -1)
                              for joint_trajectory, point, min_arc_length
                              in zip(self.joint_trajectories,  joint_positions, self.joint_arc_lengths)]

            self.joint_arc_lengths = [joint_trajectory.get_absolute_arc_length_of_point(closest_point[0])[0]
                                      if closest_point[0] is not None else joint_trajectory.full_arc_length
                                      for joint_trajectory, closest_point
                                      in zip(self.joint_trajectories, closest_points)]
        else:
            self.joint_arc_lengths = np.zeros(len(self.joint_trajectories))

    def _extract_joint_positions_from_frame(self, frame):
        self.skeleton.clear_cached_global_matrices()
        joint_positions = [self.skeleton.nodes[joint_name].get_global_position(frame, True)
                           for joint_name in self.joint_names]
        return joint_positions

    def get_residual_vector(self, aligned_quat_frames):
        residual_vector = np.zeros(self.n_canonical_frames)
        last_joint_positions = None
        joint_arc_lengths = copy(self.joint_arc_lengths)
        for i in range(self.n_canonical_frames):
            joint_positions = self._extract_joint_positions_from_frame(aligned_quat_frames[i])
            is_active = [traj.is_active(arc_length) for traj, arc_length in zip(self.joint_trajectories, joint_arc_lengths)]
            #print is_active
            if np.any(is_active):
                target_joint_positions = [joint_trajectory.query_point_by_absolute_arc_length(arc_length)
                                          for joint_trajectory, arc_length
                                          in zip(self.joint_trajectories, joint_arc_lengths)]
                actual_center = np.average(joint_positions)
                target_center = np.average(target_joint_positions)
                residual_vector[i] = np.linalg.norm(actual_center-target_center)

            if last_joint_positions is not None:
               joint_arc_lengths = [arc_length + np.linalg.norm(np.asarray(joint_position) - np.asarray(last_joint_position))
                                    for joint_position, last_joint_position, arc_length
                                    in zip(joint_positions, last_joint_positions, joint_arc_lengths)]
            last_joint_positions = joint_positions
        print("evaluated constraint set", sum(residual_vector)/len(aligned_quat_frames))
        return residual_vector

    def get_length_of_residual_vector(self):
        return self.n_canonical_frames
