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
Created on Mon Aug 03 18:59:44 2015

@author: Erik Herrmann
"""
import numpy as np
from math import sqrt
from anim_utils.animation_data.utils import convert_quaternion_frame_to_cartesian_frame, align_point_clouds_2D, transform_point_cloud, calculate_point_cloud_distance
from .keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE

class PoseConstraint(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(PoseConstraint, self).__init__(constraint_desc, precision, weight_factor)
        self.skeleton = skeleton
        self.pose_constraint = constraint_desc["frame_constraint"]
        if "velocity_constraint" in constraint_desc:
            self.velocity_constraint = constraint_desc["velocity_constraint"]
        else:
            self.velocity_constraint = None
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE
        self.node_names = constraint_desc["node_names"]
        self.weights = constraint_desc["weights"]

    def evaluate_motion_spline(self, aligned_spline):
        frame1 = aligned_spline.evaluate(self.canonical_keyframe)
        # get point cloud of first two frames
        point_cloud1 = np.array(self.skeleton.convert_quaternion_frame_to_cartesian_frame(frame1, self.node_names))
        frame2 = aligned_spline.evaluate(self.canonical_keyframe+1)
        root_pos2 = self.skeleton.nodes[self.node_names[0]].get_global_position(frame2)
        if self.velocity_constraint is not None:
            velocity = root_pos2 - point_cloud1[0]  # measure only the velocity of the root
            vel_error = np.linalg.norm(self.velocity_constraint - velocity)
        else:
            vel_error = 0.0
        theta, offset_x, offset_z = align_point_clouds_2D(self.pose_constraint,
                                                          point_cloud1,
                                                          self.weights)
        t_point_cloud = transform_point_cloud(point_cloud1, theta, offset_x, offset_z)

        error = calculate_point_cloud_distance(self.pose_constraint, t_point_cloud)
        #print("evaluate pose constraint", error, vel_error)

        return error + vel_error

    def evaluate_motion_sample(self, aligned_quat_frames):
        """ Evaluates the difference between the pose of at the canonical frame of the motion and the pose constraint.

        Parameters
        ----------
        * aligned_quat_frames: np.ndarray
            Motion aligned to previous motion in quaternion format

        Returns
        -------
        * error: float
            Difference to the desired constraint value.
        """

        frame1 = aligned_quat_frames[self.canonical_keyframe]
        # get point cloud of first two frames
        point_cloud1 = np.array(self.skeleton.convert_quaternion_frame_to_cartesian_frame(frame1, self.node_names))
        vel_error = 0
        if self.canonical_keyframe + 1 < len(aligned_quat_frames):
            frame2 = aligned_quat_frames[self.canonical_keyframe + 1]
            root_pos2 = self.skeleton.nodes[self.node_names[0]].get_global_position(frame2)
            velocity = root_pos2-point_cloud1[0]  # measure only the velocity of the root
            vel_error = np.linalg.norm(self.velocity_constraint - velocity)
        theta, offset_x, offset_z = align_point_clouds_2D(self.pose_constraint,
                                                          point_cloud1,
                                                          self.weights)
        t_point_cloud = transform_point_cloud(point_cloud1, theta, offset_x, offset_z)

        error = calculate_point_cloud_distance(self.pose_constraint, t_point_cloud)
        print("evaluate pose constraint", error, vel_error)
        return error + vel_error

    def get_residual_vector_spline(self, aligned_spline):
        return self.get_residual_vector_frame(aligned_spline.evaluate(self.canonical_keyframe))

    def get_residual_vector(self, aligned_quat_frames):
        return self.get_residual_vector_frame(aligned_quat_frames[self.canonical_keyframe])

    def get_residual_vector_frame(self, frame):
        # get point cloud of first frame
        point_cloud = self.skeleton.convert_quaternion_frame_to_cartesian_frame(frame, self.node_names)

        theta, offset_x, offset_z = align_point_clouds_2D(self.pose_constraint,
                                                          point_cloud,
                                                          self.weights)
        t_point_cloud = transform_point_cloud(point_cloud, theta, offset_x, offset_z)
        residual_vector = []
        for i in range(len(t_point_cloud)):
            d = [self.pose_constraint[i][0] - t_point_cloud[i][0],
                 self.pose_constraint[i][1] - t_point_cloud[i][1],
                 self.pose_constraint[i][2] - t_point_cloud[i][2]]
            residual_vector.append(sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))
        return residual_vector

    def get_length_of_residual_vector(self):
        return len(list(self.skeleton.node_name_frame_map.keys()))
