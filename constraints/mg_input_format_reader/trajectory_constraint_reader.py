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
import math
from anim_utils.utilities.log import write_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_ERROR, LOG_MODE_INFO
from transformations import quaternion_from_euler, euler_from_quaternion, euler_matrix
from .utils import _transform_point_from_cad_to_opengl_cs, _transform_unconstrained_indices_from_cad_to_opengl_cs
from .constants import *

DISTANCE_WARNING = "Warning: shift second to last control point because it is too close to the last control point"


def _init_active_region(traj_constraint):
    if "semanticAnnotation" in list(traj_constraint[0].keys()):
        active_region = dict()
        active_region["start_point"] = None
        active_region["end_point"] = None
        return active_region
    else:
        return None


def _end_active_region(active_region, control_points):
    if active_region["start_point"] is None:
        active_region["start_point"] = control_points[0]
    if active_region["end_point"] is None:
        active_region["end_point"] = control_points[-1]


def _update_active_region(active_region, point, new_active):
    if new_active and active_region["start_point"] is None:
        active_region["start_point"] = point
    elif not new_active and active_region["start_point"] is not None and active_region["end_point"] is None:
        active_region["end_point"] = point


def _is_active_trajectory_region(control_points, index):
    if "semanticAnnotation" in list(control_points[index].keys()):
        if "collisionAvoidance" in list(control_points[index]["semanticAnnotation"].keys()):
            return control_points[index]["semanticAnnotation"]["collisionAvoidance"]
    return True


class TrajectoryConstraintReader(object):
    def __init__(self, activate_coordinate_transform=True, scale_factor=1.0):
        self.activate_coordinate_transform = activate_coordinate_transform
        self.scale_factor = scale_factor

    def _filter_control_points_simple(self, control_points, distance_threshold=0.0):
        filtered_control_points = {P_KEY: list(), O_KEY: list()}
        previous_point = None
        n_control_points = len(control_points)
        last_distance = None
        for idx in range(n_control_points):
            result = self._filter_control_point(control_points, n_control_points, idx, previous_point,
                                                last_distance, distance_threshold)
            if result is not None:
                position, orientation, last_distance = result
                #n_points = len(filtered_control_points)
                #if idx == n_control_points - 1:
                #    last_added_point_idx = n_points - 1
                #    delta = filtered_control_points[P_KEY][last_added_point_idx] - position
                #    if np.linalg.norm(delta) < distance_threshold:
                #        filtered_control_points[last_added_point_idx][P_KEY] += delta
                #        write_log(DISTANCE_WARNING)
                filtered_control_points[P_KEY].append(position)
                filtered_control_points[O_KEY].append(orientation)
                previous_point = position
        return filtered_control_points

    def _filter_control_points_ca(self, control_points, distance_threshold=-1):
        filtered_control_points = list()
        active_regions = list()
        previous_point = None
        n_control_points = len(control_points)
        was_active = False
        last_distance = None
        count = -1
        for idx in range(n_control_points):
            is_active = _is_active_trajectory_region(control_points, idx)
            if not is_active:
                was_active = is_active
                continue
            if not was_active and is_active:
                active_region = _init_active_region(control_points)
                filtered_control_points.append(list())
                active_regions.append(active_region)
                count += 1
            if count < 0:
                continue
            tmp_distance_threshold = distance_threshold
            if active_regions[count] is not None:
                tmp_distance_threshold = -1
            result = self._filter_control_point(control_points, n_control_points, idx, previous_point,
                                                last_distance, tmp_distance_threshold)
            if result is None:
                continue
            else:
                point, orientation, last_distance = result
                n_points = len(filtered_control_points[count])
                if idx == n_control_points - 1:
                    last_added_point_idx = n_points - 1
                    delta = filtered_control_points[count][last_added_point_idx] - point
                    if np.linalg.norm(delta) < distance_threshold:
                        filtered_control_points[count][last_added_point_idx] += delta
                        write_log(
                            "Warning: shift second to last control point because it is too close to the last control point")
                filtered_control_points[count].append(point)

                if active_regions[count] is not None:
                    _update_active_region(active_regions[count], point, is_active)
                previous_point = point
                was_active = is_active

        # handle invalid region specification
        region_points = list()
        for idx in range(len(filtered_control_points)):
            region_points.append(len(filtered_control_points[idx]))
            if active_regions[idx] is not None:
                if len(filtered_control_points[idx]) < 2:
                    filtered_control_points[idx] = None
                else:
                    _end_active_region(active_regions[idx], filtered_control_points[idx])
        # print "loaded", len(control_point_list),"active regions with",region_points,"points"
        return filtered_control_points, active_regions

    def _filter_control_point(self, control_points, n_control_points, index, previous_point, last_distance,
                              distance_threshold):

        control_point = control_points[index]
        if P_KEY not in list(control_point.keys()) or control_point[P_KEY] == [None, None, None]:
            write_log("Warning: skip undefined control point")
            return None

        # set component of the position to 0 where it is is set to None to allow a 3D spline definition
        position = control_point[P_KEY]
        point = [p * self.scale_factor if p is not None else 0 for p in position]
        point = np.asarray(_transform_point_from_cad_to_opengl_cs(point, self.activate_coordinate_transform))
        if previous_point is not None and np.linalg.norm(point - previous_point) < 0.001:
            return None

        if O_KEY in list(control_point.keys()) and None not in control_point[O_KEY]:
            #q = quaternion_from_euler(*np.radians(control_point[O_KEY]))
            ref_vector = [0, 0, 1, 1]
            m = euler_matrix(*np.radians(control_point[O_KEY]))
            orientation = np.dot(m, ref_vector)#[:3]


            #ref_vector = [0, 1]
            #angle = np.radians(control_point[O_KEY][1])
            #sa = math.sin(angle)
            #ca = math.cos(angle)
            #m = np.array([[ca, -sa], [sa, ca]])
            #orientation = np.dot(m, ref_vector)
            orientation /= np.linalg.norm(orientation)
            orientation = np.array([orientation[0],0,orientation[2]])
        else:
            orientation = None
        #orientation = None

        if previous_point is None or index == n_control_points - 1:
            return point, orientation, last_distance
        else:
            # add the point if there is no distance threshold or if it is the first point,
            #  or if it is the last point or larger than or equal to the distance threshold
            distance = np.linalg.norm(point - previous_point)
            if distance_threshold > 0.0 and distance < distance_threshold:
                return None
            if last_distance is not None and distance < last_distance / 10.0:  # TODO add toggle of filter to config
                return None
            return point, orientation, distance

    def _extract_control_point_list(self, action_desc, joint_name):
        control_point_list = None
        for c in action_desc[CONSTRAINTS_KEY]:
            if "joint" in list(c.keys()) and TRAJECTORY_CONSTRAINTS_KEY in list(c.keys()) and joint_name == c["joint"]:
                control_point_list = c[TRAJECTORY_CONSTRAINTS_KEY]
                break  # there should only be one list per joint and elementary action
        return control_point_list

    def _find_semantic_annotation(self, control_points):
        semantic_annotation = None
        for p in control_points:
            if "semanticAnnotation" in list(p.keys()) and not "collisionAvoidance" in list(p["semanticAnnotation"].keys()):
                semantic_annotation = p["semanticAnnotation"]
                break
        return semantic_annotation

    def _find_unconstrained_indices(self, trajectory_constraint_data):
        """extract unconstrained dimensions"""
        unconstrained_indices = list()
        idx = 0
        for p in trajectory_constraint_data:
            if [P_KEY] in list(p.keys()):
                for v in p[P_KEY]:
                    if v is None:
                        unconstrained_indices.append(idx)
                    idx += 1
                break # check only one point
        return _transform_unconstrained_indices_from_cad_to_opengl_cs(unconstrained_indices, self.activate_coordinate_transform)

    def _check_for_collision_avoidance_annotation(self, trajectory_constraint_desc, control_points):
        """ find start and end control point of an active region if there exists one.
        Note this functions expects that there is not more than one active region.

        :param trajectory_constraint_desc:
        :param control_points:
        :return: dict containing "start_point" and "end_point" or None
        """
        assert len(trajectory_constraint_desc) == len(control_points), str(len(trajectory_constraint_desc)) +" != " +  str(  len(control_points))
        active_region = None
        if "semanticAnnotation" in list(trajectory_constraint_desc[0].keys()):
            active_region = dict()
            active_region["start_point"] = None
            active_region["end_point"] = None
            c_index = 0
            for c in trajectory_constraint_desc:
                if "semanticAnnotation" in list(c.keys()):
                    if c["semanticAnnotation"]["collisionAvoidance"]:
                        active_region["start_point"] = control_points[c_index]
                    elif active_region["start_point"] is not None and active_region["end_point"] is None:
                        active_region["end_point"] = control_points[c_index]
                        break
                c_index += 1
        return active_region

    def create_trajectory_from_control_points(self, control_points, distance_threshold=-1):
        desc = dict()
        desc["control_points_list"] = []
        desc["orientation_list"] = []
        desc["active_regions"] = []
        desc["semantic_annotation"] = self._find_semantic_annotation(control_points)
        desc["unconstrained_indices"] = self._find_unconstrained_indices(control_points)
        desc["control_points_list"] = [self._filter_control_points_simple(control_points, distance_threshold)]
        return desc

    def extract_trajectory_desc(self, elementary_action_list, action_index, joint_name, distance_threshold=-1):
        """ Extract the trajectory information from the constraint list
        Returns:
        -------
        * desc : dict
        \tConstraint definition that contains a list of control points, unconstrained_indices, active_regions and a possible
        annotation.
        """

        control_points = self._extract_control_point_list(elementary_action_list[action_index], joint_name)
        if control_points is not None:
            return self.create_trajectory_from_control_points(control_points, distance_threshold)
        return {"control_points_list": []}
