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
Created on Mon Jul 27 12:00:15 2015

@author: Erik Herrmann
"""
import numpy as np
from anim_utils.utilities.log import write_log, write_message_to_log, LOG_MODE_DEBUG
from .action_constraints import ActionConstraints
from .spatial_constraints import TrajectoryConstraint
from .spatial_constraints.splines.parameterized_spline import ParameterizedSpline
from .spatial_constraints import TrajectorySetConstraint
from ..constraints.mg_input_format_reader import P_KEY, O_KEY, T_KEY
from . import *
from ..constraints.spatial_constraints.splines.utils import get_tangents

REFERENCE_2D_OFFSET = np.array([0.0, -1.0])# components correspond to x, z - we assume the motions are initially oriented into that direction
LEFT_HAND_JOINT = "LeftToolEndSite"
RIGHT_HAND_JOINT = "RightToolEndSite"


class ActionConstraintsBuilder(object):
    """Generates ActionConstraints instances based in an MGInputFormatReader.
    
    Parameters
    ----------
    * mg_input : MGInputFormatReader
        Class to access constraints defined in an input file.
    * motion_state_graph : MotionStateGraph
        Contains a list of motion nodes that can generate short motion clips.
    """
    def __init__(self, motion_state_graph, algorithm_config):
        self.mg_input = None
        self.motion_state_graph = motion_state_graph
        self.default_constraint_weight = 1.0
        self.constraint_precision = 1.0
        self.spline_super_sampling_factor = 20
        self.set_algorithm_config(algorithm_config)

    def set_algorithm_config(self, algorithm_config):
        if "trajectory_following_settings" in list(algorithm_config.keys()):
            trajectory_following_settings = algorithm_config["trajectory_following_settings"]
            self.closest_point_search_accuracy = trajectory_following_settings["closest_point_search_accuracy"]
            self.closest_point_search_max_iterations = trajectory_following_settings["closest_point_search_max_iterations"]
            self.default_spline_type = trajectory_following_settings["spline_type"]
            self.spline_arc_length_parameter_granularity = trajectory_following_settings["arc_length_granularity"]
            self.control_point_distance_threshold = trajectory_following_settings["control_point_filter_threshold"]
            if "spline_supersampling_factor" in list(trajectory_following_settings.keys()):
                self.spline_super_sampling_factor = trajectory_following_settings["spline_super_sampling_factor"]

        self.collision_avoidance_constraints_mode = algorithm_config["collision_avoidance_constraints_mode"]

    def build_list_from_input_file(self, mg_input):
        """
        Returns:
        --------
        * action_constraints : list<ActionConstraints>
          List of constraints for the elementary actions extracted from an input file.
        """
        self.mg_input = mg_input
        self._init_start_pose(mg_input)
        action_constraints_list = []
        for idx in range(self.mg_input.get_number_of_actions()):
            action_constraints_list.append(self._build_action_constraint(idx))

        self._detect_action_cycles(action_constraints_list)
        return action_constraints_list

    def _detect_action_cycles(self, action_constraints_list):
        """detect cycles to adapt transitions
        """
        n_actions = len(action_constraints_list)
        for idx,action_constraints in enumerate(action_constraints_list):
            action_name = action_constraints.action_name
            if self.motion_state_graph.node_groups[action_name].has_cycle_states():
                if idx > 0 and action_constraints_list[idx-1].action_name == action_name:
                    action_constraints.cycled_previous = True
                if idx+1 < n_actions and action_constraints_list[idx+1].action_name == action_name:
                    action_constraints.cycled_next = True
                    write_message_to_log(str(idx) + action_name +" cycle "+ str(n_actions),LOG_MODE_DEBUG)

    def _build_action_constraint(self, action_index):
        action_constraints = ActionConstraints()
        action_constraints.motion_state_graph = self.motion_state_graph
        action_constraints.action_name = self.mg_input.get_elementary_action_name(action_index)
        action_constraints.start_pose = self.get_start_pose()
        action_constraints.group_id = self.mg_input.get_group_id()
        self._add_keyframe_constraints(action_constraints, action_index)
        self._add_keyframe_annotations(action_constraints, action_index)
        self._add_trajectory_constraints(action_constraints, action_index)
        action_constraints._initialized = True
        return action_constraints

    def _init_start_pose(self, mg_input):
        """ Sets the pose at the beginning of the elementary action sequence
            Estimates the optimal start orientation from the constraints if none is given.
        """
        self.start_pose = mg_input.get_start_pose()
        if self.start_pose["orientation"] is None:
            root_trajectories = self._create_trajectory_constraints_for_joint(0, self.motion_state_graph.skeleton.root)
            self.start_pose["orientation"] = [0, 0, 0]
            if len(root_trajectories) > 0:
                if root_trajectories[0] is not None:
                    self.start_pose["orientation"] = self.get_start_orientation_from_trajectory(root_trajectories[0])
            write_message_to_log("Set start orientation from trajectory to"+ str(self.start_pose["orientation"]), LOG_MODE_DEBUG)

    def get_start_pose(self):
        return self.start_pose

    def get_start_orientation_from_trajectory(self, root_trajectory):
        start, tangent, angle = root_trajectory.get_angle_at_arc_length_2d(0.0, REFERENCE_2D_OFFSET)
        return [0, angle, 0]

    def _add_keyframe_annotations(self, action_constraints, index):
        if index > 0:
            action_constraints.prev_action_name = self.mg_input.get_elementary_action_name(index - 1)
        action_constraints.keyframe_annotations = self.mg_input.get_keyframe_annotations(index)

    def _add_keyframe_constraints(self, action_constraints, index):
        node_group = self.motion_state_graph.node_groups[action_constraints.action_name]
        action_constraints.keyframe_constraints = self.mg_input.get_ordered_keyframe_constraints(index, node_group)
        if len(action_constraints.keyframe_constraints) > 0:
            action_constraints.contains_user_constraints = self._has_user_defined_constraints(action_constraints)
            self._merge_two_hand_constraints(action_constraints)
        #print action_constraints.action_name, action_constraints.keyframe_constraints, action_constraints.contains_user_constraints

    def _has_user_defined_constraints(self, action_constraints):
        for keyframe_label_constraints in list(action_constraints.keyframe_constraints.values()):
                if len(keyframe_label_constraints) > 0:
                    if len(keyframe_label_constraints[0]) > 0:
                        return True
        return False

    def _merge_two_hand_constraints(self, action_constraints):
        """ Create a special constraint if two hand joints are constrained on the same keyframe
        """
        for mp_name in list(action_constraints.keyframe_constraints.keys()):
            keyframe_constraints_map = self._map_constraints_by_label(action_constraints.keyframe_constraints[mp_name])
            action_constraints.keyframe_constraints[mp_name], merged_constraints = \
                self._merge_two_hand_constraints_in_keyframe_label_map(keyframe_constraints_map)
            if merged_constraints:
                action_constraints.contains_two_hands_constraints = True

    def _map_constraints_by_label(self, keyframe_constraints):
        """ separate constraints based on keyframe label
        """
        keyframe_constraints_map = dict()
        for desc in keyframe_constraints:
            keyframe_label = desc["semanticAnnotation"]["keyframeLabel"]
            if keyframe_label not in list(keyframe_constraints_map.keys()):
                keyframe_constraints_map[keyframe_label] = list()
            keyframe_constraints_map[keyframe_label].append(desc)
        return keyframe_constraints_map

    def _merge_two_hand_constraints_in_keyframe_label_map(self, keyframe_constraints_map):
        """perform the merging for specific keyframe labels
        """
        merged_constraints = False
        merged_keyframe_constraints = list()
        for keyframe_label in list(keyframe_constraints_map.keys()):
            new_constraint_list, is_merged = self._merge_two_hand_constraint_for_label(keyframe_constraints_map[keyframe_label])
            merged_keyframe_constraints += new_constraint_list
            if is_merged:
                merged_constraints = True
        return merged_keyframe_constraints, merged_constraints

    def _merge_two_hand_constraint_for_label(self, constraint_list):
        left_hand_indices = [index for (index, desc) in enumerate(constraint_list) if desc['joint'] == LEFT_HAND_JOINT]
        right_hand_indices = [index for (index, desc) in enumerate(constraint_list) if desc['joint'] == RIGHT_HAND_JOINT]
        if len(left_hand_indices) == 0 or len(right_hand_indices) == 0:
            #print "did not find two hand keyframe constraint"
            return constraint_list, False

        merged_constraint_list = list()
        left_hand_index = left_hand_indices[0]
        right_hand_index = right_hand_indices[0]
        merged_constraint_list.append(self._create_two_hand_constraint_definition(constraint_list, left_hand_index, right_hand_index))
        merged_constraint_list += [desc for (index, desc) in enumerate(constraint_list)
                                        if index != left_hand_index and index != right_hand_index]
        return merged_constraint_list, True

    def _create_two_hand_constraint_definition(self, constraint_list, left_hand_index, right_hand_index):
        joint_names = [LEFT_HAND_JOINT, RIGHT_HAND_JOINT]
        positions = [constraint_list[left_hand_index][P_KEY],
                     constraint_list[right_hand_index][P_KEY]]
        orientations = [constraint_list[left_hand_index][O_KEY],
                        constraint_list[right_hand_index][O_KEY]]
        time = constraint_list[left_hand_index][T_KEY]
        semantic_annotation = constraint_list[left_hand_index]["semanticAnnotation"]
        merged_constraint_desc = {"joint": joint_names,
                                   "positions": positions,
                                   "orientations": orientations,
                                   "time": time,
                                   "merged": True,
                                   "semanticAnnotation": semantic_annotation}
        #print "merged keyframe constraint", merged_constraint_desc
        return merged_constraint_desc

    def _create_root_trajectory(self, action_index):
        root_trajectory = None
        root_trajectories = self._create_trajectory_constraints_for_joint(action_index,
                                                                          self.motion_state_graph.skeleton.aligning_root_node)

        if len(root_trajectories) > 0:
           root_trajectory = root_trajectories[0]
        return root_trajectory

    def _add_trajectory_constraints(self, action_constraints, action_index):
        """ Extracts the root_trajectory if it is found and trajectories for other joints.
            If semanticAnnotation is found they are treated as collision avoidance constraint.
        """
        action_constraints.trajectory_constraints = list()
        action_constraints.collision_avoidance_constraints = list()
        action_constraints.annotated_trajectory_constraints = list()
        action_constraints.root_trajectory = self._create_root_trajectory(action_index)
        self._add_joint_trajectory_constraints(action_constraints, action_index)

    def _add_joint_trajectory_constraints(self, action_constraints, action_index):
        for joint_name in list(self.motion_state_graph.skeleton.nodes.keys()):
            if joint_name != self.motion_state_graph.skeleton.root:
                self._add_trajectory_constraint(action_constraints, action_index, joint_name)
        if self.collision_avoidance_constraints_mode == CA_CONSTRAINTS_MODE_SET and len(
                action_constraints.collision_avoidance_constraints) > 0:
            self._add_ca_trajectory_constraint_set(action_constraints)

    def _add_trajectory_constraint(self, action_constraints, action_index, joint_name):
        trajectory_constraints = self._create_trajectory_constraints_for_joint(action_index, joint_name, add_tangents=False)
        for c in trajectory_constraints:
            if c is not None:
                if c.is_collision_avoidance_constraint:
                    action_constraints.collision_avoidance_constraints.append(c)
                if c.semantic_annotation is not None:
                    action_constraints.annotated_trajectory_constraints.append(c)
                else:
                    action_constraints.trajectory_constraints.append(c)

    def _add_ca_trajectory_constraint_set(self, action_constraints):
        if action_constraints.root_trajectory is not None:
           joint_trajectories = [action_constraints.root_trajectory] + action_constraints.collision_avoidance_constraints
           joint_names = [action_constraints.root_trajectory.joint_name] + [traj.joint_name for traj in joint_trajectories]
        else:
           joint_trajectories = action_constraints.collision_avoidance_constraints
           joint_names = [traj.joint_name for traj in joint_trajectories]

        action_constraints.ca_trajectory_set_constraint = TrajectorySetConstraint(joint_trajectories, joint_names,
                                                                                  self.motion_state_graph.skeleton,
                                                                                  self.constraint_precision,
                                                                                  self.default_constraint_weight)

    def _create_trajectory_constraints_for_joint(self, action_index, joint_name, add_tangents=True):
        """ Create a spline based on a trajectory constraint definition read from the input file.
            Components containing None are set to 0, but marked as ignored in the unconstrained_indices list.
            Note all elements in constraints_list must have the same dimensions constrained and unconstrained.

        Returns
        -------
        * trajectory: List(TrajectoryConstraint)
        \t The trajectory constraints defined by the control points from the
            trajectory_constraint or an empty list if there is no constraint
        """
        distance_threshold = 0.0
        if add_tangents:
            distance_threshold = self.control_point_distance_threshold
        desc = self.mg_input.extract_trajectory_desc(action_index, joint_name, distance_treshold=distance_threshold)
        control_points_list = desc["control_points_list"]
        if len(control_points_list) > 0 and len(control_points_list[0][P_KEY]) > 0:
            control_points = control_points_list[0]
            if add_tangents:
                #orientations = complete_orientations_from_tangents(control_points[P_KEY], control_points[O_KEY])
                #orientations = complete_tangents(control_points[P_KEY], control_points[O_KEY])

                supersampling_size = self.spline_super_sampling_factor*len(control_points)
                points, orientations = get_tangents(control_points[P_KEY], supersampling_size)
                if control_points[O_KEY][-1] is not None:
                    orientations[-1] = control_points[O_KEY][-1]
                traj_constraint = TrajectoryConstraint(joint_name, points, orientations,
                                                   self.default_spline_type, 0.0,
                                                   desc["unconstrained_indices"],
                                                   self.motion_state_graph.skeleton,
                                                   self.constraint_precision, self.default_constraint_weight,
                                                   self.closest_point_search_accuracy,
                                                   self.closest_point_search_max_iterations,
                                                   self.spline_arc_length_parameter_granularity)

                return [traj_constraint]
            else:
                print(joint_name, control_points[P_KEY])
                traj_constraint = TrajectoryConstraint(joint_name, control_points[P_KEY],None,
                                                       self.default_spline_type, 0.0,
                                                       desc["unconstrained_indices"],
                                                       self.motion_state_graph.skeleton,
                                                       self.constraint_precision, self.default_constraint_weight,
                                                       self.closest_point_search_accuracy,
                                                       self.closest_point_search_max_iterations,
                                                       self.spline_arc_length_parameter_granularity)

                return [traj_constraint]
        else:
            return []

