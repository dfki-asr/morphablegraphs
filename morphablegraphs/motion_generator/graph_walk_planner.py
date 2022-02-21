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
import random
from copy import copy, deepcopy
import numpy as np
from anim_utils.animation_data.utils import create_transformation_matrix
from anim_utils.utilities.log import write_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR
from ..constraints.motion_primitive_constraints import MotionPrimitiveConstraints
from ..constraints.spatial_constraints.keyframe_constraints import Direction2DConstraint
from ..constraints.spatial_constraints.keyframe_constraints import GlobalTransformConstraint
from .graph_walk import GraphWalk, GraphWalkEntry
from ..motion_model import NODE_TYPE_END


class PlannerState(object):
    def __init__(self, current_node, graph_walk, travelled_arc_length, overstepped=False):
        self.graph_walk = graph_walk
        self.travelled_arc_length = travelled_arc_length
        self.current_node = current_node
        self.overstepped = overstepped

    def create_copy(self):
        current_node = copy(self.current_node)
        travelled_arc_length = self.travelled_arc_length
        graph_walk = GraphWalk(self.graph_walk.motion_state_graph, self.graph_walk.mg_input, self.graph_walk._algorithm_config)
        graph_walk.motion_vector.frames = [deepcopy(self.graph_walk.motion_vector.frames[-1])]
        return PlannerState(current_node, graph_walk, travelled_arc_length, self.overstepped)


class GraphWalkPlanner(object):
    def __init__(self, motion_state_graph,  algorithm_config):
        self.motion_state_graph = motion_state_graph
        trajectory_following_settings = algorithm_config["trajectory_following_settings"]
        self.step_look_ahead_distance = trajectory_following_settings["look_ahead_distance"]

        if "constrain_start_orientation" in list(trajectory_following_settings.keys()):
            self.constrain_start_orientation = trajectory_following_settings["constrain_start_orientation"]
        else:
            self.constrain_start_orientation = True

        if "constrain_transition_orientation" in list(trajectory_following_settings.keys()):
            self.constrain_transition_orientation = trajectory_following_settings["constrain_transition_orientation"]
        else:
            self.constrain_transition_orientation = False

        if "generate_half_step_constraint" in list(trajectory_following_settings.keys()):
            self.generate_half_step_constraint = trajectory_following_settings["generate_half_step_constraint"]
        else:
            self.generate_half_step_constraint = False

        self.use_local_coordinates = algorithm_config["use_local_coordinates"]
        self.mp_generator = None
        self.state = None
        self.action_constraints = None
        self.arc_length_of_end = 0.0
        self.node_group = None
        self.trajectory = None
        self._n_steps_looking_ahead = 1
        self._n_option_eval_samples = 10

    def set_state(self, graph_walk, mp_generator, action_state, action_constraints, arc_length_of_end):
        self.mp_generator = mp_generator
        self.state = PlannerState(action_state.current_node, graph_walk, action_state.travelled_arc_length, action_state.overstepped)
        self.action_constraints = action_constraints
        self.trajectory = action_constraints.root_trajectory
        self.arc_length_of_end = arc_length_of_end
        self.node_group = self.action_constraints.get_node_group()

    def get_best_start_node(self):
        start_nodes = self.motion_state_graph.get_start_nodes(self.action_constraints.action_name)
        if len(start_nodes) > 1:
            options = [(self.action_constraints.action_name, next_node) for next_node in start_nodes]
            return self.select_next_step(self.state, options, add_orientation=self.constrain_start_orientation)
        else:
            return self.action_constraints.action_name, start_nodes[0]

    def get_transition_options(self, state):
        if self.trajectory is not None:
            if state.overstepped:
                next_node_type = NODE_TYPE_END
            else:
                next_node_type = self.node_group.get_transition_type_for_action_from_trajectory(state.graph_walk,
                                                                                            self.action_constraints,
                                                                                            state.travelled_arc_length,
                                                                                            self.arc_length_of_end)
        else:
            next_node_type = self.node_group.get_transition_type_for_action(state.graph_walk, self.action_constraints)

        edges = self.motion_state_graph.nodes[self.state.current_node].outgoing_edges
        options = [edge_key for edge_key in list(edges.keys()) if edges[edge_key].transition_type == next_node_type]
        #print "options",next_node_type, options
        return options, next_node_type

    def get_best_transition_node(self):
        options, next_node_type = self.get_transition_options(self.state)
        n_transitions = len(options)
        if n_transitions == 1:
            next_node = options[0]
        elif n_transitions > 1:
            if self.trajectory is not None:
                next_node = self.select_next_step(self.state, options, add_orientation=self.constrain_transition_orientation)
            else:  # use random transition if there is no path to follow
                random_index = random.randrange(0, n_transitions, 1)
                next_node = options[random_index]
        else:
            n_outgoing_edges = len(self.motion_state_graph.nodes[self.state.current_node].outgoing_edges)
            write_message_to_log("Error: Could not find a transition from state " + str(self.state.current_node)
                                 + " " + str(n_outgoing_edges))
            next_node = self.node_group.get_random_start_state()
            next_node_type = self.motion_state_graph.nodes[next_node].node_type
        if next_node is None:
            write_message_to_log("Error: Could not find a transition of type " + next_node_type +
                                 " from state " + str(self.state.current_node))
        return next_node, next_node_type

    def _add_constraint_with_orientation(self, constraint_desc, goal_arc_length, mp_constraints):
        goal_position = self.trajectory.query_point_by_absolute_arc_length(goal_arc_length).tolist()
        tangent = self.trajectory.query_orientation_by_absolute_arc_length(goal_arc_length)
        tangent /= np.linalg.norm(tangent)

        constraint_desc["position"] = goal_position
        pos_constraint = GlobalTransformConstraint(self.motion_state_graph.skeleton, constraint_desc, 1.0, 1.0)
        mp_constraints.constraints.append(pos_constraint)
        dir_constraint_desc = {"joint": self.motion_state_graph.skeleton.aligning_root_node, "canonical_keyframe": -1, "dir_vector": tangent,
                               "semanticAnnotation": {"keyframeLabel": "end", "generated": True}}
        # TODO add weight to configuration
        dir_constraint = Direction2DConstraint(self.motion_state_graph.skeleton, dir_constraint_desc, 1.0, 1.0)
        mp_constraints.constraints.append(dir_constraint)

    def _add_constraint(self, constraint_desc, goal_arc_length, mp_constraints):
        constraint_desc["position"] = self.trajectory.query_point_by_absolute_arc_length(goal_arc_length).tolist()
        pos_constraint = GlobalTransformConstraint(self.motion_state_graph.skeleton, constraint_desc, 1.0, 1.0)
        mp_constraints.constraints.append(pos_constraint)

    def _generate_node_evaluation_constraints(self, state, add_orientation=False):
        joint = self.motion_state_graph.skeleton.aligning_root_node
        goal_arc_length = state.travelled_arc_length + self.step_look_ahead_distance

        mp_constraints = MotionPrimitiveConstraints()
        mp_constraints.skeleton = self.motion_state_graph.skeleton
        start_pose = state.graph_walk.motion_vector.start_pose
        mp_constraints.aligning_transform = create_transformation_matrix(start_pose["position"], start_pose["orientation"])
        mp_constraints.start_pose = state.graph_walk.motion_vector.start_pose
        constraint_desc = {"joint": joint, "canonical_keyframe": -1, "n_canonical_frames": 0,
                           "semanticAnnotation": {"keyframeLabel": "end", "generated": True}}
        # the canonical keyframe is updated per option based on the keyframeLabel
        half_step_constraint_desc = {"joint": joint, "canonical_keyframe": -1, "n_canonical_frames": 0,
                                     "semanticAnnotation": {"keyframeLabel": "middle", "generated": True}}

        if add_orientation:
            self._add_constraint_with_orientation(constraint_desc, goal_arc_length, mp_constraints)
        else:
            self._add_constraint(constraint_desc, goal_arc_length, mp_constraints)

        if self.generate_half_step_constraint:
            half_goal_arc_length = state.travelled_arc_length + self.step_look_ahead_distance / 2
            self._add_constraint(half_step_constraint_desc, half_goal_arc_length, mp_constraints)

        if self.use_local_coordinates and False:
            mp_constraints = mp_constraints.transform_constraints_to_local_cos()

        return mp_constraints

    def select_next_step(self, state, options, add_orientation=False):
        #next_node = self._look_one_step_ahead(state, options, add_orientation)
        mp_constraints = self._generate_node_evaluation_constraints(state, add_orientation)
        if state.current_node is None or True:
            errors, s_vectors = self._evaluate_options(state, mp_constraints, options)
        else:
            errors, s_vectors = self._evaluate_options_looking_ahead(state, mp_constraints, options, add_orientation)
        min_idx = np.argmin(errors)
        next_node = options[min_idx]
        write_message_to_log("####################################Next node is" +str(next_node), LOG_MODE_DEBUG)
        return next_node

    def _evaluate_option(self, node_name, mp_constraints, prev_frames):
        motion_primitive_node = self.motion_state_graph.nodes[node_name]
        canonical_keyframe = motion_primitive_node.get_n_canonical_frames() - 1
        for c in mp_constraints.constraints:
            if c.keyframe_label == "end":
                c.canonical_keyframe = canonical_keyframe
            elif c.keyframe_label == "middle":
                c.canonical_keyframe = canonical_keyframe/2

        if motion_primitive_node.cluster_tree is not None:

            s_vector = self.mp_generator._get_best_fit_sample_using_cluster_tree(motion_primitive_node, mp_constraints,
                                                                             prev_frames, 1)
        else:
            samples = motion_primitive_node.sample_low_dimensional_vectors(self._n_option_eval_samples)
            s_vector, error = self.mp_generator.evaluate_samples_using_constraints(samples, motion_primitive_node, mp_constraints,
                                                                             prev_frames)
        write_message_to_log("Evaluated option " + str(node_name) + str(mp_constraints.min_error), LOG_MODE_DEBUG)
        return s_vector, mp_constraints.min_error

    def _evaluate_options(self, state, mp_constraints, options):
        errors = np.empty(len(options))
        s_vectors = []
        index = 0
        for node_name in options:
            #print "option", node_name
            s_vector, error = self._evaluate_option(node_name, mp_constraints, state.graph_walk.motion_vector.frames)
            errors[index] = error
            s_vectors.append(s_vector)
            index += 1
        return errors, s_vectors

    def _evaluate_options_looking_ahead(self, state, mp_constraints, options, add_orientation=False):
        errors = np.empty(len(options))
        next_node = options[0]
        #TODO add state fork
        index = 0
        for node_name in options:
            print("evaluate",node_name)
            node_state = state.create_copy()
            step_count = 0

            while step_count < self._n_steps_looking_ahead:
                s_vector, error = self._evaluate_option(node_name, mp_constraints, state.graph_walk.motion_vector.frames)
                #write_log("Evaluated option", node_name, mp_constraints.min_error,"at level", n_steps)
                errors[index] += error
                self._update_path(node_state, node_name, s_vector)
                print("advance along",node_name)
                #if node_state.current_node is not None:
                #    new_options, next_node_type = self.get_transition_options(node_state)
                    #errors[index] += self._look_ahead_deeper(node_state, new_options, self._n_steps_looking_ahead, add_orientation)
                step_count += 1

            index += 1
        min_idx = np.argmin(errors)
        next_node = options[min_idx]
        return next_node

    def _look_ahead_deeper(self, state, options, max_depth, add_orientation=False):
        #print "#####################################Look deeper from", state.current_node
        mp_constraints = self._generate_node_evaluation_constraints(state, add_orientation)
        errors, s_vectors = self._evaluate_options(state, mp_constraints, options)
        min_idx = np.argmin(errors)
        error = errors[min_idx]
        self._update_path(state, options[min_idx], s_vectors[min_idx])
        if max_depth > 0:
            new_options, next_node_type = self.get_transition_options(state)
            error += self._look_ahead_deeper(state, new_options, max_depth-1, add_orientation)
        return error

    def _update_path(self, state, next_node, s_vector):
        motion_spline = self.motion_state_graph.nodes[next_node].back_project(s_vector, use_time_parameters=False)
        state.graph_walk.append_quat_frames(motion_spline.get_motion_vector())
        max_arc_length = state.travelled_arc_length + self.step_look_ahead_distance  # was originally set to 80
        closest_point, distance = self.trajectory.find_closest_point(state.graph_walk.motion_vector.frames[-1][:3],state.travelled_arc_length,max_arc_length)
        new_travelled_arc_length, eval_point = self.trajectory.get_absolute_arc_length_of_point(closest_point, min_arc_length=state.travelled_arc_length)
        if new_travelled_arc_length == -1:
            new_travelled_arc_length = self.trajectory.full_arc_length
        state.travelled_arc_length = new_travelled_arc_length
        new_step = GraphWalkEntry(self.motion_state_graph, next_node, s_vector, new_travelled_arc_length, 0, 0, None)
        state.graph_walk.steps.append(new_step)
        state.current_node = next_node

