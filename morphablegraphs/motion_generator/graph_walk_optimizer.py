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

from copy import deepcopy
import numpy as np
from anim_utils.utilities.log import write_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR
from ..constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE, SPATIAL_CONSTRAINT_TYPE_TRAJECTORY, SPATIAL_CONSTRAINT_TYPE_TRAJECTORY_SET, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT
from ..constraints.time_constraints_builder import TimeConstraintsBuilder
from .optimization.optimizer_builder import OptimizerBuilder
from ..constraints.motion_primitive_constraints import MotionPrimitiveConstraints

GRAPH_WALK_OPTIMIZATION_TWO_HANDS = "none"
GRAPH_WALK_OPTIMIZATION_ALL = "all"
GRAPH_WALK_OPTIMIZATION_TWO_HANDS = "two_hands"
GRAPH_WALK_OPTIMIZATION_END_POINT = "trajectory_end"
CONSTRAINT_FILTER_LIST = [SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE, SPATIAL_CONSTRAINT_TYPE_TRAJECTORY, SPATIAL_CONSTRAINT_TYPE_TRAJECTORY_SET, SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT]


class GraphWalkOptimizer(object):
    def __init__(self, motion_primitive_graph, algorithm_config):
        self.motion_primitive_graph = motion_primitive_graph
        self.time_error_minimizer = OptimizerBuilder(algorithm_config).build_time_error_minimizer()
        self.global_error_minimizer = OptimizerBuilder(algorithm_config).build_global_error_minimizer_residual()
        self.collision_avoidance_error_minimizer = OptimizerBuilder(algorithm_config).build_spatial_error_minimizer()
        self.set_algorithm_config(algorithm_config)

    def set_algorithm_config(self, algorithm_config):
        self._algorithm_config = algorithm_config
        self.spatial_mode = algorithm_config["global_spatial_optimization_mode"]
        self.optimize_collision_avoidance_constraints_extra = algorithm_config["optimize_collision_avoidance_constraints_extra"]
        self._global_spatial_optimization_steps = algorithm_config["global_spatial_optimization_settings"]["max_steps"]
        self._position_weight_factor = algorithm_config["global_spatial_optimization_settings"]["position_weight"]
        self._orientation_weight_factor = algorithm_config["global_spatial_optimization_settings"]["orientation_weight"]
        self.optimized_actions_for_time_constraints = algorithm_config["global_time_optimization_settings"]["optimized_actions"]

    def _is_optimization_required(self, action_constraints):
        return self.spatial_mode == GRAPH_WALK_OPTIMIZATION_ALL and action_constraints.contains_user_constraints or \
               self.spatial_mode == GRAPH_WALK_OPTIMIZATION_TWO_HANDS and action_constraints.contains_two_hands_constraints

    def optimize(self, graph_walk, action_state, action_constraints):
        if self._is_optimization_required(action_constraints):
            start_step = max(action_state.start_step - self._global_spatial_optimization_steps, 0)
            message = " ".join(map(str, ["Start spatial graph walk optimization at", start_step, "looking back", self._global_spatial_optimization_steps, "steps"]))
            write_message_to_log(message, LOG_MODE_DEBUG)
            graph_walk = self.optimize_spatial_parameters_over_graph_walk(graph_walk, start_step)

        elif self.spatial_mode == GRAPH_WALK_OPTIMIZATION_END_POINT and action_constraints.root_trajectory is not None:
            start_step = max(len(graph_walk.steps) - self._global_spatial_optimization_steps, 0)
            message = " ".join(map(str, ["Start spatial graph walk optimization at", start_step, "looking back", self._global_spatial_optimization_steps, "steps"]))
            write_message_to_log(message, LOG_MODE_DEBUG)
            graph_walk = self.optimize_spatial_parameters_over_graph_walk(graph_walk, start_step)

        if self.optimize_collision_avoidance_constraints_extra and action_constraints.collision_avoidance_constraints is not None and len(action_constraints.collision_avoidance_constraints) > 0 :
            write_message_to_log("Optimize collision avoidance parameters", LOG_MODE_DEBUG)
            graph_walk = self.optimize_for_collision_avoidance_constraints(graph_walk, action_constraints, action_state.start_step)
        return graph_walk

    def optimize_spatial_parameters_over_graph_walk(self, graph_walk, start_step=0):
        initial_guess = graph_walk.get_global_spatial_parameter_vector(start_step)
        constraint_count = self._filter_constraints(graph_walk, start_step)
        self._adapt_constraint_weights(graph_walk, start_step)
        if constraint_count > 0:
            if start_step == 0:
                prev_frames = None
            else:
                prev_frames = graph_walk.get_quat_frames()[:graph_walk.steps[start_step].start_frame]
            data = (self.motion_primitive_graph, graph_walk.steps[start_step:],
                    self._algorithm_config["global_spatial_optimization_settings"]["error_scale_factor"],
                    self._algorithm_config["global_spatial_optimization_settings"]["quality_scale_factor"],
                    prev_frames, 1.0)
            # init_error_sum = 10000
            init_error_sum = max(abs(np.sum(self.global_error_minimizer._objective_function(initial_guess, data))), 1.0)
            write_message_to_log("Sum of errors" + str(init_error_sum), LOG_MODE_DEBUG)
            data = (self.motion_primitive_graph, graph_walk.steps[start_step:],
                    self._algorithm_config["global_spatial_optimization_settings"]["error_scale_factor"],
                    self._algorithm_config["global_spatial_optimization_settings"]["quality_scale_factor"],
                    prev_frames, init_error_sum)

            self.global_error_minimizer.set_objective_function_parameters(data)
            optimal_parameters = self.global_error_minimizer.run(initial_guess)
            graph_walk.update_spatial_parameters(optimal_parameters, start_step)
            graph_walk.convert_graph_walk_to_quaternion_frames(start_step, use_time_parameters=False)
        else:
            write_message_to_log("No user defined constraints", LOG_MODE_INFO)
        return graph_walk

    def _filter_constraints(self, graph_walk, start_step):
        constraint_count = 0
        for step in graph_walk.steps[start_step:]: #TODO add pose constraint for pick and place
            reduced_constraints = []
            for constraint in step.motion_primitive_constraints.constraints:
                if constraint.constraint_type not in CONSTRAINT_FILTER_LIST:
                     reduced_constraints.append(constraint)
            step.motion_primitive_constraints.constraints = reduced_constraints
            #initial_guess += step.parameters[:step.n_spatial_components].tolist()
            constraint_count += len(step.motion_primitive_constraints.constraints)
        return constraint_count

    def _adapt_constraint_weights(self, graph_walk, start_step):
        if self.spatial_mode == GRAPH_WALK_OPTIMIZATION_ALL or self.spatial_mode == GRAPH_WALK_OPTIMIZATION_TWO_HANDS:
            for step in graph_walk.steps[start_step:]:
                for constraint in step.motion_primitive_constraints.constraints:
                    if not "generated" in list(constraint.semantic_annotation.keys()):
                        constraint.weight_factor = self._position_weight_factor
        else: # self.spatial_mode == GRAPH_WALK_OPTIMIZATION_END_POINT
             for constraint in graph_walk.steps[-1].motion_primitive_constraints.constraints:
                 if constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
                     constraint.weight_factor = self._position_weight_factor
                 elif constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D:
                     constraint.weight_factor = self._orientation_weight_factor

    def optimize_time_parameters_over_graph_walk(self, graph_walk):
        for idx, ea in enumerate(graph_walk.elementary_action_list):
            #print "evaluate", ea.action_name,ea.start_step,ea.end_step
            prev_action_idx = max(idx - (self.optimized_actions_for_time_constraints-1), 0)
            start_step = graph_walk.elementary_action_list[prev_action_idx].start_step
            end_step = ea.end_step
            time_constraints = TimeConstraintsBuilder(graph_walk, start_step, end_step).build(self.motion_primitive_graph, graph_walk)
            if time_constraints is not None:
                data = (self.motion_primitive_graph, graph_walk, time_constraints,
                        self._algorithm_config["global_time_optimization_settings"]["error_scale_factor"],
                        self._algorithm_config["global_time_optimization_settings"]["quality_scale_factor"])
                self.time_error_minimizer.set_objective_function_parameters(data)
                #initial_guess = graph_walk.get_global_time_parameter_vector(start_step)
                initial_guess = time_constraints.get_initial_guess(graph_walk)
                #print "initial_guess",prev_action_idx, time_constraints.start_step, end_step, initial_guess, time_constraints.constraint_list
                optimal_parameters = self.time_error_minimizer.run(initial_guess)
                #print "result ",optimal_parameters
                graph_walk.update_time_parameters(optimal_parameters, start_step, end_step)
                #graph_walk.convert_graph_walk_to_quaternion_frames(start_step, use_time_parameters=True)
            else:
                write_message_to_log("No time constraints for action "+str(idx), LOG_MODE_DEBUG)

        return graph_walk

    def optimize_for_collision_avoidance_constraints(self, graph_walk, action_constraints, start_step=0):
        reduced_motion_vector = deepcopy(graph_walk.motion_vector)
        reduced_motion_vector.reduce_frames(graph_walk.steps[start_step].start_frame)
        write_message_to_log("start frame " + str(graph_walk.steps[start_step].start_frame), LOG_MODE_DEBUG)
        step_index = start_step
        n_steps = len(graph_walk.steps)
        print(reduced_motion_vector.n_frames, graph_walk.get_num_of_frames(), reduced_motion_vector.n_frames - graph_walk.get_num_of_frames())
        while step_index < n_steps:
            node = self.motion_primitive_graph.nodes[graph_walk.steps[step_index].node_key]
            print(graph_walk.steps[step_index].node_key, node.n_canonical_frames, graph_walk.steps[step_index].start_frame)
            motion_primitive_constraints = MotionPrimitiveConstraints()
            active_constraint = False
            for trajectory in action_constraints.collision_avoidance_constraints:
                if reduced_motion_vector.frames is not None:
                    trajectory.set_min_arc_length_from_previous_frames(reduced_motion_vector.frames)
                else:
                    trajectory.min_arc_length = 0.0
                trajectory.set_number_of_canonical_frames(node.n_canonical_frames)
                #discrete_trajectory = trajectory.create_discrete_trajectory(original_frames[step.start_frame:step.end_frame])
                motion_primitive_constraints.constraints.append(trajectory)
                active_constraint = True
            if active_constraint:
                data = (node, motion_primitive_constraints, reduced_motion_vector.frames,
                        self._algorithm_config["local_optimization_settings"]["error_scale_factor"],
                        self._algorithm_config["local_optimization_settings"]["quality_scale_factor"])
                self.collision_avoidance_error_minimizer.set_objective_function_parameters(data)
                graph_walk.steps[step_index].parameters = self.collision_avoidance_error_minimizer.run(graph_walk.steps[step_index].parameters)
            motion_primitive_sample = node.back_project(graph_walk.steps[step_index].parameters, use_time_parameters=False)
            reduced_motion_vector.append_quat_frames(motion_primitive_sample.get_motion_vector())
            step_index += 1
        write_message_to_log("start frame " + str(step_index) + " "+ str(len(graph_walk.steps)), LOG_MODE_DEBUG)
        assert (len(graph_walk.motion_vector.frames)) == len(reduced_motion_vector.frames), (str(len(graph_walk.motion_vector.frames))) + "," + str(len(reduced_motion_vector.frames))
        graph_walk.motion_vector = reduced_motion_vector
        return graph_walk
