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
import time
import numpy as np
from .algorithm_configuration import DEFAULT_ALGORITHM_CONFIG
from .graph_walk import GraphWalk, GraphWalkEntry
from .graph_walk_optimizer import GraphWalkOptimizer
from .graph_walk_planner import GraphWalkPlanner
from .motion_generator_state import MotionGeneratorState
from .motion_primitive_generator import MotionPrimitiveGenerator
from .scene_interface import SceneInterface
from anim_utils.utilities.log import clear_log, write_message_to_log, LOG_MODE_INFO, LOG_MODE_ERROR
from anim_utils.motion_editing import MotionEditing, MotionGrounding, FootplantConstraintGenerator, add_heels_to_skeleton
from ..constraints import OPTIMIZATION_MODE_ALL
from ..constraints.action_constraints_builder import ActionConstraintsBuilder
from ..constraints.mg_input_format_reader import MGInputFormatReader
from ..constraints.motion_primitive_constraints_builder import MotionPrimitiveConstraintsBuilder
from ..motion_model.motion_state_group import NODE_TYPE_END


class MotionGenerator(object):
    """
    Provides a method to synthesize a motion based on a json input file

    Parameters
    ----------
    * motion_state_graph : MotionStateGraph
        Motion graph structure where nodes represent statistical motion models.
    * algorithm_config : dict
        Contains options for the algorithm.
    * service_config: dict
        Contains paths to the motion data and information about the input and output format.
    """
    def __init__(self, motion_state_graph, service_config, algorithm_config):
        self._service_config = service_config
        self._algorithm_config = algorithm_config
        self._motion_state_graph = motion_state_graph
        self.graph_walk_planner = GraphWalkPlanner(self._motion_state_graph, algorithm_config)
        self.graph_walk = None
        self.action_constraints_builder = ActionConstraintsBuilder(self._motion_state_graph, algorithm_config)
        self.mp_constraints_builder = MotionPrimitiveConstraintsBuilder()
        self.mp_constraints_builder.set_algorithm_config(self._algorithm_config)
        self.end_step_length_factor = self._algorithm_config["trajectory_following_settings"]["end_step_length_factor"]
        self.step_look_ahead_distance = self._algorithm_config["trajectory_following_settings"]["look_ahead_distance"]
        self.activate_global_optimization = False
        self.graph_walk_optimizer = GraphWalkOptimizer(self._motion_state_graph, algorithm_config)
        if "motion_grounding_settings" in list(algorithm_config.keys()):
            motion_grounding_settings = algorithm_config["motion_grounding_settings"]
        else:
            motion_grounding_settings = DEFAULT_ALGORITHM_CONFIG["motion_grounding_settings"]
        skeleton_model = self._motion_state_graph.skeleton.skeleton_model
        self.scene_interface = SceneInterface()
        if skeleton_model is not None and "heel_offset" in skeleton_model:
            self.footplant_constraint_generator = FootplantConstraintGenerator(self._motion_state_graph.skeleton,
                                                                               skeleton_model,
                                                                               motion_grounding_settings,
                                                                               self.scene_interface)
            if skeleton_model["joints"]["left_heel"] not in list(self._motion_state_graph.skeleton.nodes.keys()):
                self._motion_state_graph.skeleton = add_heels_to_skeleton(self._motion_state_graph.skeleton,
                                                                          skeleton_model["joints"]["left_ankle"],
                                                                          skeleton_model["joints"]["right_ankle"],
                                                                          skeleton_model["joints"]["left_heel"],
                                                                          skeleton_model["joints"]["right_heel"],
                                                                          skeleton_model["heel_offset"])
        else:
            self.footplant_constraint_generator = None
        self.set_algorithm_config(algorithm_config)

    def generate_motion(self, mg_input, activate_joint_map, activate_coordinate_transform,
                        complete_motion_vector=True, speed=1.0, prev_graph_walk=None):
        """
        Converts a json input file with a list of elementary actions and constraints
        into a motion vector

        Parameters
        ----------
        * mg_input :  dict
            Dict contains a list of actions with constraints.
        * activate_joint_map: bool
            Maps left hand to left hand endsite and right hand to right hand endsite
        * activate_coordinate_transform: bool
            Converts input coordinates from CAD coordinate system to OpenGL coordinate system
        * complete_motion_vector: bool
            Include fixed degrees of freedom in the returned motion
        * speed: float
            Speed scale factor
        * prev_graph_walk : GraphWalk
            Optional previous graph walk that can be extended

        Returns
        -------
        * motion_vector : AnnotatedMotionVector
           Contains a list of quaternion frames and their annotation based on actions.
        """

        clear_log()
        write_message_to_log("Start motion synthesis", LOG_MODE_INFO)
        mg_input_reader = MGInputFormatReader(self._motion_state_graph, activate_joint_map,
                                              activate_coordinate_transform)

        if not mg_input_reader.read_from_dict(mg_input):
            write_message_to_log("Error: Could not process input constraints", LOG_MODE_ERROR)
            return None

        start_time = time.time()

        start_pose = mg_input_reader.get_start_pose()
        x_offset = start_pose["position"][0]
        z_offset = start_pose["position"][2]
        self.scene_interface.set_offset(x_offset, z_offset)
        offset = mg_input_reader.center_constraints()

        action_constraint_list = self.action_constraints_builder.build_list_from_input_file(mg_input_reader)

        if prev_graph_walk is None:
            self.graph_walk = GraphWalk(self._motion_state_graph, mg_input_reader, self._algorithm_config)
        else:
            self.graph_walk = prev_graph_walk
            self.graph_walk.mg_input = mg_input_reader
            start_action_idx = self.graph_walk.get_number_of_actions()
            action_constraint_list = action_constraint_list[start_action_idx:]

        for constraints in action_constraint_list:
            self._generate_action(constraints)

        time_in_seconds = time.time() - start_time
        write_message_to_log("Finished synthesis in " + str(int(time_in_seconds / 60)) + " minutes "
                             + str(time_in_seconds % 60) + " seconds", LOG_MODE_INFO)

        motion_vector = self.graph_walk.convert_to_annotated_motion(speed)

        self._post_process_motion(motion_vector, complete_motion_vector)

        motion_vector.translate_root(offset)

        return motion_vector

    def _generate_action(self, action_constraints):
        """ Extends the graph walk with an action based on the given constraints.

            Parameters
            ---------
            * action_constraints: ActionConstraints
                Constraints for the action
        """
        self.mp_generator = MotionPrimitiveGenerator(action_constraints, self._algorithm_config)
        self.mp_constraints_builder.set_action_constraints(action_constraints)
        action_state = MotionGeneratorState(self._algorithm_config)
        if action_constraints.root_trajectory is not None:
            max_arc_length = action_constraints.root_trajectory.full_arc_length
        else:
            max_arc_length = np.inf
        action_state.initialize_from_previous_graph_walk(self.graph_walk, max_arc_length, action_constraints.cycled_next)
        arc_length_of_end = self.get_end_step_arc_length(action_constraints)
        optimization_steps = self.graph_walk_optimizer._global_spatial_optimization_steps

        self.graph_walk_planner.set_state(self.graph_walk, self.mp_generator, action_state, action_constraints, arc_length_of_end)
        node_key = self.graph_walk_planner.get_best_start_node()
        self._generate_motion_primitive(action_constraints, node_key, action_state)

        while not action_state.is_end_state():
            self.graph_walk_planner.set_state(self.graph_walk, self.mp_generator, action_state, action_constraints, arc_length_of_end)
            node_key, next_node_type = self.graph_walk_planner.get_best_transition_node()
            self._generate_motion_primitive(action_constraints, node_key, action_state, next_node_type==NODE_TYPE_END)

            if self.activate_global_optimization and action_state.temp_step % optimization_steps == 0:
                start_step = action_state.temp_step - optimization_steps
                self.graph_walk_optimizer.optimize_spatial_parameters_over_graph_walk(self.graph_walk,
                                                                                      self.graph_walk.step_count + start_step)

        self.graph_walk.step_count += action_state.temp_step
        self.graph_walk.update_frame_annotation(action_constraints.action_name,
                                           action_state.action_start_frame, self.graph_walk.get_num_of_frames())


        self.graph_walk = self.graph_walk_optimizer.optimize(self.graph_walk, action_state, action_constraints)
        self.graph_walk.add_entry_to_action_list(action_constraints.action_name,
                                            action_state.start_step, len(self.graph_walk.steps) - 1,
                                            action_constraints)
        write_message_to_log("Reached end of elementary action " + action_constraints.action_name, LOG_MODE_INFO)

    def _generate_motion_primitive(self, action_constraints, node_key, action_state, is_last_step=False):
        """ Extends the graph walk with a motion primitive based on the given constraints.

            Parameters
            ---------
            * action_constraints: ActionConstraints
                Constraints for the action
            * node_key: tuple (string, string)
                Key identifying the motion primitive model
            * action_state: MotionGeneratorState
                Information on the current state of the motion generator
            * is_last_step: bool
                Sets whether or not the motion primitive is an ending state of the current action.
        """
        new_node_type = self._motion_state_graph.nodes[node_key].node_type
        self.mp_constraints_builder.set_status(node_key,
                                               action_state.travelled_arc_length,
                                               self.graph_walk,
                                               is_last_step)
        mp_constraints = self.mp_constraints_builder.build()
        graph_node = self._motion_state_graph.nodes[node_key]
        prev_mp_name = ""
        prev_parameters = None
        if len(self.graph_walk.steps) > 0:
            prev_mp_name = self.graph_walk.steps[-1].node_key[1]
            prev_parameters = self.graph_walk.steps[-1].parameters

        if len(mp_constraints.constraints) > 0:
            new_parameters = self.mp_generator.generate_constrained_sample(graph_node, mp_constraints, prev_mp_name,
                                                                       self.graph_walk.get_quat_frames(),
                                                                       prev_parameters)
        else:
            new_parameters = self.mp_generator.generate_random_sample(node_key, prev_mp_name, prev_parameters)

        motion_spline = self._motion_state_graph.nodes[node_key].back_project(new_parameters, use_time_parameters=False)

        prev_mv_frames = self.graph_walk.get_quat_frames()
        if prev_mv_frames is not None:
            prev_end_point = prev_mv_frames[-1, :3]
        else:
            prev_end_point = np.array([np.inf, np.inf, np.inf])
        print("append", len(self.graph_walk.steps))
        new_mv = motion_spline.get_motion_vector()
        n_new_frames = len(new_mv)
        self.graph_walk.append_quat_frames(new_mv)

        new_travelled_arc_length = 0
        if action_constraints.root_trajectory is not None:
            mv_frames = self.graph_walk.get_quat_frames()
            new_end_point = mv_frames[-1, :3]
            if self.check_overstepping(node_key, action_constraints, new_end_point, prev_end_point) and False:
                self.graph_walk.motion_vector.frames = self.graph_walk.motion_vector.frames[-n_new_frames:]
                self.graph_walk.motion_vector.n_frames -= n_new_frames
                self.graph_walk.motion_vector._prev_n_frames -= n_new_frames
                action_state.overstepped = True
                return

            new_travelled_arc_length = self._update_travelled_arc_length(action_constraints, self.graph_walk.get_quat_frames(),
                                                                         action_state.travelled_arc_length)
        new_step = GraphWalkEntry(self._motion_state_graph, node_key, new_parameters,
                                  new_travelled_arc_length, action_state.step_start_frame,
                                  self.graph_walk.get_num_of_frames() - 1, mp_constraints)

        self.graph_walk.steps.append(new_step)

        action_state.transition(node_key, new_node_type, new_travelled_arc_length, self.graph_walk.get_num_of_frames())


    def check_overstepping(self, node_key, action_constraints, new_end_point, prev_end_point):
        trajectory_end = action_constraints.root_trajectory.get_last_control_point()
        old_distance = np.linalg.norm(trajectory_end - prev_end_point)
        new_distance = np.linalg.norm(trajectory_end - new_end_point)
        average_step_length = self._motion_state_graph.nodes[node_key].average_step_length
        if old_distance < average_step_length and old_distance < new_distance:
            print("overstepped", old_distance, new_distance, prev_end_point, new_end_point, trajectory_end,
                  average_step_length)
            return True
        else:
            return False

    def _post_process_motion(self, motion_vector, complete_motion_vector):
        """
        Applies inverse kinematics constraints on a annotated motion vector and adds values for static DOFs
        that are not part of the motion model.

        Parameters
        ----------
        * motion_vector : AnnotatedMotionVector
            Contains motion but also the constraints
        * complete_motion_vector: bool
            Sets DOFs that are not modelled by the motion model using default values.

        Returns
        -------
        * motion_vector : AnnotatedMotionVector
           Contains a list of quaternion frames and their annotation based on actions.
        """
        ik_settings = self._algorithm_config["inverse_kinematics_settings"]
        has_model = self._motion_state_graph.skeleton.skeleton_model is not None
        if self._algorithm_config["activate_motion_grounding"] and has_model and self.scene_interface is not None and "motion_grounding_settings" in self._algorithm_config:
            self.run_motion_grounding(motion_vector, ik_settings)
            #self.run_motion_grounding(motion_vector, ik_settings)


        if self._algorithm_config["activate_inverse_kinematics"]:
            me = MotionEditing(self._motion_state_graph.skeleton, self._algorithm_config["inverse_kinematics_settings"])
            version = 1
            if "version" in self._algorithm_config["inverse_kinematics_settings"]:
                version = self._algorithm_config["inverse_kinematics_settings"]["version"]

            write_message_to_log("Modify using inverse kinematics"+str(version), LOG_MODE_INFO)
            if version == 1:
                me.modify_motion_vector(motion_vector)
                me.fill_rotate_events(motion_vector)
            elif version == 2:
                me.modify_motion_vector2(motion_vector)

        if complete_motion_vector:
            motion_vector.frames = self._motion_state_graph.skeleton.add_fixed_joint_parameters_to_motion(motion_vector.frames)

    def run_motion_grounding(self, motion_vector, ik_settings):
        grounding_settings = self._algorithm_config["motion_grounding_settings"]
        skeleton_model = self._motion_state_graph.skeleton.skeleton_model
        damp_angle = grounding_settings["damp_angle"] * np.pi
        damp_factor = grounding_settings["damp_factor"]
        grounding = MotionGrounding(self._motion_state_graph.skeleton, ik_settings, skeleton_model,
                                    use_analytical_ik=True, damp_angle=damp_angle, damp_factor=damp_factor)
        if grounding_settings["generate_foot_plant_constraints"] and self.footplant_constraint_generator is not None:
            constraints, blend_ranges, ground_contacts = self.footplant_constraint_generator.generate_from_graph_walk(
                motion_vector)
            motion_vector.grounding_constraints = constraints
            motion_vector.ground_contacts = ground_contacts
            grounding.set_constraints(constraints)
            if grounding_settings["activate_blending"]:
                for target_joint in blend_ranges:
                    joint_list = [skeleton_model["ik_chains"][target_joint]["root"],
                                  skeleton_model["ik_chains"][target_joint]["joint"], target_joint]
                    for frame_range in blend_ranges[target_joint]:
                        grounding.add_blend_range(joint_list, tuple(frame_range))
        grounding.run(motion_vector, self.scene_interface)

    def get_end_step_arc_length(self, action_constraints):
        node_group = action_constraints.get_node_group()
        end_state = node_group.get_random_end_state()
        if end_state is not None:
            arc_length_of_end = self._motion_state_graph.nodes[
                                         end_state].average_step_length * self.end_step_length_factor
        else:
            arc_length_of_end = 0.0

        return arc_length_of_end

    def _update_travelled_arc_length(self, action_constraints, new_quat_frames,  prev_travelled_arc_length):
        """update travelled arc length based on new closest point on trajectory """

        max_arc_length = prev_travelled_arc_length + self.step_look_ahead_distance  # was originally set to 80
        closest_point, distance = action_constraints.root_trajectory.find_closest_point(
            new_quat_frames[-1][:3], prev_travelled_arc_length, max_arc_length)
        new_travelled_arc_length, eval_point = action_constraints.root_trajectory.get_absolute_arc_length_of_point(
            closest_point, min_arc_length=prev_travelled_arc_length)
        if new_travelled_arc_length == -1:
            new_travelled_arc_length = action_constraints.root_trajectory.full_arc_length
        return new_travelled_arc_length

    def get_skeleton(self):
        return self._motion_state_graph.skeleton

    def set_algorithm_config(self, algorithm_config):
        """
        Parameters
        ----------
        * algorithm_config : dict
            Contains options for the algorithm.
        """
        if algorithm_config is None:
            self._algorithm_config = DEFAULT_ALGORITHM_CONFIG
        else:
            self._algorithm_config = algorithm_config
        self.graph_walk_optimizer.set_algorithm_config(self._algorithm_config)
        if "trajectory_following_settings" in list(algorithm_config.keys()):
            trajectory_following_settings = algorithm_config["trajectory_following_settings"]
            self.end_step_length_factor = trajectory_following_settings["end_step_length_factor"]
            self.step_look_ahead_distance = trajectory_following_settings["look_ahead_distance"]
        self.activate_global_optimization = algorithm_config["global_spatial_optimization_mode"] == OPTIMIZATION_MODE_ALL
        self.mp_constraints_builder.set_algorithm_config(self._algorithm_config)
