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
Created on Tue Jul 14 18:39:41 2015

@author: Erik Herrmann
"""

from datetime import datetime
import collections
import json
import numpy as np
from anim_utils.animation_data import MotionVector, align_quaternion_frames
from anim_utils.animation_data.motion_concatenation import align_and_concatenate_frames
from anim_utils.utilities.log import write_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR
from .annotated_motion_vector import AnnotatedMotionVector
from ..constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION
from .keyframe_event_list import KeyframeEventList
from ..constraints.spatial_constraints.splines.utils import plot_annotated_spline

DEFAULT_PLACE_ACTION_LIST = ["placeRight", "placeLeft","insertRight","insertLeft","screwRight", "screwLeft"] #list of actions in which the orientation constraints are ignored


class GraphWalkEntry(object):
    def __init__(self, motion_state_graph, node_key, parameters, arc_length, start_frame, end_frame, motion_primitive_constraints=None):
        self.node_key = node_key
        self.parameters = parameters
        self.arc_length = arc_length
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.motion_primitive_constraints = motion_primitive_constraints
        self.n_spatial_components = motion_state_graph.nodes[node_key].get_n_spatial_components()
        self.n_time_components = motion_state_graph.nodes[node_key].get_n_time_components()

    @staticmethod
    def from_json(motion_state_graph, data):
        return GraphWalkEntry(motion_state_graph, tuple(data["node_key"]),
                              np.array(data["parameters"]), data["arc_length"],
                              data["start_frame"], data["end_frame"])

    def to_json(self):
        data = dict()
        data["node_key"] =self.node_key
        data["parameters"] = self.parameters.tolist()
        data["arc_length"] = self.arc_length
        data["start_frame"] = self.start_frame
        data["end_frame"] = self.end_frame
        return data


class HighLevelGraphWalkEntry(object):
    def __init__(self, action_name, start_step, end_step, action_constraints):
        self.action_name = action_name
        self.start_step = start_step
        self.end_step = end_step
        self.action_constraints = action_constraints


class GraphWalk(object):
    """ Product of the MotionGenerate class. Contains the graph walk used to generate the frames,
        a mapping of frame segments
        to elementary actions and a list of events for certain frames.
    """
    def __init__(self, motion_state_graph, mg_input, algorithm_config, start_pose=None, create_ca_vis_data=False):
        self.elementary_action_list = []
        self.steps = []
        self.motion_state_graph = motion_state_graph
        self.step_count = 0
        self.mg_input = mg_input
        self._algorithm_config = algorithm_config
        self.motion_vector = MotionVector(self.motion_state_graph.skeleton, algorithm_config)
        if start_pose is None:
            start_pose = mg_input.get_start_pose()
        self.motion_vector.start_pose = start_pose

        smoothing_settings = algorithm_config["smoothing_settings"]
        self.spatial_smoothing_method = "smoothing"
        self.apply_smoothing = smoothing_settings["spatial_smoothing"] # set whether the exported motion is smoothed at transitions
        if "spatial_smoothing_method" in smoothing_settings:
            self.spatial_smoothing_method = smoothing_settings["spatial_smoothing_method"]

        self.motion_vector.apply_spatial_smoothing = False # deactivate smoothing during the synthesis
        self.use_time_parameters = algorithm_config["activate_time_variation"]
        self.constrain_place_orientation = algorithm_config["inverse_kinematics_settings"]["constrain_place_orientation"]
        write_message_to_log("Use time parameters" + str(self.use_time_parameters), LOG_MODE_DEBUG)
        self.keyframe_event_list = KeyframeEventList(create_ca_vis_data)
        self.place_action_list = DEFAULT_PLACE_ACTION_LIST

    def add_entry_to_action_list(self, action_name, start_step, end_step, action_constraints):
        self.elementary_action_list.append(HighLevelGraphWalkEntry(action_name, start_step, end_step, action_constraints))

    def convert_to_annotated_motion(self, step_size=1.0):
        self.motion_vector.apply_spatial_smoothing = self.apply_smoothing # set wether or not smoothing is applied
        self.motion_vector.spatial_smoothing_method = self.spatial_smoothing_method
        self.convert_graph_walk_to_quaternion_frames(use_time_parameters=self.use_time_parameters, step_size=step_size)
        self.keyframe_event_list.update_events(self, 0)
        annotated_motion_vector = AnnotatedMotionVector(self.motion_state_graph.skeleton, self._algorithm_config)
        annotated_motion_vector.frames = self.motion_vector.frames
        annotated_motion_vector.n_frames = self.motion_vector.n_frames
        annotated_motion_vector.frame_time = self.motion_state_graph.skeleton.frame_time
        annotated_motion_vector.keyframe_event_list = self.keyframe_event_list
        annotated_motion_vector.skeleton = self.motion_state_graph.skeleton
        annotated_motion_vector.mg_input = self.mg_input
        version = 0
        if "version" in self._algorithm_config["inverse_kinematics_settings"]:
            version = self._algorithm_config["inverse_kinematics_settings"]["version"]
        if version == 1:
            annotated_motion_vector.ik_constraints = self._create_ik_constraints()
        elif version == 2:
            annotated_motion_vector.ik_constraints = self._create_ik_constraints2()
        annotated_motion_vector.graph_walk = self
        return annotated_motion_vector

    def get_action_from_keyframe(self, keyframe):
        found_action_index = -1
        step_index = self.get_step_from_keyframe(keyframe)
        write_message_to_log("Found keyframe in step " + str(step_index), LOG_MODE_DEBUG)
        if step_index < 0:
            return found_action_index
        for action_index, action in enumerate(self.elementary_action_list):
            if action.start_step <= step_index <= action.end_step:
                found_action_index = action_index
        return found_action_index

    def get_step_from_keyframe(self, keyframe):
        found_step_index = -1
        for step_index, step in enumerate(self.steps):
            #Note the start_frame and end_frame are warped in update_temp_motion_vector
            #print step.start_frame, keyframe, step.end_frame
            if step.start_frame <= keyframe <= step.end_frame:
                found_step_index = step_index
        return found_step_index

    def convert_graph_walk_to_quaternion_frames(self, start_step=0, use_time_parameters=False, step_size=1.0):
        """
        :param start_step:
        :return:
        """
        if start_step == 0:
            start_frame = 0
        else:
            start_frame = self.steps[start_step].start_frame
        self.motion_vector.clear(end_frame=start_frame)
        for step in self.steps[start_step:]:
            step.start_frame = start_frame
            #write_log(step.node_key, len(step.parameters))
            quat_frames = self.motion_state_graph.nodes[step.node_key].back_project(step.parameters, use_time_parameters, step_size).get_motion_vector()
            if step.node_key[1].lower().endswith("leftstance"):
                foot_joint = "foot_r"
            elif step.node_key[1].lower().endswith("rightstance"):
                foot_joint = "foot_l"
            else:
                foot_joint = None
            self.motion_vector.append_frames(quat_frames, foot_joint)
            step.end_frame = self.get_num_of_frames()-1
            start_frame = step.end_frame + 1

    def get_global_spatial_parameter_vector(self, start_step=0):
        initial_guess = []
        for step in self.steps[start_step:]:
            initial_guess += step.parameters[:step.n_spatial_components].tolist()
        return initial_guess

    def get_global_time_parameter_vector(self, start_step=0):
        initial_guess = []
        for step in self.steps[start_step:]:
            initial_guess += step.parameters[step.n_spatial_components:].tolist()
        return initial_guess

    def update_spatial_parameters(self, parameter_vector, start_step=0):
        write_message_to_log("Update spatial parameters", LOG_MODE_DEBUG)
        offset = 0
        for step in self.steps[start_step:]:
            new_alpha = parameter_vector[offset:offset+step.n_spatial_components]
            step.parameters[:step.n_spatial_components] = new_alpha
            offset += step.n_spatial_components

    def update_time_parameters(self, parameter_vector, start_step, end_step):
        offset = 0
        for step in self.steps[start_step:end_step]:
            new_gamma = parameter_vector[offset:offset+step.n_time_components]
            step.parameters[step.n_spatial_components:] = new_gamma
            offset += step.n_time_components

    def append_quat_frames(self, new_frames):
        self.motion_vector.append_frames(new_frames)

    def get_quat_frames(self):
        return self.motion_vector.frames

    def get_num_of_frames(self):
        return self.motion_vector.n_frames

    def update_frame_annotation(self, action_name, start_frame, end_frame):
        """ Adds a dictionary to self.frame_annotation marking start and end
            frame of an action.
        """
        self.keyframe_event_list.update_frame_annotation(action_name, start_frame, end_frame)

    def _create_ik_constraints(self):
        ik_constraints = []
        for idx, action in enumerate(self.elementary_action_list):
            write_message_to_log("Create IK constraints for action" + " " + str(idx) + " " + str(action.start_step) + " " + str(self.steps[action.start_step].start_frame), LOG_MODE_DEBUG)
            if not self.constrain_place_orientation and action.action_name in self.place_action_list:
                constrain_orientation = False
            else:
                constrain_orientation = True
            start_step = action.start_step
            end_step = action.end_step
            elementary_action_ik_constraints = dict()
            elementary_action_ik_constraints["keyframes"] = dict()
            elementary_action_ik_constraints["trajectories"] = list()
            elementary_action_ik_constraints["collision_avoidance"] = list()
            frame_offset = self.steps[start_step].start_frame
            for step in self.steps[start_step: end_step+1]:
                time_function = None
                if self.use_time_parameters and self.motion_state_graph.nodes[step.node_key].get_n_time_components() > 0:
                    time_function = self.motion_state_graph.nodes[step.node_key].back_project_time_function(step.parameters)

                step_keyframe_constraints = step.motion_primitive_constraints.convert_to_ik_constraints(self.motion_state_graph, frame_offset, time_function, constrain_orientation)

                elementary_action_ik_constraints["keyframes"].update(step_keyframe_constraints)
                elementary_action_ik_constraints["collision_avoidance"] += step.motion_primitive_constraints.get_ca_constraints()

                frame_offset += step.end_frame - step.start_frame + 1

            if self._algorithm_config["collision_avoidance_constraints_mode"] == "ik":
                elementary_action_ik_constraints["trajectories"] += self._create_ik_trajectory_constraints_from_ca_trajectories(idx)
            elementary_action_ik_constraints["trajectories"] += self._create_ik_trajectory_constraints_from_annotated_trajectories(idx)
            ik_constraints.append(elementary_action_ik_constraints)
        return ik_constraints

    def _create_ik_constraints2(self):
        ik_constraints = collections.OrderedDict()
        for idx, action in enumerate(self.elementary_action_list):
            write_message_to_log("Create IK constraints for action" + " " + str(idx) + " " + str(action.start_step) + " " + str(self.steps[action.start_step].start_frame), LOG_MODE_DEBUG)
            if not self.constrain_place_orientation and action.action_name in self.place_action_list:
                constrain_orientation = False
            else:
                constrain_orientation = True
            start_step = action.start_step
            end_step = action.end_step
            frame_offset = self.steps[start_step].start_frame
            for step in self.steps[start_step: end_step + 1]:
                time_function = None
                if self.use_time_parameters and self.motion_state_graph.nodes[step.node_key].get_n_time_components() > 0:
                    time_function = self.motion_state_graph.nodes[step.node_key].back_project_time_function(step.parameters)

                step_constraints = step.motion_primitive_constraints.convert_to_ik_constraints(
                    self.motion_state_graph, frame_offset, time_function, constrain_orientation, version=2)
                ik_constraints.update(step_constraints)

                frame_offset += step.end_frame - step.start_frame + 1

        return ik_constraints

    def _create_ik_trajectory_constraints_from_ca_trajectories(self, action_idx):
        frame_annotation = self.keyframe_event_list.frame_annotation['elementaryActionSequence'][action_idx]
        trajectory_constraints = list()
        action = self.elementary_action_list[action_idx]
        for ca_constraint in action.action_constraints.collision_avoidance_constraints:
            traj_constraint = dict()
            traj_constraint["trajectory"] = ca_constraint
            traj_constraint["fixed_range"] = False  # search for closer start
            traj_constraint["constrain_orientation"] = False
            traj_constraint["start_frame"] = frame_annotation["startFrame"]
            traj_constraint["end_frame"] = frame_annotation["endFrame"]
            #TODO find a better solution than this workaround that undoes the joint name mapping from hands to tool bones for ca constraints
            if self.mg_input.activate_joint_mapping and ca_constraint.joint_name in list(self.mg_input.inverse_joint_name_map.keys()):
                joint_name = self.mg_input.inverse_joint_name_map[ca_constraint.joint_name]
            else:
                joint_name = ca_constraint.joint_name

            traj_constraint["joint_name"] = joint_name
            traj_constraint["delta"] = 1.0
            trajectory_constraints.append(traj_constraint)
        return trajectory_constraints

    def _create_ik_trajectory_constraints_from_annotated_trajectories(self, action_idx):
        write_message_to_log("extract annotated trajectories", LOG_MODE_DEBUG)
        frame_annotation = self.keyframe_event_list.frame_annotation['elementaryActionSequence'][action_idx]
        start_frame = frame_annotation["startFrame"]
        trajectory_constraints = list()
        action = self.elementary_action_list[action_idx]
        for constraint in action.action_constraints.annotated_trajectory_constraints:
            label = list(constraint.semantic_annotation.keys())[0]
            write_message_to_log("trajectory constraint label " + str(list(constraint.semantic_annotation.keys())), LOG_MODE_DEBUG)
            action_name = action.action_name
            for step in self.steps[action.start_step: action.end_step+1]:
                motion_primitive_name = step.node_key[1]
                write_message_to_log("look for action annotation of " + action_name+" "+motion_primitive_name, LOG_MODE_DEBUG)

                if motion_primitive_name not in self.motion_state_graph.node_groups[action_name].motion_primitive_annotation_regions:
                    continue
                annotations = self.motion_state_graph.node_groups[action_name].motion_primitive_annotation_regions[motion_primitive_name]
                write_message_to_log("action annotation" + str(annotations) +" "+  str(frame_annotation["startFrame"]) + " " + str(frame_annotation["endFrame"]),
                                     LOG_MODE_DEBUG)

                if label not in list(annotations.keys()):
                    continue
                annotation_range = annotations[label]
                traj_constraint = dict()
                traj_constraint["trajectory"] = constraint
                traj_constraint["constrain_orientation"] = True
                traj_constraint["fixed_range"] = True
                time_function = None
                if self.use_time_parameters and self.motion_state_graph.nodes[step.node_key].get_n_time_components() > 0:
                    time_function = self.motion_state_graph.nodes[step.node_key].back_project_time_function(step.parameters)
                if time_function is None:
                    traj_constraint["start_frame"] = start_frame + annotation_range[0]
                    traj_constraint["end_frame"] = start_frame + annotation_range[1]
                else:
                    #add +1 for correct mapping TODO verify for all cases
                    traj_constraint["start_frame"] = start_frame + int(time_function[annotation_range[0]]) + 1
                    traj_constraint["end_frame"] = start_frame + int(time_function[annotation_range[1]]) + 1

                if self.mg_input.activate_joint_mapping and constraint.joint_name in list(self.mg_input.inverse_joint_name_map.keys()):
                    joint_name = self.mg_input.inverse_joint_name_map[constraint.joint_name]
                else:
                    joint_name = constraint.joint_name

                traj_constraint["joint_name"] = joint_name
                traj_constraint["delta"] = 1.0
                write_message_to_log( "create ik trajectory constraint from label " + str(label), LOG_MODE_DEBUG)
                trajectory_constraints.append(traj_constraint)
        return trajectory_constraints

    def get_average_keyframe_constraint_error(self):
        keyframe_constraint_errors = []
        step_index = 0
        prev_frames = None
        for step_idx, step in enumerate(self.steps):
            quat_frames = self.motion_state_graph.nodes[step.node_key].back_project(step.parameters, use_time_parameters=False).get_motion_vector()
            skeleton = self.motion_vector.skeleton
            aligned_frames = align_and_concatenate_frames(skeleton, skeleton.aligning_root_node, quat_frames, prev_frames,
                                         self.motion_vector.start_pose, 0)
            for c_idx, constraint in enumerate(step.motion_primitive_constraints.constraints):
                if constraint.constraint_type in [SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION] and\
                   not "generated" in list(constraint.semantic_annotation.keys()):
                    error = constraint.evaluate_motion_sample(aligned_frames)
                    write_message_to_log("Error of Keyframe constraint " +str(step_idx) + "-" + str(c_idx) +": " +str(error), LOG_MODE_DEBUG)
                    keyframe_constraint_errors.append(error)
            prev_frames = aligned_frames
            step_index += 1
        if len(keyframe_constraint_errors) > 0:
            return np.average(keyframe_constraint_errors)
        else:
            return -1

    def get_generated_constraints(self):
        step_count = 0
        generated_constraints = dict()
        for step in self.steps:
            key = str(step.node_key) + str(step_count)
            generated_constraints[key] = []
            for constraint in step.motion_primitive_constraints.constraints:
                if constraint.is_generated():
                    generated_constraints[key].append(constraint.position)
            step_count += 1
        return generated_constraints

    def get_average_error(self):
        average_error = 0
        for step in self.steps:
            average_error += step.motion_primitive_constraints.min_error
        if average_error > 0:
            average_error /= len(self.steps)
        return average_error

    def get_number_of_object_evaluations(self):
        objective_evaluations = 0
        for step in self.steps:
            objective_evaluations += step.motion_primitive_constraints.evaluations
        return objective_evaluations

    def print_statistics(self):
        print(self.get_statistics_string())

    def get_statistics_string(self):
        average_error = self.get_average_error()
        evaluations_string = "Total number of objective evaluations " + str(self.get_number_of_object_evaluations())
        error_string = "Average error for " + str(len(self.steps)) + \
                       " motion primitives: " + str(average_error)
        average_keyframe_error = self.get_average_keyframe_constraint_error()
        if average_keyframe_error > -1:
            average_keyframe_error_string = "Average keyframe constraint error " + str(average_keyframe_error)
        else:
            average_keyframe_error_string = "No keyframe constraint specified"
        average_time_per_step = 0.0
        for step in self.steps:
            average_time_per_step += step.motion_primitive_constraints.time
        average_time_per_step /= len(self.steps)
        average_time_string = "Average time per motion primitive " + str(average_time_per_step)
        return average_keyframe_error_string + "\n" + evaluations_string + "\n" + average_time_string + "\n" + error_string

    def export_generated_constraints(self, file_path="goals.path"):
        """ Converts constraints that were generated based on input constraints into a json dictionary for a debug visualization
        """
        root_control_point_data = []
        hand_constraint_data = []
        for idx, step in enumerate(self.steps):
            step_constraints = {"semanticAnnotation": {"step": idx}}
            for c in step.motion_primitive_constraints.constraints:
                if c.constraint_type == "keyframe_position" and c.joint_name == self.motion_state_graph.skeleton.root:
                    p = c.position
                    if p is not None:
                        step_constraints["position"] = [p[0], -p[2], None]
                elif c.constraint_type == "keyframe_2d_direction":
                        step_constraints["direction"] = c.direction_constraint.tolist()
                elif c.constraint_type == "ca_constraint":
                    #if c.constraint_type in ["RightHand", "LeftHand"]:
                    position = [c.position[0], -c.position[2], c.position[1]]
                    hand_constraint = {"position": position}
                    hand_constraint_data.append(hand_constraint)
            root_control_point_data.append(step_constraints)


        constraints = {"tasks": [{"elementaryActions":[{
                                                      "action": "walk",
                                                      "constraints": [{"joint": "Hips",
                                                                       "keyframeConstraints": root_control_point_data  },
                                                                      {"joint": "RightHand",
                                                                       "keyframeConstraints": hand_constraint_data}]
                                                      }]
                                 }]
                       }

        constraints["startPose"] = {"position":[0,0,0], "orientation": [0,0,0]}
        constraints["session"] = "session"
        with open(file_path, "wb") as out:
            json.dump(constraints, out)

    def get_number_of_actions(self):
        return len(self.elementary_action_list)

    def plot_constraints(self, file_name="traj"):
        for idx, action in enumerate(self.elementary_action_list):
            start_frame = self.steps[action.start_step].start_frame
            end_frame = self.steps[action.end_step].end_frame

            root_motion = self.motion_vector.frames#[start_frame:end_frame,:3]
            if action.action_constraints.root_trajectory is not None:
                traj_constraint = action.action_constraints.root_trajectory
                plot_annotated_spline(traj_constraint,root_motion, file_name+str(idx)+".png")

    def to_json(self):
        data = dict()
        data["algorithm_config"] = self._algorithm_config
        data["start_pose"] = self.motion_vector.start_pose
        data["steps"] = []
        for step in self.steps:
            data["steps"].append(step.to_json())
        return data

    @staticmethod
    def from_json(graph, data):
        graph_walk = GraphWalk(graph, None, data["algorithm_config"], data["start_pose"])
        graph_walk.steps = []
        for step_data in data["steps"]:
            graph_walk.steps.append(GraphWalkEntry.from_json(graph, step_data))
        return graph_walk

    def save_to_file(self, file_path):
        with open(file_path, "wb") as out:
            json.dump(self.to_json(), out)
