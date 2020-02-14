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
Created on Mon Jul 27 18:38:15 2015

@author: Erik Herrmann
"""
from copy import copy
import numpy as np
import collections
from .motion_primitive_constraints import MotionPrimitiveConstraints
from .spatial_constraints import PoseConstraint, GlobalTransformConstraint, PoseConstraintQuatFrame, TwoHandConstraintSet, LookAtConstraint, FeetConstraint
from anim_utils.animation_data.motion_concatenation import align_and_concatenate_frames, get_transform_from_start_pose, get_node_aligning_2d_transform
from . import OPTIMIZATION_MODE_ALL, OPTIMIZATION_MODE_KEYFRAMES, OPTIMIZATION_MODE_TWO_HANDS
from .spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
from ..motion_model.action_meta_info import KEYFRAME_LABEL_END, KEYFRAME_LABEL_START, KEYFRAME_LABEL_MIDDLE
from .keyframe_event import KeyframeEvent
from .locomotion_constraints_builder import LocomotionConstraintsBuilder

DEFAULT_IMPORTANT_JOINT_LIST = ["pelvis", "right_wrist", "left_wrist", "right_ankle", "left_ankle"]

class MotionPrimitiveConstraintsBuilder(object):
    """ Extracts a list of constraints for a motion primitive from ElementaryActionConstraints
        based on the variables set by the method set_status. Generates constraints for path following.
    """

    mp_constraint_types = ["position", "orientation", "time"]

    def __init__(self):
        self.action_constraints = None
        self.algorithm_config = None
        self.status = {}
        self.motion_state_graph = None
        self.node_group = None
        self.skeleton = None
        self.precision = {"pos": 1.0, "rot": 1.0, "smooth": 1.0}
        self.trajectory_following_settings = dict()
        self.local_optimization_mode = "None"
        self.ca_constraint_mode = "None"
        self.use_local_coordinates = False
        self.use_transition_constraint = False
        self.generate_half_step_constraint = False
        self.pose_constraint_node_names = None
        self.locomotion_constraint_builder = None

    def set_action_constraints(self, action_constraints):
        self.action_constraints = action_constraints
        self.motion_state_graph = action_constraints.motion_state_graph
        self.node_group = self.action_constraints.get_node_group()
        self.skeleton = action_constraints.motion_state_graph.skeleton

        if self.skeleton.skeleton_model is not None:
            important_joints = []
            joint_map = self.skeleton.skeleton_model["joints"]
            for j in DEFAULT_IMPORTANT_JOINT_LIST:
                if j in joint_map:
                    important_joints.append(joint_map[j])
            self.pose_constraint_node_names = important_joints
        else:
            self.pose_constraint_node_names = list(self.skeleton.joint_weight_map.keys())

        self.locomotion_constraint_builder = LocomotionConstraintsBuilder(self.skeleton, self, self.trajectory_following_settings)

    def set_algorithm_config(self, algorithm_config):
        self.algorithm_config = algorithm_config
        self.trajectory_following_settings = algorithm_config["trajectory_following_settings"]
        self.local_optimization_mode = algorithm_config["local_optimization_mode"]
        self.ca_constraint_mode = algorithm_config["collision_avoidance_constraints_mode"]
        self.use_local_coordinates = algorithm_config["use_local_coordinates"]
        self.use_mgrd = algorithm_config["constrained_sampling_mode"] == "random_spline"
        self.use_transition_constraint = self.trajectory_following_settings["use_transition_constraint"]
        if "generate_half_step_constraint" in list(self.trajectory_following_settings.keys()):
            self.generate_half_step_constraint = self.trajectory_following_settings["generate_half_step_constraint"]

        if self.locomotion_constraint_builder is not None:
            self.locomotion_constraint_builder.set_algorithm_config(self.trajectory_following_settings)

    def set_status(self, node_key, last_arc_length, graph_walk, is_last_step=False):
        n_prev_frames = graph_walk.get_num_of_frames()
        prev_frames = graph_walk.get_quat_frames()
        n_canonical_frames = self.motion_state_graph.nodes[node_key].get_n_canonical_frames()
        #create a sample to estimate the trajectory arc lengths
        mp_sample_frames = self.motion_state_graph.nodes[node_key].sample(False).get_motion_vector()
        if self.use_local_coordinates:
            aligned_sample_frames = align_and_concatenate_frames(self.skeleton, self.skeleton.aligning_root_node,
                                                                       mp_sample_frames, prev_frames,
                                                                       graph_walk.motion_vector.start_pose,
                                                                       0)
            self.status["aligned_sample_frames"] = aligned_sample_frames[n_prev_frames:]
        self.status["action_name"] = node_key[0]
        self.status["motion_primitive_name"] = node_key[1]
        self.status["n_canonical_frames"] = n_canonical_frames
        self.status["last_arc_length"] = last_arc_length # defined in actionstate.transition() based on the closest point on the path
        self.status["n_prev_frames"] = n_prev_frames

        if prev_frames is None:
            last_pos = self.action_constraints.start_pose["position"]

        else:
            last_pos = prev_frames[-1][:3]
        last_pos = copy(last_pos)
        last_pos[1] = 0.0
        self.status["last_pos"] = last_pos
        self.status["prev_frames"] = prev_frames
        self.status["is_last_step"] = is_last_step
        if self.use_mgrd or self.use_local_coordinates:
            self._set_aligning_transform(node_key, prev_frames)
        else:
            self.status["aligning_transform"] = None

    def _set_aligning_transform(self, node_key, prev_frames):
        if prev_frames is None:
            self.status["aligning_transform"] = get_transform_from_start_pose(self.action_constraints.start_pose)
        else:
            sample = self.motion_state_graph.nodes[node_key].sample(False)
            frames = sample.get_motion_vector()
            self.status["aligning_transform"] = get_node_aligning_2d_transform(self.skeleton, self.skeleton.aligning_root_node, prev_frames, frames)

    def build(self):
        if self.use_local_coordinates:
            start_pose = None
        else:
            start_pose = self.action_constraints.start_pose
        mp_constraints = MotionPrimitiveConstraints.from_dict(self.skeleton, self.status,
                                                              self.trajectory_following_settings,
                                                              self.precision,
                                                              start_pose)
        if self.action_constraints.root_trajectory is not None:
            node_key = (self.action_constraints.action_name, self.status["motion_primitive_name"])
            self.locomotion_constraint_builder.add_constraints(mp_constraints, node_key,
                                                               self.action_constraints.root_trajectory,
                                                               self.status["last_arc_length"],
                                                               self.status["is_last_step"])
            if self.use_transition_constraint:
                self._add_pose_constraint(mp_constraints)
        if len(self.action_constraints.keyframe_constraints.keys()) > 0:
            self._add_keyframe_constraints(mp_constraints)
            # generate frame constraints for the last step based on the previous state
            # if not already done for the trajectory following
            if self.status["is_last_step"] and not mp_constraints.pose_constraint_set:
                self._add_pose_constraint(mp_constraints)
        if mp_constraints.action_name in ["pickBoth","placeBoth"] and mp_constraints.motion_primitive_name == "reach":
            self._add_feet_constraint(mp_constraints)
        self._add_trajectory_constraints(mp_constraints)
        self._add_events_to_event_list(mp_constraints)
        self._decide_on_optimization(mp_constraints)
        return mp_constraints

    def _add_trajectory_constraints(self, mp_constraints):
        for trajectory_constraint in self.action_constraints.trajectory_constraints:
            # set the previous arc length as new min arc length
            if self.status["prev_frames"] is not None:
                trajectory_constraint.set_min_arc_length_from_previous_frames(self.status["prev_frames"])
                trajectory_constraint.set_number_of_canonical_frames(self.status["n_canonical_frames"])
            mp_constraints.constraints.append(trajectory_constraint)

    def _add_feet_constraint(self, mp_constraints):
        if "LeftFoot" in list(self.skeleton.nodes.keys()) and "RightFoot" in list(self.skeleton.nodes.keys()):
            left_position = self.skeleton.nodes["LeftFoot"].get_global_position(self.status["prev_frames"][-1])
            right_position = self.skeleton.nodes["RightFoot"].get_global_position(self.status["prev_frames"][-1])
            desc = {"left":left_position, "right": right_position}
            desc["semanticAnnotation"] = {}
            desc["semanticAnnotation"]["keyframeLabel"] = "end"
            desc["canonical_keyframe"] = self._get_keyframe_from_label("end")
            feet_constraint = FeetConstraint(self.skeleton, desc, 1.0, 2.0)
            mp_constraints.constraints.append(feet_constraint)

    def _add_pose_constraint(self, mp_constraints):
        if mp_constraints.settings["transition_pose_constraint_factor"] > 0.0 and self.status["prev_frames"] is not None:
            pose_constraint_desc = self.create_pose_constraint(self.status["prev_frames"], self.pose_constraint_node_names)
            pose_constraint_desc = self._map_label_to_canonical_keyframe(pose_constraint_desc)
            pose_constraint = PoseConstraint(self.skeleton, pose_constraint_desc, self.precision["smooth"],
                                              mp_constraints.settings["transition_pose_constraint_factor"])
            mp_constraints.constraints.append(pose_constraint)
            mp_constraints.pose_constraint_set = True

    def _add_pose_constraint_quat_frame(self, mp_constraints):
        pose_constraint_desc = self._create_pose_constraint_angular_from_preceding_motion()
        pose_constraint_quat_frame = PoseConstraintQuatFrame(self.skeleton, pose_constraint_desc,
                                                             self.precision["smooth"],
                                                             mp_constraints.settings["transition_pose_constraint_factor"])
        mp_constraints.constraints.append(pose_constraint_quat_frame)
        mp_constraints.pose_constraint_set = True

    def _add_keyframe_constraints(self, mp_constraints):
        """ Extract keyframe constraints of the motion primitive name.
        """

        if self.status["motion_primitive_name"] in self.action_constraints.keyframe_constraints.keys():
            #print("create constraints")
            #print(len(self.action_constraints.keyframe_constraints[self.status["motion_primitive_name"]]))
            for c_desc in self.action_constraints.keyframe_constraints[self.status["motion_primitive_name"]]:
                keyframe_constraint = self.create_keyframe_constraint(c_desc)
                if keyframe_constraint is not None:
                    mp_constraints.constraints.append(keyframe_constraint)
            #print("added", len(mp_constraints.constraints), "constraints")

    def create_keyframe_constraint(self, c_desc):
        c = None
        if "keyframeLabel" in c_desc["semanticAnnotation"].keys():
            c_desc = self._map_label_to_canonical_keyframe(c_desc)
            constraint_factor = self.trajectory_following_settings["position_constraint_factor"]

            if "merged" in c_desc.keys():
                c = TwoHandConstraintSet(self.skeleton, c_desc, self.precision["pos"], constraint_factor)
            elif "look_at" in c_desc.keys():
                c = LookAtConstraint(self.skeleton, c_desc, self.precision["pos"], constraint_factor)
            else:
                c = GlobalTransformConstraint(self.skeleton, c_desc, self.precision["pos"], constraint_factor)
        return c

    def _decide_on_optimization(self, mp_constraints):
        if self.local_optimization_mode == OPTIMIZATION_MODE_ALL:
            mp_constraints.use_local_optimization = True
        elif self.local_optimization_mode == OPTIMIZATION_MODE_KEYFRAMES:
            mp_constraints.use_local_optimization = len(self.action_constraints.keyframe_constraints.keys()) > 0 \
                                                    or self.status["is_last_step"]
        elif self.local_optimization_mode == OPTIMIZATION_MODE_TWO_HANDS:
            mp_constraints.use_local_optimization = self.action_constraints.contains_two_hands_constraints and not self.status["is_last_step"]
        else:
            mp_constraints.use_local_optimization = False

    def _add_events_to_event_list(self, mp_constraints):
        labeled_frames = self.motion_state_graph.node_groups[self.action_constraints.action_name].labeled_frames
        for label in self.action_constraints.keyframe_annotations.keys():
            print("try to set annotations for label ", label)
            if mp_constraints.motion_primitive_name in labeled_frames.keys():
                mp_name = mp_constraints.motion_primitive_name
                if label in labeled_frames[mp_name]:
                    event_list = self.action_constraints.keyframe_annotations[label]["annotations"]

                    # add keyframe constraint based on joint and label
                    constraint = None
                    if len(event_list) == 1:
                        #only if there is only one constraint on one joint otherwise correspondence is not clear
                        joint_name = event_list[0]["parameters"]["joint"]
                        for c in mp_constraints.constraints:
                            if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION and c.joint_name == joint_name and c.keyframe_label == label :
                                constraint = c
                                break

                    mp_constraints.keyframe_event_list[label] = KeyframeEvent(label,
                                                                              self._get_keyframe_from_label(label),
                                                                              event_list, constraint)

    def _map_label_to_canonical_keyframe(self, keyframe_constraint_desc):
        """ Enhances the keyframe constraint definition with a canonical keyframe set based on label
            for a keyframe on the canonical timeline
        :param keyframe_constraint:
        :return: Enhanced keyframe description or None if label was not found
        """
        #assert "keyframeLabel" in keyframe_constraint_desc["semanticAnnotation"].keys()
        keyframe_constraint_desc = copy(keyframe_constraint_desc)
        keyframe_constraint_desc["n_canonical_frames"] = self.status["n_canonical_frames"]
        keyframe_label = keyframe_constraint_desc["semanticAnnotation"]["keyframeLabel"]
        keyframe = self._get_keyframe_from_label(keyframe_label)
        if keyframe is not None:
            keyframe_constraint_desc["canonical_keyframe"] = keyframe
        else:
            return None
        return keyframe_constraint_desc

    def _get_keyframe_from_label(self, keyframe_label):
        return self.motion_state_graph.node_groups[self.action_constraints.action_name]. \
            get_keyframe_from_label(self.status["motion_primitive_name"], keyframe_label,
                                    self.status["n_canonical_frames"])

    def _create_pose_constraint_angular_from_preceding_motion(self):
        return MotionPrimitiveConstraintsBuilder.create_pose_constraint_angular(self.status["prev_frames"][-1])

    def create_pose_constraint(self, frames, node_names=None):
        if node_names is not None:
            weights_map = collections.OrderedDict()
            for node_name in node_names:
                if node_name in self.skeleton.joint_weight_map:
                    weights_map[node_name] = self.skeleton.joint_weight_map[node_name]
            weights = list(weights_map.values())
        else:
            node_names = list(self.skeleton.joint_weight_map.keys())
            weights = list(self.skeleton.joint_weight_map.values())

        last_pose = np.array(self.skeleton.convert_quaternion_frame_to_cartesian_frame(frames[-1], node_names))
        pre_root_pos = self.skeleton.nodes[node_names[0]].get_global_position(frames[-2])
        v = last_pose[0]-pre_root_pos  # measure only the velocity of the root
        frame_constraint = {"keyframeLabel": "start",
                            "frame_constraint": last_pose,
                            "velocity_constraint": v,
                            "semanticAnnotation": {"keyframeLabel": "start"},
                            "node_names": node_names,
                            "weights": weights}
        return frame_constraint

    @classmethod
    def create_pose_constraint_angular(cls, frame):
        frame_constraint = {"frame_constraint": frame, "keyframeLabel": "start", "semanticAnnotation": {"keyframeLabel": "start"}}
        return frame_constraint
