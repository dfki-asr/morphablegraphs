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
from ..constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT
from ..constraints.keyframe_event import KeyframeEvent
from anim_utils.utilities.io_helper_functions import write_to_json_file

UNCONSTRAINED_EVENTS_TRANSFER_POINT = "transfer_point"


class KeyframeEventList(object):
    def __init__(self, create_ca_vis_data=False, add_rotate_events=False):
        self.frame_annotation = dict()
        self.frame_annotation['elementaryActionSequence'] = []
        self._keyframe_events_dict = dict()
        self.keyframe_events_dict = dict()
        self.ca_constraints = dict()
        self.create_ca_vis_data = create_ca_vis_data
        self.add_rotate_events = add_rotate_events

    def update_events(self, graph_walk, start_step):
        self._create_event_dict(graph_walk)
        self._create_frame_annotation(graph_walk, start_step)
        self._add_event_list_to_frame_annotation(graph_walk)
        self.keyframe_events_dict = {"events": self.get_keyframe_events_dict(),
                                     "elementaryActionSequence": self.frame_annotation["elementaryActionSequence"]}
        if self.create_ca_vis_data:
            self._create_collision_data_from_ca_constraints(graph_walk)
            self.keyframe_events_dict["collisionContent"] = self.ca_constraints

    def update_frame_annotation(self, action_name, start_frame, end_frame):
        """Adds a dictionary to self.frame_annotation marking start and end
            frame of an action.
        """
        action_frame_annotation = dict()
        action_frame_annotation["startFrame"] = start_frame
        action_frame_annotation["elementaryAction"] = action_name
        action_frame_annotation["endFrame"] = end_frame
        self.frame_annotation['elementaryActionSequence'].append(action_frame_annotation)

    def _create_event_dict(self, graph_walk):
        self._create_events_from_keyframe_constraints(graph_walk)
        self._add_unconstrained_events_from_annotation(graph_walk)
        # create rotation events to allow accurate rotation of the objects if the orientation is not constrained
        if not graph_walk.constrain_place_orientation:
            self._add_empty_rotate_events_for_detach(graph_walk)

    def _create_frame_annotation(self, graph_walk, start_step=0):
        self.frame_annotation['elementaryActionSequence'] = []
        for action in graph_walk.elementary_action_list:
            start_frame = graph_walk.steps[action.start_step].start_frame
            end_frame = graph_walk.steps[action.end_step].end_frame
            self.update_frame_annotation(action.action_name, start_frame, end_frame)

    def _create_events_from_keyframe_constraints(self, graph_walk):
        """ convert constraints of the motion primitive sequence into annotations
        """
        print("create event dict", len(graph_walk.steps))
        self._keyframe_events_dict = dict()
        frame_offset = 0
        for step_idx, step in enumerate(graph_walk.steps):
            time_function = None
            if graph_walk.use_time_parameters:
                time_function = graph_walk.motion_state_graph.nodes[step.node_key].back_project_time_function(step.parameters)

            if step.motion_primitive_constraints is not None:
                for keyframe_event in step.motion_primitive_constraints.keyframe_event_list.values():
                    event_keyframe_index = keyframe_event.extract_keyframe_index(time_function, frame_offset)
                    existing_events = None
                    if event_keyframe_index in self._keyframe_events_dict.keys():
                        existing_events = self._keyframe_events_dict[event_keyframe_index]
                    keyframe_event.merge_event_list(existing_events)
                    self._keyframe_events_dict[event_keyframe_index] = keyframe_event

            else:
                print("no constraints")
            frame_offset += step.end_frame - step.start_frame + 1

    def get_keyframe_events_dict(self):
        result = dict()
        for key in list(self._keyframe_events_dict.keys()):
            #print key, self._keyframe_events_dict[key]
            result[key] = self._keyframe_events_dict[key].event_list
        return result

    def export_to_file(self, prefix):
        write_to_json_file(prefix + "_annotations" + ".json", self.frame_annotation)
        write_to_json_file(prefix + "_actions" + ".json", self.keyframe_events_dict)

    def _add_empty_rotate_events_for_detach(self, graph_walk):
        """ create events with empty rotation that is later filled after IK"""
        #print "generate empty rotate events"
        for keyframe in self._keyframe_events_dict.keys():
            if self._keyframe_events_dict[keyframe].constraint is not None:
                orientation = self._keyframe_events_dict[keyframe].constraint.orientation
                if orientation is None or orientation == [None, None, None, None]:
                    continue
                for event in self._keyframe_events_dict[keyframe].event_list:
                    if event["event"] == "detach":
                        action_index = graph_walk.get_action_from_keyframe(keyframe)
                        if action_index < 0:
                            continue
                        if graph_walk.elementary_action_list[action_index].action_name in graph_walk.place_action_list:

                            rotate_event = dict()
                            rotate_event["event"] = "rotate"
                            rotate_event["parameters"] = dict()
                            rotate_event["parameters"]["target"] = event["parameters"]["target"]
                            rotate_event["parameters"]["joint"] = event["parameters"]["joint"]
                            rotate_event["parameters"]["globalOrientation"] = list(orientation)
                            rotate_event["parameters"]["relativeOrientation"] = [None, None, None]
                            rotate_event["parameters"]["referenceKeyframe"] = int(keyframe)
                            #rotate_event["parameters"]["pickKeyframe"] = int(keyframe)
                            if event["event"] == "attach":
                                rotate_keyframe = keyframe + 1
                            else:
                                rotate_keyframe = keyframe - 1
                            if rotate_keyframe >= 0:
                                if rotate_keyframe not in list(self._keyframe_events_dict.keys()):
                                    self._keyframe_events_dict[rotate_keyframe] = KeyframeEvent(None,-1,[])
                                self._keyframe_events_dict[rotate_keyframe].event_list.append(rotate_event)

    def _add_event_list_to_frame_annotation(self, graph_walk):
        """ Converts a list of events from the simulation event format to a format expected by CA
        :return:
        """
        keyframe_event_list = []
        for keyframe in list(self._keyframe_events_dict.keys()):
            for event_desc in self._keyframe_events_dict[keyframe].event_list:
                event = dict()
                if graph_walk.mg_input is not None and graph_walk.mg_input.activate_joint_mapping:
                    if isinstance(event_desc["parameters"]["joint"], str):
                        event["jointName"] = graph_walk.mg_input.inverse_map_joint(event_desc["parameters"]["joint"])
                    else:
                        event["jointName"] = list(map(graph_walk.mg_input.inverse_map_joint, event_desc["parameters"]["joint"]))
                else:
                    event["jointName"] = event_desc["parameters"]["joint"]
                event["jointName"] = self._map_both_hands_event(event, graph_walk.mg_input.activate_joint_mapping)
                event_type = event_desc["event"]
                target = event_desc["parameters"]["target"]
                event[event_type] = target
                event["frameNumber"] = int(keyframe)
                keyframe_event_list.append(event)
        self.frame_annotation["events"] = keyframe_event_list

    def _add_unconstrained_events_from_annotation(self, graph_walk):
        """The method assumes the start and end frames of each step were already warped by calling convert_to_motion
        """
        if graph_walk.mg_input is not None:
            for action_index, action_entry in enumerate(graph_walk.elementary_action_list):
                keyframe_annotations = graph_walk.mg_input.keyframe_annotations[action_index]
                for key in list(keyframe_annotations.keys()):
                    if key == UNCONSTRAINED_EVENTS_TRANSFER_POINT:
                        self._add_transition_event(graph_walk, keyframe_annotations, action_entry)

    def _add_transition_event(self, graph_walk, keyframe_annotations, action_entry):
        """ Look for the frame with the closest distance and add a transition event for it
        """
        if len(keyframe_annotations[UNCONSTRAINED_EVENTS_TRANSFER_POINT]["annotations"]) == 2:
            joint_name_a = keyframe_annotations[UNCONSTRAINED_EVENTS_TRANSFER_POINT]["annotations"][0]["parameters"]["joint"]
            joint_name_b = keyframe_annotations[UNCONSTRAINED_EVENTS_TRANSFER_POINT]["annotations"][1]["parameters"]["joint"]
            attach_joint = joint_name_a
            for event_parameters in keyframe_annotations[UNCONSTRAINED_EVENTS_TRANSFER_POINT]["annotations"]:
                if event_parameters["event"] == "attach":
                    attach_joint = event_parameters["parameters"]["joint"]

            if isinstance(joint_name_a, str):
                keyframe_range_start = graph_walk.steps[action_entry.start_step].start_frame
                keyframe_range_end = min(graph_walk.steps[action_entry.end_step].end_frame+1, graph_walk.motion_vector.n_frames)
                least_distance = np.inf
                closest_keyframe = graph_walk.steps[action_entry.start_step].start_frame
                for frame_index in range(keyframe_range_start, keyframe_range_end):
                    position_a = graph_walk.motion_state_graph.skeleton.nodes[joint_name_a].get_global_position(graph_walk.motion_vector.frames[frame_index])
                    position_b = graph_walk.motion_state_graph.skeleton.nodes[joint_name_b].get_global_position(graph_walk.motion_vector.frames[frame_index])
                    distance = np.linalg.norm(position_a - position_b)
                    if distance < least_distance:
                        least_distance = distance
                        closest_keyframe = frame_index
                target_object = keyframe_annotations[UNCONSTRAINED_EVENTS_TRANSFER_POINT]["annotations"][0]["parameters"]["target"]
                event_list = [{"event":"transfer", "parameters": {"joint" : attach_joint, "target": target_object}}]
                self._keyframe_events_dict[closest_keyframe] = KeyframeEvent(None,-1,event_list)
                print("added transfer event", closest_keyframe)

    def _map_both_hands_event(self, event, activate_joint_mapping=False):
        if isinstance(event["jointName"], list):
            if activate_joint_mapping:
                if "RightHand" in event["jointName"] and "LeftHand" in event["jointName"]:
                    return "BothHands"
                else:
                    return str(event["jointName"])
            else:
                if "RightToolEndSite" in event["jointName"] and "LeftToolEndSite" in event["jointName"]:
                    return "BothHands"
                else:
                    return str(event["jointName"])
        else:
            return str(event["jointName"])

    def _create_collision_data_from_ca_constraints(self, graph_walk):
        """ Convert CA constraints into an annotation dictionary used by the collision avoidance visualization.
        """
        self.ca_constraints = dict()
        for step in graph_walk.steps:
            for c in step.motion_primitive_constraints.constraints:
                if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT:
                    keyframe_range_start = step.start_frame
                    keyframe_range_end = min(step.end_frame+1, graph_walk.motion_vector.n_frames)
                    least_distance = np.inf
                    closest_keyframe = step.start_frame
                    for frame_index in range(keyframe_range_start, keyframe_range_end):
                        position = graph_walk.motion_state_graph.skeleton.nodes[c.joint_name].get_global_position(graph_walk.motion_vector.frames[frame_index])
                        d = position - c.position
                        d = np.dot(d,d)
                        if d < least_distance:
                            closest_keyframe = frame_index
                            least_distance = d
                    if closest_keyframe not in list(self.ca_constraints.keys()):
                        self.ca_constraints[closest_keyframe] = []
                    self.ca_constraints[closest_keyframe].append(c.joint_name)
