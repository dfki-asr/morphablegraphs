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
from .utils import _transform_point_from_cad_to_opengl_cs, CONSTRAINT_TYPES


class KeyframeConstraintReader(object):
    def __init__(self, activate_coordinate_transform=True, scale_factor=1.0):
        self.activate_coordinate_transform = activate_coordinate_transform
        self.scale_factor = scale_factor

    def _extract_all_keyframe_constraints(self, constraint_list, node_group):
        """Orders the keyframe constraint for the labels found in the metainformation of
           the elementary actions based on labels as keys
        Returns
        -------
        * keyframe_constraints : dict of dict of lists
          Lists of constraints for each motion primitive in the subgraph.
          access as keyframe_constraints["label"]["joint"][index]
        """
        keyframe_constraints = dict()
        annotations = node_group.label_to_motion_primitive_map.keys()
        for label in annotations:
            constraints = self._extract_constraints_for_keyframe_label(constraint_list, label)
            keyframe_constraints[label] = constraints
        return keyframe_constraints

    def _extract_constraints_for_keyframe_label(self, input_constraint_list, label):
        """ Returns the constraints associated with the given label. Ordered
            based on joint names.
        Returns
        ------
        * key_constraints : dict of lists
        \t contains the list of the constrained joints
        """

        keyframe_constraints = dict()
        for joint_constraints in input_constraint_list:
            if "joint" in joint_constraints.keys():
                joint_name = joint_constraints["joint"]
                if joint_name not in keyframe_constraints:
                    keyframe_constraints[joint_name] = dict()
                for constraint_type in CONSTRAINT_TYPES:
                    if constraint_type not in keyframe_constraints[joint_name]:
                        keyframe_constraints[joint_name][constraint_type] = list()
                    if constraint_type in joint_constraints.keys():
                        filtered_list = self.filter_constraints_by_label(joint_constraints[constraint_type], label)
                        keyframe_constraints[joint_name][constraint_type] += filtered_list
        return keyframe_constraints

    def filter_constraints_by_label(self, constraints, label):
        keyframe_constraints = []
        for constraint in constraints:
            if self._constraint_definition_has_label(constraint, label):
                keyframe_constraints.append(constraint)
        return keyframe_constraints

    def _constraint_definition_has_label(self, constraint_definition, label):
        """ Checks if the label is in the semantic annotation dict of a constraint
        """
        if "semanticAnnotation" in constraint_definition.keys():
            if label in constraint_definition["semanticAnnotation"].keys():
                return True
        elif "keyframeLabel" in constraint_definition.keys() and constraint_definition["keyframeLabel"] == label:
            constraint_definition["semanticAnnotation"] = {label: True}
            return True
        return False

    def _reorder_keyframe_constraints_by_motion_primitive_name(self, node_group, keyframe_constraints):
        """ Order constraints extracted by _extract_all_keyframe_constraints for each state
        Returns
        -------
        reordered_constraints: dict of lists
        """
        reordered_constraints = dict()
        # iterate over keyframe labels
        for keyframe_label in keyframe_constraints.keys():
            mp_names = node_group.label_to_motion_primitive_map[keyframe_label]
            for mp_name in mp_names:
                time_info = node_group.labeled_frames[mp_name][keyframe_label]
                if mp_name not in reordered_constraints.keys():
                    reordered_constraints[mp_name] = list()
                # iterate over joints constrained at that keyframe
                for joint_name in keyframe_constraints[keyframe_label].keys():
                    # iterate over constraints for that joint
                    for c_type in CONSTRAINT_TYPES:
                        new_constraint = self._filter_by_constraint_type(keyframe_constraints, keyframe_label, joint_name, time_info, c_type)
                        reordered_constraints[mp_name] += new_constraint
        return reordered_constraints

    def get_ordered_keyframe_constraints(self, action_list, action_index, node_group):
        """
        Returns
        -------
            reordered_constraints: dict of lists
            dict of constraints lists applicable to a specific motion primitive of the node_group
        """
        keyframe_constraints = self._extract_all_keyframe_constraints(action_list[action_index]["constraints"], node_group)
        return self._reorder_keyframe_constraints_by_motion_primitive_name(node_group, keyframe_constraints)

    def _filter_by_constraint_type(self, constraints, keyframe_label, joint_name, time_info, c_type):
        filtered_constraints = list()
        if c_type in constraints[keyframe_label][joint_name].keys():
            for constraint in constraints[keyframe_label][joint_name][c_type]:
                # create constraint definition usable by the algorithm
                # and add it to the list of constraints for that state
                constraint_desc = self._extend_keyframe_constraint_definition(keyframe_label, joint_name, constraint,
                                                                              time_info, c_type)
                filtered_constraints.append(constraint_desc)
        return filtered_constraints

    def _extend_keyframe_constraint_definition(self, keyframe_label, joint_name, constraint, time_info, c_type):
        """ Creates a dict containing all properties stated explicitly or implicitly in the input constraint
        Parameters
        ----------
        * keyframe_label : string
          keyframe label
        * joint_name : string
          Name of the joint
        * constraint : dict
          Read from json input file
        * time_info : string
          Time information corresponding to an annotation read from morphable graph meta information

         Returns
         -------
         *constraint_desc : dict
          Contains the keys joint, position, orientation, time, semanticAnnotation
        """
        position = [None, None, None]
        orientation = [None, None, None]
        time = None
        event_name = None
        event_target = None

        if "position" in constraint.keys():
            position = constraint["position"]
        if "orientation" in constraint.keys():
            orientation = constraint["orientation"]
            if orientation != [None, None, None]:
                orientation = _transform_point_from_cad_to_opengl_cs(orientation, self.activate_coordinate_transform)
                print("orientation", orientation)
        if "time" in constraint.keys():
            time = constraint["time"]
        # check if last or fist frame from annotation
        position = _transform_point_from_cad_to_opengl_cs(position, self.activate_coordinate_transform)
        if "semanticAnnotation" in constraint.keys():
            semantic_annotation = constraint["semanticAnnotation"]
        else:
            semantic_annotation = dict()

        if "eventName" in constraint:
            event_name = constraint["eventName"]
            event_target = constraint["eventTarget"]
        # self._add_legacy_constrained_gmm_info(time_info, semantic_annotation)
        semantic_annotation["keyframeLabel"] = keyframe_label
        extended_constraint_desc = {"joint": joint_name,
                           "position": position,
                           "orientation": orientation,
                           "time": time,
                           "semanticAnnotation": semantic_annotation,
                           "eventName": event_name,
                           "eventTarget": event_target}

        if c_type == "directionConstraints":
            # position is interpreted as look at target
            # TODO improve integration of look at constraints
            extended_constraint_desc["look_at"] = True
        return extended_constraint_desc

    def _add_legacy_constrained_gmm_info(self, time_information, semantic_annotation):
        first_frame = None
        last_frame = None
        if time_information == "lastFrame":
            last_frame = True
        elif time_information == "firstFrame":
            first_frame = True
        semantic_annotation["firstFrame"] = first_frame
        semantic_annotation["lastFrame"] = last_frame
