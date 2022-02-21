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
import os
import numpy as np
import collections
from anim_utils.animation_data import MotionVector, ROTATION_TYPE_QUATERNION, SkeletonBuilder, BVHReader, BVHWriter
from anim_utils.utilities.io_helper_functions import write_to_json_file
from ..constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION


class AnnotatedMotionVector(MotionVector):
    def __init__(self, skeleton=None, algorithm_config=None, rotation_type=ROTATION_TYPE_QUATERNION):
        super(AnnotatedMotionVector, self).__init__(skeleton, algorithm_config, rotation_type)
        self.keyframe_event_list = None
        self.mg_input = None
        self.graph_walk = None
        self.grounding_constraints = None
        self.ground_contacts = None
        self.ik_constraints = collections.OrderedDict()

    def export(self, output_filename, add_time_stamp=False, export_details=False):
        """ Saves the resulting animation frames, the annotation and actions to files.
        Also exports the input file again to the output directory, where it is
        used as input for the constraints visualization by the animation server.
        """

        MotionVector.export(self, self.skeleton, output_filename, add_time_stamp)
        self.export_annotation(output_filename)

    def export_annotation(self, output_filename):
        if self.mg_input is not None:
            write_to_json_file(output_filename + ".json", self.mg_input.mg_input_file)
        if self.keyframe_event_list is not None:
            self.keyframe_event_list.export_to_file(output_filename)

    def load_from_file(self, file_name):
        bvh = BVHReader(file_name)
        self.skeleton = SkeletonBuilder().load_from_bvh(bvh)

    def generate_bvh_string(self):
        quat_frames = np.array(self.frames)
        if len(quat_frames) > 0 and len(quat_frames[0]) < self.skeleton.reference_frame_length:
            quat_frames = self.skeleton.add_fixed_joint_parameters_to_motion(quat_frames)
        bvh_writer = BVHWriter(None, self.skeleton, quat_frames, self.skeleton.frame_time, True)
        return bvh_writer.generate_bvh_string()

    def to_unity_format(self, scale=1.0):
        """ Converts the frames into a custom json format for use in a Unity client"""
        animated_joints = [j for j, n in list(self.skeleton.nodes.items()) if
                           "EndSite" not in j and len(n.children) > 0]  # self.animated_joints
        unity_frames = []

        for node in list(self.skeleton.nodes.values()):
            node.quaternion_index = node.index

        for frame in self.frames:
            unity_frame = self._convert_frame_to_unity_format(frame, animated_joints, scale)
            unity_frames.append(unity_frame)

        result_object = dict()
        result_object["frames"] = unity_frames
        result_object["frameTime"] = self.frame_time
        result_object["jointSequence"] = animated_joints
        if self.graph_walk is not None:
            result_object["events"] = self._extract_event_list_from_keyframes()
        return result_object

    def _convert_frame_to_unity_format(self, frame, animated_joints, scale=1.0):
        """ Converts the frame into a custom json format and converts the transformations
            to the left-handed coordinate system of Unity.
            src: http://answers.unity3d.com/questions/503407/need-to-convert-to-right-handed-coordinates.html
        """
        unity_frame = {"rotations": [], "rootTranslation": None}
        for node_name in self.skeleton.nodes.keys():
            if node_name in animated_joints:
                node = self.skeleton.nodes[node_name]
                if node_name == self.skeleton.root:
                    t = frame[:3] * scale
                    unity_frame["rootTranslation"] = {"x": -t[0], "y": t[1], "z": t[2]}

                if node_name in self.skeleton.animated_joints:  # use rotation from frame
                    # TODO fix: the animated_joints is ordered differently than the nodes list for the latest model
                    index = self.skeleton.animated_joints.index(node_name)
                    offset = index * 4 + 3
                    r = frame[offset:offset + 4]
                    unity_frame["rotations"].append({"x": -r[1], "y": r[2], "z": r[3], "w": -r[0]})
                else:  # use fixed joint rotation
                    r = node.rotation
                    unity_frame["rotations"].append({"x": -float(r[1]), "y": float(r[2]), "z":float(r[3]), "w": -float(r[0])})
        return unity_frame

    def _extract_event_list_from_keyframes(self):
        frame_offset = 0
        event_list = list()
        for step in self.graph_walk.steps:
            time_function = None
            if self.graph_walk.use_time_parameters:
                time_function = self.graph_walk.motion_state_graph.nodes[step.node_key].back_project_time_function(
                    step.parameters)
            for c in step.motion_primitive_constraints.constraints:
                if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION and c.event_name is not None:
                    event_keyframe_index = c.extract_keyframe_index(time_function, frame_offset)
                    event_list.append({"eventName": c.event_name,
                                       "eventTarget": c.event_target,
                                       "keyframe": event_keyframe_index})
            frame_offset += step.end_frame - step.start_frame + 1
        print("extracted", event_list)
        return event_list
