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

from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
from ..spatial_constraint_base import SpatialConstraintBase


class KeyframeConstraintBase(SpatialConstraintBase):

    def __init__(self, constraint_desc, precision, weight_factor=1.0):
        assert "canonical_keyframe" in list(constraint_desc.keys())
        super(KeyframeConstraintBase, self).__init__(precision, weight_factor)
        self.semantic_annotation = constraint_desc["semanticAnnotation"]
        self.keyframe_label = constraint_desc["semanticAnnotation"]["keyframeLabel"]
        self.canonical_keyframe = constraint_desc["canonical_keyframe"]
        if "time" in list(constraint_desc.keys()) and constraint_desc["time"] is not None:
            print(constraint_desc["time"])
            self.desired_time = float(constraint_desc["time"])
        else:
            self.desired_time = None
        self.event_name = None
        self.event_target = None
        if "canonical_end_keyframe" in constraint_desc:
            self.canonical_end_keyframe = constraint_desc["canonical_end_keyframe"]
        else:
            self.canonical_end_keyframe = None
        self.relative_joint_name = None
        if "relative_joint_name" in constraint_desc:
            self.relative_joint_name = constraint_desc["relative_joint_name"]
        self.mirror_joint_name = None
        if "mirror_joint_name" in constraint_desc:
            self.mirror_joint_name = constraint_desc["mirror_joint_name"]
        self.constrained_parent = None
        self.vector_to_parent = None
        if "constrained_parent" in constraint_desc and "vector_to_parent" in constraint_desc:
            self.constrained_parent = constraint_desc["constrained_parent"]
            self.vector_to_parent = constraint_desc["vector_to_parent"]
        self.src_tool_cos = None
        self.dest_tool_cos = None
        if "src_tool_cos" in constraint_desc and "dest_tool_cos" in constraint_desc:
            self.src_tool_cos = constraint_desc["src_tool_cos"]
            self.dest_tool_cos = constraint_desc["dest_tool_cos"]
        self.constrain_position_in_region = False
        if "constrain_position_in_region" in constraint_desc:
            self.constrain_position_in_region = constraint_desc["constrain_position_in_region"]
        self.constrain_orientation_in_region = False
        if "constrain_orientation_in_region" in constraint_desc:
            self.constrain_orientation_in_region = constraint_desc["constrain_orientation_in_region"]
        self.look_at = False
        if "look_at" in constraint_desc:
            self.look_at = constraint_desc["look_at"]
            

    def is_generated(self):
        return self.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION and "generated" in list(self.semantic_annotation.keys())

    def extract_keyframe_index(self, time_function, frame_offset):
        #TODO: test and verify for all cases
        if time_function is not None:
            return frame_offset + int(time_function[self.canonical_keyframe]) + 1  # add +1 to map the frame correctly
        else:
            return frame_offset + self.canonical_keyframe