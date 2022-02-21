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
Created on Mon Aug 03 19:02:55 2015

@author: Erik Herrmann
"""


import numpy as np
from .global_transform_constraint import GlobalTransformConstraint
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_RELATIVE_POSITION


class RelativeTransformConstraint(GlobalTransformConstraint):
    """
    * constraint_desc: dict
        Contains joint, position, orientation and semantic Annotation
    """

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(RelativeTransformConstraint, self).__init__(skeleton, constraint_desc, precision, weight_factor)
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_RELATIVE_POSITION
        self.offset = constraint_desc["offset"]

    def _evaluate_joint_position(self, frame):
        global_m = self.skeleton.nodes[self.joint_name].get_global_matrix(frame)
        pos = np.dot(global_m, self.offset)[:3]
        d = np.linalg.norm(self.position-pos)
        return d

