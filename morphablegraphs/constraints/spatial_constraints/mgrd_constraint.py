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

class MGRDKeyframeConstraint(object):
    """  A combination of a PoseConstraint and a SemanticConstraint for the integration with interact.

    Attributes:
        point (Vec3F): point in global Cartesian space..
        orientation (Vec4F): global orientation of the point as quaternion.
        joint_name (str): name of the constrained joint.
        weight (float): an weight for a linear combination of errors by the motion filter.
        annotations (List<string>): annotations that should be met
        time (float): the time in seconds on which the semantic annotation should be found. (None checks for occurrence)

    """
    def __init__(self, pose_constraint, semantic_constraint):
        self.joint_name = pose_constraint.joint_name
        self.weight = pose_constraint.weight
        self.point = pose_constraint.point
        self.orientation = pose_constraint.orientation
        self.annotations = semantic_constraint.annotations
        self.time = semantic_constraint.time
