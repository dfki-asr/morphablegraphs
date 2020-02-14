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
SPATIAL_CONSTRAINT_TYPE_TRAJECTORY = "trajectory"
SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION = "keyframe_position"
SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D = "keyframe_2d_direction"
SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE = "keyframe_pose"
SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION = "keyframe_two_hands"
SPATIAL_CONSTRAINT_TYPE_TRAJECTORY_SET = "trajectory_set"
SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT = "keyframe_look_at"
SPATIAL_CONSTRAINT_TYPE_KEYFRAME_FEET = "keyframe_feet"
SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT = "ca_constraint"
SPATIAL_CONSTRAINT_TYPE_KEYFRAME_RELATIVE_POSITION = "keyframe_relative_position"
from .mgrd_constraint import MGRDKeyframeConstraint
from .trajectory_constraint import TrajectoryConstraint
from .trajectory_set_constraint import TrajectorySetConstraint
from .discrete_trajectory_constraint import DiscreteTrajectoryConstraint
from .keyframe_constraints.pose_constraint import PoseConstraint
from .keyframe_constraints.direction_2d_constraint import Direction2DConstraint
from .keyframe_constraints.global_transform_constraint import GlobalTransformConstraint
from .keyframe_constraints.relative_transform_constraint import RelativeTransformConstraint
from .keyframe_constraints.pose_constraint_quat_frame import PoseConstraintQuatFrame
from .keyframe_constraints.two_hand_constraint import TwoHandConstraintSet
from .keyframe_constraints.look_at_constraint import LookAtConstraint
from .keyframe_constraints.feet_constraint import FeetConstraint
from .keyframe_constraints.local_trajectory_constraint import LocalTrajectoryConstraint, SPATIAL_CONSTRAINT_TYPE_LOCAL_TRAJECTORY
