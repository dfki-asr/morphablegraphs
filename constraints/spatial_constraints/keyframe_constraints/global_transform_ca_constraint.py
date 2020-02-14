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
from .global_transform_constraint import GlobalTransformConstraint
from .. import SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT


class GlobalTransformCAConstraint(GlobalTransformConstraint):
    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0, step_idx=-1):
        super(GlobalTransformCAConstraint, self).__init__(skeleton, constraint_desc, precision, weight_factor)
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT
        self.step_idx = step_idx

    def evaluate_motion_spline(self, aligned_spline):
        errors = np.zeros(self.n_canonical_frames)
        for i in range(self.n_canonical_frames):
            errors[i] = self._evaluate_joint_position(aligned_spline.evaluate(i))
        error = min(errors)
        print("ca constraint", error)
        return error#min(errors)

    def evaluate_motion_sample(self, aligned_quat_frames):
        errors = np.zeros(len(aligned_quat_frames))
        for i, frame in enumerate(aligned_quat_frames):
            errors[i] = self._evaluate_joint_position(frame)
        return min(errors)

    def get_residual_vector_spline(self, aligned_spline):
        return [self.evaluate_motion_spline(aligned_spline)]

    def get_residual_vector(self, aligned_frames):
        return [self.evaluate_motion_sample(aligned_frames)]

    def get_length_of_residual_vector(self):
        return 1
