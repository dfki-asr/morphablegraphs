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
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:36:36 2015

@author: Erik Herrmann
"""

class SpatialConstraintBase(object):

    def __init__(self, precision, weight_factor=1.0):
        self.precision = precision
        self.weight_factor = weight_factor
        self.constraint_type = None

    def evaluate_motion_sample(self, aligned_quat_frames):
        pass

    def evaluate_motion_sample_with_precision(self, aligned_quat_frames):
        error = self.evaluate_motion_sample(aligned_quat_frames)
        if error < self.precision:
            success = True
        else:
            success = False
        return error, success

    def get_residual_vector(self, aligned_frames):
        pass

    def get_length_of_residual_vector(self):
        pass
