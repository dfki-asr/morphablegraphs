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
from sklearn.mixture.gaussian_mixture import GaussianMixture
from .motion_spline import MotionSpline


class StaticMotionPrimitive(object):
    """ Implements the interface of a motion primitive but always returns the same motion example
    """
    def __init__(self):
        self.motion_spline = None
        self.name = ""
        self.has_time_parameters = False
        self.has_semantic_parameters = False
        self.n_canonical_frames = 0
        self.animated_joints = []

    def _initialize_from_json(self, data):
        self.name = data["name"]
        self.spatial_coefs = np.array(data["spatial_coeffs"])
        self.knots = np.array(data["knots"])
        self.n_canonical_frames = data["n_canonical_frames"]
        self.time_function = np.array(list(range(self.n_canonical_frames)))
        self.motion_spline = MotionSpline(self.spatial_coefs, self.time_function, self.knots, None)
        self.gmm = GaussianMixture(n_components=1, covariance_type='full')
        #self.gmm.fit([0])
        if "skeleton" in data:
            self.animated_joints = data["skeleton"]["animated_joints"]

    def sample_low_dimensional_vector(self, use_time_parameters=True):
        return [0]

    def sample(self, use_time_parameters=True):
        return self.motion_spline

    def back_project(self, s, use_time_parameters=True, speed=1.0):
        return self.motion_spline

    def back_project_time_function(self, gamma, speed=1.0):
        return self.time_function

    def get_n_spatial_components(self):
        return 1

    def get_n_time_components(self):
        return 0

    def get_gaussian_mixture_model(self):
        return self.gmm

    def get_n_canonical_frames(self):
        return self.n_canonical_frames

    def get_animated_joints(self):
        return self.animated_joints
