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
import scipy.interpolate as si
try:
    from mgrd import TimeSpline as MGRDTimeSpline
    has_mgrd = True
except ImportError:
    has_mgrd = False
    pass

if has_mgrd:
    DEFAULT_FRAME_TIME = 0.013889
    SPLINE_DEGREE = 3
    N_TIME_DIM = 1

    class LegacyTemporalSplineModel(object):
        def __init__(self, data, frame_time=DEFAULT_FRAME_TIME):
            self.eigen_vectors = np.array(data['eigen_vectors_time'])
            self.mean_vector = np.array(data['mean_time_vector'])
            self.n_basis = int(data['n_basis_time'])
            self.n_dim = N_TIME_DIM
            self.n_components = len(self.eigen_vectors.T)
            self.knots = np.asarray(data['b_spline_knots_time'])
            self.eigen_coefs = list(zip(* self.eigen_vectors))
            self.n_canonical_frames = data['n_canonical_frames']
            self.canonical_time_range = np.arange(0, self.n_canonical_frames)

            self.frame_time = frame_time
            self.semantic_labels = []
            self.motion_primitive = None  # TODO improve: this has to be set currently from the outside because the motion primitive model is constructed after the spline model

        def get_n_components(self):
            return self.n_components

        def clone(self, labels=None):
            return self

        def create_spline(self, gamma, labels=None):
            sample_time_function = self._back_transform_gamma_to_canonical_time_function(gamma)
            knots, time_coeffs, degree = si.splrep(self.canonical_time_range, sample_time_function, w=None, k=SPLINE_DEGREE)
            time_coeffs = time_coeffs[:-4]
            time_coeffs = time_coeffs.reshape(len(time_coeffs), 1)
            #knots = knots.reshape(len(knots))
            return MGRDTimeSpline(time_coeffs, knots, N_TIME_DIM, degree, self)

        def back_project_time_function(self, gamma):
            """ Backtransform a lowdimensional vector gamma to the timewarping
            function t(t').

            Parameters
            ----------
            * gamma: numpy.ndarray
            \tThe lowdimensional vector

            Returns
            -------
            * time_function: numpy.ndarray
            \tThe indices of the timewarping function t(t')
            """
            return self._back_transform_gamma_to_canonical_time_function(gamma)
            #sample_time_function = self._invert_canonical_to_sample_time_function(canonical_time_function)
            #return canonical_time_function

        def _back_transform_gamma_to_canonical_time_function(self, gamma):
            """backtransform gamma to a discrete timefunction reconstruct t by evaluating the harmonics and the mean
            """
            mean_t = self._mean_temporal()
            n_latent_dim = len(self.eigen_coefs)
            t_eigen_spline = [(self.knots, self.eigen_coefs[i], 3) for i in range(n_latent_dim)]
            t_eigen_discrete = np.array([si.splev(self.canonical_time_range, spline_definition) for spline_definition in t_eigen_spline]).T
            canonical_time_function = [0]
            for i in range(self.n_canonical_frames):
                canonical_time_function.append(canonical_time_function[-1] + np.exp(mean_t[i] + np.dot(t_eigen_discrete[i], gamma)))
            # undo step from timeVarinaces.transform_timefunction during alignment
            canonical_time_function = np.array(canonical_time_function[1:])
            canonical_time_function -= 1.0
            return canonical_time_function

        def _mean_temporal(self):
            """Evaluates the mean time b-spline for the canonical time range.
            Returns
            -------
            * mean_t: np.ndarray
                Discretized mean time function.
            """
            mean_tck = (self.knots, self.mean_vector, SPLINE_DEGREE)
            return si.splev(self.canonical_time_range, mean_tck)

        # def _invert_canonical_to_sample_time_function(self, canonical_time_function):
        #     """ calculate inverse spline and then sample that inverse spline
        #         # i.e. calculate t'(t) from t(t')
        #     """
        #     # 1 get a valid inverse spline
        #     x_sample = np.arange(self.n_canonical_frames)
        #     sample_time_spline = si.splrep(canonical_time_function, x_sample, w=None, k=3)
        #     # 2 sample discrete data from inverse spline
        #     # canonical_time_function gets inverted to map from sample to canonical time
        #     frames = np.linspace(1, stop=canonical_time_function[-2], num=np.round(canonical_time_function[-2]))
        #     sample_time_function = si.splev(frames, sample_time_spline)
        #     sample_time_function = np.insert(sample_time_function, 0, 0)
        #     sample_time_function = np.insert(sample_time_function, len(sample_time_function), self.n_canonical_frames-1)
        #     return sample_time_function
