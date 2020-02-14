#!/usr/bin/env python
#

# Copyright 2019 DFKI GmbH, Daimler AG.
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
Created on Mon Jan 26 14:11:11 2015

@author: Han Du, Erik Herrmann, Markus Mauer
"""

import numpy as np
import json
from sklearn.mixture.gaussian_mixture import GaussianMixture, _compute_precision_cholesky
import scipy.interpolate as si
from . import B_SPLINE_DEGREE
from .motion_spline import MotionSpline
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import savgol_filter


class MotionPrimitive(object):
    """ Represent a motion primitive which can be sampled
    Parameters
    ----------
    * filename: string
    \tThe filename with the saved data in json format.
    Attributes
    ----------
    * s_pca: dictionary
    \tThe result of the spacial PCA. It is a dictionary having the
    (eigen_vectors, mean_vectors, n_basis,n_dim, maxima, n_components, knots) as values
    * t_pca: dictionary
    \tThe result of the temporal PCA. It is a dictionary having the
    (eigen_vectors, mean_vectors, n_basis,n_dim, knots) as values
    * gaussian_mixture_model: sklearn.mixture.GMM
    \tstatistical model on the low dimensional representation of motion samples
    *name: string
    \tIdentifier of the motion primitive
    *n_canonical_frames: int
    \tNumber of frames in the canonical timeline of the spatial data
    *translation_maxima: numpy.ndarray
    \tScaling factor to reconstruct the unnormalized translation parameters of a motion after inverse pca
    """
    def __init__(self, filename):
        self.filename = filename
        self.name = ""
        self.gaussian_mixture_model = None
        self.s_pca = dict()
        self.t_pca = dict()
        self.n_canonical_frames = 0
        self.translation_maxima = np.array([1.0, 1.0, 1.0])
        self.smooth_time_parameters = False
        self.has_time_parameters = True
        self.has_semantic_parameters = False
        self.animated_joints = []
        if self.filename is not None:
            self._load(self.filename)

    def _load(self, filename=None):
        """ Load a motion primitive from a file

        Parameters
        ----------
        * filename: string, optinal
        \tThe path to the saved json file. If None (default) the filename
        of the object will be taken.
        """
        with open(filename, 'rb') as infile:
            tmp = json.load(infile)
            infile.close()
            self._initialize_from_json(tmp)

    def _initialize_from_json(self, data):
        """ Load morphable model parameters from a dictionary and initialize
            the fda library and the Gaussian Mixture model.

        Parameters
        ----------
        * data: dictionary
        \tThe dictionary must contain all parameters for the motion primitive

        """
        #load name data and canonical frames of the motion
        if 'name' in list(data.keys()):
            self.name = data['name']
        if 'semantic_label' in list(data.keys()):
            print('low dimensional vector has semantic information!')
            self.has_semantic_parameters = True
            self.semantic_labels = data['semantic_label']
        self.n_canonical_frames = data['n_canonical_frames']
        self.canonical_time_range = np.arange(0, self.n_canonical_frames)
        # initialize parameters for the motion sampling and back projection
        self._init_gmm_from_json(data)
        self._init_spatial_parameters_from_json(data)
        if 'eigen_vectors_time' in list(data.keys()):
            self._init_time_parameters_from_json(data)
            self.has_time_parameters = True
        else:
            self.has_time_parameters = False
            self.t_pca = dict()
            self.t_pca["n_components"] = 0

        if "animated_joints" in data:
            self.animated_joints = data["animated_joints"]

    def _init_gmm_from_json(self, data):
        """ Initialize the Gaussian Mixture model.

        Parameters
        ----------
        * data: dictionary
        \tThe dictionary must contain all parameters for the Gaussian Mixture Model.

        """
        n_components = len(np.array(data['gmm_weights']))
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm.weights_ = np.array(data['gmm_weights'])
        gmm.means_ = np.array(data['gmm_means'])
        gmm.converged_ = True
        gmm.covariances_ = np.array(data['gmm_covars'])
        gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_,
                                                               covariance_type='full')
        gmm.n_dims = len(gmm.means_[0])
        self.gaussian_mixture_model = gmm


    def _init_spatial_parameters_from_json(self, data):
        """  Set the parameters for back_project_spatial_function.

        Parameters
        ----------
        * data: dictionary
        \tThe dictionary must contain all parameters for the spatial pca.

        """
        self.translation_maxima = np.array(data['translation_maxima'])
        self.s_pca = dict()
        self.s_pca["eigen_vectors"] = np.transpose(np.array(data['eigen_vectors_spatial']))
        self.s_pca["mean_vector"] = np.array(data['mean_spatial_vector'])
        self.s_pca["n_basis"] = int(data['n_basis_spatial'])
        self.s_pca["n_dim"] = int(data['n_dim_spatial'])
        self.s_pca["n_components"] = len(self.s_pca["eigen_vectors"].T)
        self.s_pca["knots"] = np.asarray(data['b_spline_knots_spatial'])

    def _init_time_parameters_from_json(self, data):
        """  Set the parameters for back_project_time_function.

        Parameters
        ----------
        * data: dictionary
        \tThe dictionary must contain all parameters for the spatial pca.
        """
        self.t_pca = dict()
        self.t_pca["eigen_vectors"] = np.array(data['eigen_vectors_time'])
        self.t_pca["mean_vector"] = np.array(data['mean_time_vector'])
        self.t_pca["n_basis"] = int(data['n_basis_time'])
        self.t_pca["n_dim"] = 1
        self.t_pca["n_components"] = len(self.t_pca["eigen_vectors"].T)
        self.t_pca["knots"] = np.asarray(data['b_spline_knots_time'])
        self.t_pca["eigen_coefs"] = list(zip(* self.t_pca["eigen_vectors"]))

    def sample_low_dimensional_vector(self, n_samples=1):
        """ Sample the motion primitive and return a low dimensional vector
        Returns
        -------
        * s:  numpy.ndarray
        """
        assert self.gaussian_mixture_model is not None, "Motion primitive not initialized."
        return self.gaussian_mixture_model.sample(n_samples)[0]

    def sample(self, use_time_parameters=True):
        """ Sample the motion primitive and return a motion spline

        Parameters
        ----------
        *use_time_parameters: boolean
        \tIf True use time function from _inverse_temporal_pca else canonical time line

        Returns
        -------
        * motion: MotionSample
        \tThe sampled motion as object of type MotionSpline
        """
        return self.back_project(np.ravel(self.sample_low_dimensional_vector()), use_time_parameters)

    def back_project(self, s, use_time_parameters=True, speed=1.0):
        """ Return a motion sample based on a low dimensional motion vector.

        Parameters
        ----------
        *s: numpy.ndarray
        \tThe low dimensional motion representation sampled from a GMM or GP
        *use_time_parameters: boolean
        \tIf True use time function from _inverse_temporal_pca else canonical time line
        Returns
        -------
        * motion: MotionSample
        \tThe sampled motion as object of type MotionSample
        """
        semantic_annotation = None
        if self.has_semantic_parameters:
            semantic_label = s[-1]
            for key, value in self.semantic_labels.items():
                if int(np.round(semantic_label)) == value:
                    semantic_annotation = key
            if semantic_annotation is None:
                raise ValueError('Unknown semantic label!')
            s = np.delete(s, -1)
        spatial_coeffs = self.back_project_spatial_coeffs(s[:self.s_pca["n_components"]])
        if self.has_time_parameters and use_time_parameters:
            time_function = self.back_project_time_function(s[self.s_pca["n_components"]:], speed)
        else:
            time_function = np.linspace(0, self.n_canonical_frames, int(self.n_canonical_frames * (1.0 / speed)))
        return MotionSpline(spatial_coeffs, time_function, self.s_pca["knots"], semantic_annotation, low_dimensional_parameters=s)

    def back_project_spatial_coeffs(self, alpha):
        """ Backtransform a lowdimensional vector alpha to a coefficients of
        a functional motion representation.

        Parameters
        ----------
        * alpha: numpy.ndarray
        \tThe lowdimensional vector

        Returns
        -------
        * motion: numpy.ndarray
        \t Reconstructed coefficients of the functional motion representation.
        """
        #reconstruct coefs of the functionial representation
        coefs = np.dot(self.s_pca["eigen_vectors"], alpha)
        coefs += self.s_pca["mean_vector"]
        coefs = coefs.reshape((self.s_pca["n_basis"], self.s_pca["n_dim"]))
        #undo the scaling on the translation
        coefs[:, :3] *= self.translation_maxima
        return coefs

    def _mean_temporal(self):
        """Evaluates the mean time b-spline for the canonical time range.
        Returns
        -------
        * mean_t: np.ndarray
            Discretized mean time function.
        """
        mean_tck = (self.t_pca["knots"], self.t_pca["mean_vector"], 3)
        return si.splev(self.canonical_time_range, mean_tck)

    def back_project_time_function(self, gamma, speed=1.0):
        """ Backtransform a lowdimensional vector gamma to the timewarping
        function t(t') and inverse it to t'(t).

        Parameters
        ----------
        * gamma: numpy.ndarray
        \tThe lowdimensional vector

        Returns
        -------
        * time_function: numpy.ndarray
        \tThe indices of the timewarping function t'(t)
        """
        canonical_time_function = self._back_transform_gamma_to_canonical_time_function(gamma)
        sample_time_function = self._invert_canonical_to_sample_time_function(canonical_time_function, speed)
        if self.smooth_time_parameters:
            return self._smooth_time_function(sample_time_function)
        else:
            return sample_time_function

    def _back_transform_gamma_to_canonical_time_function(self, gamma):
        """backtransform gamma to a discrete timefunction reconstruct t by evaluating the harmonics and the mean
        """
        mean_t = self._mean_temporal()
        n_latent_dim = len(self.t_pca["eigen_coefs"])
        t_eigen_spline = [(self.t_pca["knots"], self.t_pca["eigen_coefs"][i], B_SPLINE_DEGREE) for i in range(n_latent_dim)]
        t_eigen_discrete = np.array([si.splev(self.canonical_time_range, spline_definition) for spline_definition in t_eigen_spline]).T
        canonical_time_function = [0]
        for i in range(self.n_canonical_frames):
            canonical_time_function.append(canonical_time_function[-1] + np.exp(mean_t[i] + np.dot(t_eigen_discrete[i], gamma)))
        # undo step from timeVarinaces.transform_timefunction during alignment
        canonical_time_function = np.array(canonical_time_function[1:])
        canonical_time_function -= 1.0
        return canonical_time_function

    def _invert_canonical_to_sample_time_function(self, canonical_time_function, speed=1.0):
        """ calculate inverse spline and then sample that inverse spline
            # i.e. calculate t'(t) from t(t')
        """
        # 1 get a valid inverse spline
        x_sample = np.arange(self.n_canonical_frames)
        sample_time_spline = si.splrep(canonical_time_function, x_sample, w=None, k=B_SPLINE_DEGREE)
        # 2 sample discrete data from inverse spline
        # canonical_time_function gets inverted to map from sample to canonical time
        num = np.round(canonical_time_function[-2]) * (1.0 / speed)
        frames = np.linspace(1, stop=canonical_time_function[-2], num=num)
        sample_time_function = si.splev(frames, sample_time_spline)
        sample_time_function = np.insert(sample_time_function, 0, 0)
        sample_time_function = np.insert(sample_time_function, len(sample_time_function), self.n_canonical_frames-1)
        return sample_time_function

    def _smooth_time_function(self, time_function):
        """temporary solution: smooth temporal parameters, then linearly extend
        """
        t_max = max(time_function)
        # it to (0, t_max)
        sigma = 5
        #t = gaussian_filter1d(time_function, sigma)
        t = savgol_filter(time_function, 15, 3)
        #a = min(t)
        #b = max(t)
        #t = [(i-a)/(b-a) * t_max for i in t]
        return np.array(t)

    def get_n_canonical_frames(self):
        return self.n_canonical_frames

    def get_n_spatial_components(self):
        return self.s_pca["n_components"]

    def get_n_time_components(self):
        if "n_components" in self.t_pca.keys():
            return self.t_pca["n_components"]
        else:
            return 0

    def path_following_obj(self, target, alpha):
        coeffs = self.back_project_spatial_coeffs(alpha)
        end_pos = coeffs[-1,:3]
        error = np.linalg.norm(target-end_pos)
        return error

    def get_spatial_jacobian(self):
        #alpha = 5
        #n_dims = 100
        # 100x5 * 5x1 = 100x1
        # 5x100
        return self.s_pca["eigen_vectors"].T

    def path_following_jac(self, target, alpha, coff_idx):
        """
        you want to know how the error changes depending on the parameter change
        so the gradient is a vector of length alpha
        Error(a) = y(a)*2
        Y(a) = M*a

        d_error/d_a = d_error/d_y* d_y/d_a
        = 2y(a) * M*a^0 = 2y(a)*M

        """

        coeffs = self.back_project_spatial_coeffs(alpha)
        end_pos = coeffs[-1, :3]
        d_error = target - end_pos #dim = 3
        d_m = self.s_pca["n_components"] # dim = 100 x alpha
        # coefs = coefs.reshape((self.s_pca["n_basis"], self.s_pca["n_dim"]))
        row_idx = self.s_pca["n_dim"]*coff_idx
        d_pos = d_m[row_idx:row_idx+3, :]
        # 1x3 * 3 x n_alpha
        return np.dot(d_error, d_pos)

    def get_animated_joints(self):
        return self.animated_joints