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
from . import LEN_CARTESIAN, LEN_QUAT
from sklearn.decomposition import PCA
from .functional_data import FunctionalData
from morphablegraphs.utilities.custom_math import cartesian_splines_distance
from morphablegraphs.motion_analysis.prepare_data import reshape_data_for_PCA, reshape_2D_data_to_motion_data, \
                                                         convert_quat_functional_data_to_cartesian_functional_data



def sfpca_objective_func(weights, data):
    """
    :param weights: parameters to be optimized
    :param data: parameters used for evaluating means squared loss
    :return: optimal weights

    STEPS:
    1. weight quaternion functional coefficients
    2. apply PCA on weighted quaternion functional coefficients, reconstruct quaternion functional coefficients using
       given number of principal components
    3. convert quaternion functional coefficients to Cartesian coefficients
    4. calculate mean squared error between reconstructed Cartesian coefficients and ground true
    """
    quat_coeffs, cartesian_coeffs, skeleton, npc, elementary_action, motion_primitive, data_repo, skeleton_json,\
    knots = data

    # initial weight vector
    dims = quat_coeffs.shape[2]
    n_joints = (dims - LEN_CARTESIAN) / LEN_QUAT
    extended_weights = np.zeros(dims)
    extended_weights[:LEN_CARTESIAN] = weights[:LEN_CARTESIAN]
    for i in range(n_joints):
        extended_weights[LEN_CARTESIAN + i*LEN_QUAT: LEN_CARTESIAN + (i+1)*LEN_QUAT] = weights[LEN_CARTESIAN + i]
    weight_mat = np.diag(extended_weights)

    # rescale quaternion coefficients
    weighted_quat_coeffs = np.dot(quat_coeffs, weight_mat)

    # apply PCA on rescaled coefficients
    pca = PCA(n_components=npc)
    projection = pca.fit_transform(reshape_data_for_PCA(weighted_quat_coeffs))

    # reconstruction
    backprojection = pca.inverse_transform(projection)
    inv_weight_mat = np.linalg.inv(weight_mat)
    backprojected_motion_data = reshape_2D_data_to_motion_data(backprojection, quat_coeffs.shape)
    unweighted_reconstruction = np.dot(backprojected_motion_data, inv_weight_mat)

    # convert quaternion functional coefficients to Cartesian coefficients
    cartesian_reconstruction = convert_quat_functional_data_to_cartesian_functional_data(elementary_action,
                                                                                         motion_primitive,
                                                                                         data_repo,
                                                                                         skeleton_json,
                                                                                         unweighted_reconstruction,
                                                                                         knots)

    err = cartesian_splines_distance(cartesian_reconstruction, cartesian_coeffs, skeleton)
    # print(err)
    return err