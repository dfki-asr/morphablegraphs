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
Created on Fri Jul 31 13:21:08 2015

@author: Han Du, Erik Herrmann
"""

import numpy as np
from scipy.optimize.optimize import approx_fprime
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
from anim_utils.animation_data.motion_concatenation import transform_quaternion_frames, get_transform_from_start_pose, align_quaternion_frames_automatically
from ...constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_TRAJECTORY_SET

def align_quaternion_frames(skeleton, node_name, new_frames, prev_frames=None, start_pose=None):
    if prev_frames is not None:
        return align_quaternion_frames_automatically(skeleton, node_name, new_frames,  prev_frames)
    elif start_pose is not None:
        m = get_transform_from_start_pose(start_pose)
        first_frame_pos = new_frames[0][:3].tolist() + [1]
        t_pos = np.dot(m, first_frame_pos)[:3]
        delta = start_pose["position"]
        delta[0] -= t_pos[0]
        delta[2] -= t_pos[2]
        m[:3, 3] = delta
        transformed_frames = transform_quaternion_frames(new_frames, m)
        return transformed_frames
    else:
        return new_frames

def obj_frame_error(s, data):
    motion_primitive, target, joint_name, frame_idx = data
    sample = motion_primitive.back_project(s, use_time_parameters=False)
    frame = sample.get_motion_vector()[frame_idx]
    p = motion_primitive.skeleton.nodes[joint_name].get_global_position(frame)
    return np.linalg.norm(target-p)


def step_goal_error(s, data):
    motion_primitive, mp_constraints, _ = data[:3]
    target_pos = mp_constraints.constraints[0].position
    spline_model = motion_primitive.motion_primitive.spatial
    sfpca = spline_model.fpca
    n_spatial = sfpca.n_components
    coeffs_flat = sfpca.project(s[:n_spatial])
    idx = (spline_model.n_coeffs-1)* spline_model.n_dims
    pos = coeffs_flat[idx:idx+3]
    pos[1] = 0
    delta = target_pos - pos
    error = np.dot(delta,delta)
    return error

def step_goal_jac(s, data):
    motion_primitive, mp_constraints, prev_frames = data[:3]

    target_pos = mp_constraints.constraints[0].position

    spline_model = motion_primitive.motion_primitive.spatial
    sfpca = spline_model.fpca
    n_spatial = sfpca.n_components
    coeffs_flat = sfpca.project(s[:n_spatial])
    idx = (spline_model.n_coeffs-1) * spline_model.n_dims
    pos = coeffs_flat[idx:idx + 3]
    pos[1] = 0
    vector_derivative = sfpca.eigen[:, idx:idx + 3]
    gradient = -(target_pos - pos)
    jac = np.zeros(len(s))
    jac[:n_spatial] = np.dot(vector_derivative, gradient)
    return 2 * jac



def step_goal_and_naturalness(s, data):
    motion_primitive, mp_constraints, _ = data[:3]
    target_pos = mp_constraints.constraints[0].position
    spline_model = motion_primitive.motion_primitive.spatial
    sfpca = spline_model.fpca
    n_spatial = sfpca.n_components
    coeffs_flat = sfpca.project(s[:n_spatial])
    idx = (spline_model.n_coeffs-1)* spline_model.n_dims
    pos = coeffs_flat[idx:idx+3]
    pos[1] = 0
    delta = target_pos - pos
    error = np.dot(delta,delta)
    n_log_likelihood = -data[0].get_gaussian_mixture_model().score(s)
    error += n_log_likelihood
    return error

def log_likelihood_jac(s, gmm):
    logLikelihoods = _estimate_log_gaussian_prob(s, gmm.means_, gmm.precisions_cholesky_, 'full')
    logLikelihoods = np.ravel(logLikelihoods)
    numerator = 0
    n_models = len(gmm.weights)
    for i in range(n_models):
        numerator += np.exp(logLikelihoods[i]) * gmm.weights[i] * np.dot(np.linalg.inv(gmm.covars[i]), (s - gmm.means[i]))
    denominator = np.exp(gmm.score(s))
    if denominator != 0:
        return numerator / denominator
    else:
        return np.ones(s.shape)

def step_goal_and_naturalness_jac(s, data):
    motion_primitive, mp_constraints, _ = data[:3]

    target_pos = mp_constraints.constraints[0].position

    spline_model = motion_primitive.motion_primitive.spatial
    sfpca = spline_model.fpca
    n_spatial = sfpca.n_components
    coeffs_flat = sfpca.project(s[:n_spatial])
    idx = (spline_model.n_coeffs-1) * spline_model.n_dims
    pos = coeffs_flat[idx:idx + 3]
    pos[1] = 0
    vector_derivative = sfpca.eigen[:, idx:idx + 3]
    gradient = -(target_pos - pos)
    jac = np.zeros(len(s))
    jac[:n_spatial] = np.dot(vector_derivative, gradient)
    return 2 * jac - log_likelihood_jac(s, motion_primitive.get_gaussian_mixture_model())



def obj_spatial_error_sum(s, data):
    """ Calculates the error of a low dimensional motion vector s 
    given a list of constraints.
    Note: Time parameters and time constraints will be ignored. 

    Parameters
    ---------
    * s : np.ndarray
        low dimensional motion representation
    * data : tuple
        Contains  motion_primitive, motion_primitive_constraints, prev_frames
        
    Returns
    -------
    * error: float
    """
    motion_primitive, mp_constraints, prev_frames = data
    mp_constraints.min_error = mp_constraints.evaluate(motion_primitive, s, prev_frames, use_time_parameters=False)
    return mp_constraints.min_error


def obj_spatial_error_sum_and_naturalness(s, data):
    """ Calculates the error of a low dimensional motion vector s given
        constraints and the prior knowledge from the statistical model

    Parameters
    ---------
    * s : np.ndarray
        low dimensional motion representation
    * data : tuple
        Contains motion_primitive, motion_primitive_constraints, prev_frames, error_scale_factor, quality_scale_factor

    Returns
    -------
    * error: float
    """

    spatial_error = obj_spatial_error_sum(s, data[:-3])# ignore the kinematic factor and quality factor
    error_scale = data[-3]
    quality_scale = data[-2]
    init_error_sum = data[-1]
    n_log_likelihood = -data[0].get_gaussian_mixture_model().score(s)
    error = error_scale * spatial_error
    error = error_scale * spatial_error + n_log_likelihood * quality_scale


def obj_spatial_error_sum_and_naturalness_jac(s, data):
    """ jacobian of error function. It is a combination of analytic solution 
        for motion primitive model and numerical solution for kinematic error
    """
    #  Extract relevant parameters from data tuple. 
    #  Note other parameters are used for calling obj_error_sum
    gmm = data[0].get_gaussian_mixture_model()
    error_scale = data[-1]
    quality_scale = data[-2]
    
    logLikelihoods = _estimate_log_gaussian_prob(s, gmm.means_, gmm.precisions_cholesky_, 'full')
    logLikelihoods = np.ravel(logLikelihoods)
    numerator = 0

    n_models = len(gmm.weights_)
    for i in range(n_models):
        numerator += np.exp(logLikelihoods[i]) * gmm.weights_[i] * np.dot(np.linalg.inv(gmm.covars_[i]), (s - gmm.means_[i]))
    denominator = np.exp(gmm.score([s])[0])
    logLikelihood_jac = numerator / denominator
    kinematic_jac = approx_fprime(s, obj_spatial_error_sum, 1e-7, data[-2:])# ignore the kinematic factor and quality factor
    jac = logLikelihood_jac * quality_scale + kinematic_jac * error_scale
    return jac


def obj_spatial_error_residual_vector(s, data):
    """ Calculates the error of a low dimensional motion vector s
    given a list of constraints and stores the error of each constraint in a list.
    Note: Time parameters and time constraints will be ignored.

    Parameters
    ---------
    * s : np.ndarray
        low dimensional motion representation
    * data : tuple
        Contains  motion_primitive, motion_primitive_constraints, prev_frames

    Returns
    -------
    * residual_vector: list
    """
    motion_primitive, motion_primitive_constraints, prev_frames, error_scale, quality_scale, init_error_sum = data
    residual_vector = motion_primitive_constraints.get_residual_vector(motion_primitive, s, prev_frames, use_time_parameters=False)
    motion_primitive_constraints.min_error = np.sum(residual_vector)
    n_variables = s.shape[0]
    n_error_values = len(residual_vector)
    while n_error_values < n_variables:
        residual_vector.append(0)
        n_error_values += 1
    return np.array(residual_vector)/init_error_sum


def obj_spatial_error_residual_vector_and_naturalness(s, data):
    """ Calculates the error of a low dimensional motion vector s
    given a list of constraints and stores the error of each constraint in a list.
    Note: Time parameters and time constraints will be ignored.

    Parameters
    ---------
    * s : np.ndarray
        low dimensional motion representation
    * data : tuple
        Contains  motion_primitive, motion_primitive_constraints, prev_frames

    Returns
    -------
    * residual_vector: list
    """
    mp, mp_constraints, prev_frames, error_scale, quality_scale, init_error_sum = data
    negative_log_likelihood = float(-data[0].get_gaussian_mixture_model().score(s.reshape(1, -1)) * quality_scale)
    residual_vector = mp_constraints.get_residual_vector(mp, s, prev_frames, use_time_parameters=False)
    mp_constraints.min_error = np.sum(residual_vector)
    n_error_values = len(residual_vector)
    for i in range(n_error_values):
        residual_vector[i] *= error_scale
        residual_vector[i] += negative_log_likelihood
    n_variables = s.shape[0]
    while n_error_values < n_variables:
        residual_vector.append(0)
        n_error_values += 1
    return np.array(residual_vector) / init_error_sum


def obj_time_error_sum(s, data):
    """ Calculates the error for time constraints for certain keyframes
    Parameters
    ---------
    * s : np.ndarray
        concatenation of low dimensional motion representations
    * data : tuple
        Contains motion_primitive_graph, graph_walk, time_constraints, motion, start_step

    Returns
    -------
    * error: float
    """
    motion_primitive_graph, graph_walk, time_constraints, error_scale, quality_scale = data
    time_error = time_constraints.evaluate_graph_walk(s, motion_primitive_graph, graph_walk)
    n_log_likelihood = -time_constraints.get_average_loglikelihood(s, motion_primitive_graph, graph_walk)
    error = error_scale * time_error + n_log_likelihood * quality_scale
    return error


def obj_global_error_sum(s, data):
    """ Calculates the error for spatial constraints for certain keyframes
    Parameters
    ---------
    * s : np.ndarray
        concatenation of low dimensional motion representations
    * data : tuple
        Contains morhable_graph, time_constraints, motion, start_step

    Returns
    -------
    * error: float
    """
    offset = 0
    error = 0
    motion_primitive_graph, graph_walk_steps, error_scale, quality_scale, prev_frames = data
    skeleton = motion_primitive_graph.skeleton
    node_name = skeleton.aligning_root_node
    for step in graph_walk_steps:
        alpha = s[offset:offset+step.n_spatial_components]
        sample_frames = motion_primitive_graph.nodes[step.node_key].back_project(alpha, use_time_parameters=False).get_motion_vector()
        step_data = motion_primitive_graph.nodes[step.node_key], step.motion_primitive_constraints, \
                       prev_frames
        prev_frames = align_quaternion_frames(skeleton, node_name, sample_frames, prev_frames,
                                              step.motion_primitive_constraints.start_pose)
        error += obj_spatial_error_sum(alpha, step_data)#_and_naturalness
        offset += step.n_spatial_components
    print("global error", error)
    return error


def obj_global_residual_vector(s, data):
    """ Calculates the error for spatial constraints for certain keyframes
    Parameters
    ---------
    * s : np.ndarray
        concatenation of low dimensional motion representations
    * data : tuple
        Contains morhable_graph, time_constraints, motion, start_step

    Returns
    -------
    * residual_vector: list
    """
    offset = 0
    residual_vector = []
    motion_primitive_graph, graph_walk_steps, error_scale, quality_scale, prev_frames, init_error_sum = data
    skeleton = motion_primitive_graph.skeleton
    node_name = skeleton.aligning_root_node
    for step in graph_walk_steps:
        alpha = s[offset:offset+step.n_spatial_components]
        sample_frames = motion_primitive_graph.nodes[step.node_key].back_project(alpha, use_time_parameters=False).get_motion_vector()
        step_data = motion_primitive_graph.nodes[step.node_key], step.motion_primitive_constraints, \
                       prev_frames, error_scale, quality_scale
        prev_frames = align_quaternion_frames(skeleton, node_name, sample_frames, prev_frames,
                                              step.motion_primitive_constraints.start_pose)
        residual_vector += obj_spatial_error_residual_vector(alpha, step_data)
        offset += step.n_spatial_components
    return np.array(residual_vector)/init_error_sum



def obj_global_residual_vector_and_naturalness(s, data):
    """ Calculates the error for spatial constraints for certain keyframes and
        adds negative log likelihood from the statistical model
    Parameters
    ---------
    * s : np.ndarray
        concatenation of low dimensional motion representations
    * data : tuple
        Contains morhable_graph, time_constraints, motion, start_step

    Returns
    -------
    * residual_vector: list
    """
    offset = 0
    residual_vector = np.array([])
    motion_primitive_graph, graph_walk_steps, error_scale, quality_scale, prev_frames, init_error_sum = data
    skeleton = motion_primitive_graph.skeleton
    node_name = skeleton.aligning_root_node
    for step in graph_walk_steps:
        alpha = np.array(s[offset:offset+step.n_spatial_components])
        sample_frames = motion_primitive_graph.nodes[step.node_key].back_project(alpha, use_time_parameters=False).coeffs
        step_data = motion_primitive_graph.nodes[step.node_key], step.motion_primitive_constraints,\
                       prev_frames, error_scale, quality_scale, 1.0
        concat_alpha = np.hstack((alpha, step.parameters[step.n_spatial_components:]))
        residual_vector = np.hstack( (residual_vector, obj_spatial_error_residual_vector_and_naturalness(concat_alpha, step_data)))
        prev_frames = align_quaternion_frames(skeleton, node_name, sample_frames, prev_frames, step.motion_primitive_constraints.start_pose)
        offset += step.n_spatial_components
    return residual_vector/init_error_sum

