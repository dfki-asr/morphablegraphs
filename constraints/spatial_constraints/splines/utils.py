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
from .parameterized_spline import ParameterizedSpline
import numpy as np
import math
from transformations import quaternion_from_euler, euler_from_quaternion

REF_VECTOR = [0.0,1.0]


def get_tangent_at_parameter(spline, u, eval_range=0.5):
    """
    Returns
    ------
    * dir_vector : np.ndarray
      The normalized direction vector
    * start : np.ndarry
      start of the tangent line / the point evaluated at arc length
    """
    tangent = [1.0,0.0]
    magnitude = 0
    while magnitude == 0:  # handle cases where the granularity of the spline is too low
        l1 = u - eval_range
        l2 = u + eval_range
        p1 = spline.query_point_by_absolute_arc_length(l1)
        p2 = spline.query_point_by_absolute_arc_length(l2)
        tangent = p2 - p1
        magnitude = np.linalg.norm(tangent)
        eval_range += 0.1
        if magnitude != 0:
            tangent /= magnitude
    return tangent


def get_angle_between_vectors(a,b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    angle = math.acos((a[0] * b[0] + a[1] * b[1]))
    return angle


def get_tangents2d(translation, eval_range=0.5):
    """ Create a list of tangents for a list of translations to be used for an AnnotatedSpline"""
    """ TODO fix """
    steps = len(translation)
    spline = ParameterizedSpline(translation)
    parameters = np.linspace(0, 1, steps)# this is not correct
    tangents = []
    for u in parameters:
        tangent = get_tangent_at_parameter(spline, u, eval_range)
        tangents.append(tangent)
    return tangents


def complete_tangents(translation, given_tangents, eval_range=0.5):
    steps = len(translation)
    spline = ParameterizedSpline(translation)
    parameters = [spline.get_absolute_arc_length_of_point(t)[0] for t in translation]
    #parameters = np.linspace(0, 1, steps)
    tangents = given_tangents
    for idx, u in enumerate(parameters):
        if tangents[idx] is None:
            tangents[idx] = get_tangent_at_parameter(spline, u, eval_range)
    return tangents


def complete_orientations_from_tangents(translation, given_orientations, eval_range=0.5, ref_vector=REF_VECTOR):
    steps = len(translation)
    spline = ParameterizedSpline(translation)
    parameters = np.linspace(0, 1, steps)
    orientations = given_orientations
    for idx, u in enumerate(parameters):
        if orientations[idx] is None:
            tangent = get_tangent_at_parameter(spline, u, eval_range)
            print("estimate tangent",idx, tangent)
            orientations[idx] = tangent_to_quaternion(tangent, ref_vector)
    return orientations


def tangent_to_quaternion(tangent, ref_vector=REF_VECTOR):
    a = ref_vector
    b = np.array([tangent[0], tangent[2]])
    angle = get_angle_between_vectors(a, b)
    return quaternion_from_euler(0, angle, 0)

def quaternion_to_tangent(q, ref_vector=REF_VECTOR):
    e = euler_from_quaternion(q)
    angle = e[1]
    sa = math.sin(angle)
    ca = math.cos(angle)
    m = np.array([[ca, -sa], [sa, ca]])
    return np.dot(m, ref_vector)


def tangents_to_quaternions(tangents, ref_vector=REF_VECTOR):
    quaternions = []
    for tangent in tangents:
        q = tangent_to_quaternion(tangent, ref_vector)
        quaternions.append(q)
    return quaternions


def get_orientations_from_tangents2d(translation, ref_vector=REF_VECTOR):
    """ Create a list of orientations for a list of translations to be used for an AnnotatedSpline.
        Note it seems that as long as the number of points are the same, the same spline parameters can be used for the
        query of the spline.
    """
    """ TODO fix """
    ref_vector = np.array(ref_vector)
    steps = len(translation)
    spline = ParameterizedSpline(translation)
    parameters = np.linspace(0,1, steps)
    orientation = []
    for u in parameters:
        tangent = get_tangent_at_parameter(spline, u, eval_range=0.1)
        a = ref_vector
        b = np.array([tangent[0], tangent[2]])
        angle = get_angle_between_vectors(a, b)
        orientation.append(quaternion_from_euler(*np.radians([0, angle, 0])))
    return orientation


def get_tangents(points, length):
    spline = ParameterizedSpline(points)
    x = np.linspace(0, spline.full_arc_length, length)
    new_points = []
    tangents = []
    for v in x:
        s, t = spline.get_tangent_at_arc_length(v)
        new_points.append(s)
        tangents.append(t)
    return new_points, tangents

def plot_annotated_spline(spline,root_motion, filename, scale_factor=0.7):
    from matplotlib import pyplot as plt

    def plot_annotated_tangent(ax, spline, x, length):

        p = spline.query_point_by_absolute_arc_length(x)*scale_factor
        t = spline.query_orientation_by_absolute_arc_length(x)*scale_factor
        start = -p[0], p[2]
        p_prime = [-p[0] + -t[0] * length, p[2] + t[2] * length]
        # p_prime = [p[0] + t[0] * length, p[2] + t[2] * length]
        points = np.array([start, p_prime]).T
        # t = tangent.T.tolist()
        #
        # print "tangent", t
        ax.plot(*points)


    fig = plt.figure()
    sub_plot_coordinate = (1, 1, 1)
    ax = fig.add_subplot(*sub_plot_coordinate)

    control_points = spline.spline.control_points
    control_points = np.array(control_points).T
    points = []
    for v in np.linspace(0,spline.full_arc_length,100):
        p = spline.query_point_by_absolute_arc_length(v)
        p = np.array([-p[0], p[2]])
        points.append(p*scale_factor)

    points = np.array(points).T
    all_points = points[0].tolist() + points[1].tolist()
    min_v = min(all_points)
    max_v = max(all_points)
    ax.set_xlim([min_v - 50, max_v + 50])
    ax.set_ylim([min_v - 50, max_v + 50])
    x = points[0]
    z = points[1]
    ax.plot(x,z)

    ax.scatter(-control_points[0]*scale_factor, control_points[2]*scale_factor)
    root_motion = root_motion.T*scale_factor
    ax.plot(-root_motion[0], root_motion[2])
    for v in np.linspace(0,spline.full_arc_length,100):
        plot_annotated_tangent(ax, spline, v, length=10)
    fig.savefig(filename,  format="png")
