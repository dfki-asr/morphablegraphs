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
from scipy.optimize import curve_fit
import numpy as np


class Plane(object):
    def __init__(self, points, **kwargs):
        if len(points) == 1 and 'normal_vector' in kwargs:
            self.p1 = np.array(points[0])
            self.normal_vector = np.array(kwargs.get('normal_vector'))
        elif len(points) == 3:
            self.p1 = np.array(points[0])
            self.normal_vector = Plane.calculate_normal_vector(points)
        elif len(points) > 3:
            self.p1, self.normal_vector = plane_fitting(points)
        else:
            raise NotImplementedError('cannot initialize plane')

    @staticmethod
    def calculate_normal_vector(points):
        '''
        Calculate normal vector from 3 points, the direction of normal vector is decided by right hand rule
        :param points: numpy.array<3d>
        :return: numpy.array<3d>
        '''
        if Plane.are_collinear(points):
            raise NotImplementedError('Enter three non-collinear points')
        vec12 = np.array(points[0] - points[1])
        vec13 = np.array(points[0] - points[2])
        normal_vector = np.cross(vec12, vec13)
        return normal_vector/np.linalg.norm(normal_vector)

    @staticmethod
    def are_collinear(points):
        assert(len(points) == 3), ('At least 3 points are needed to define a plane.')
        vec12 = np.array(points[0]) - np.array(points[1])
        vec13 = np.array(points[0]) - np.array(points[2])
        res = np.dot(vec12, vec13)/(np.linalg.norm(vec12)*np.linalg.norm(vec13))
        return np.isclose(res, 1.0)

    def is_before_plane(self, point):
        '''
        Test given point is before the plane or not
        :param point: numpy.array<3d>
        :return: bool
        '''
        dir_vec = np.asarray(point) - self.p1
        dir_vec = dir_vec/np.linalg.norm(dir_vec)
        if np.isclose(np.dot(dir_vec, self.normal_vector), 0):
            return False
        elif np.dot(dir_vec, self.normal_vector) > 0:
            return True
        else:
            return False

    def distance(self, point):
        '''
        Calculate the directed distance from a point to the plane
        :param point: umpy.array<3d>
        :return: float
        '''
        offset = np.asarray(point) - self.p1
        distance = np.dot(offset, self.normal_vector/np.linalg.norm(self.normal_vector))
        return distance


def func(x, a, b, c):
    z = a*x[:, 0] + b*x[:, 1] + c
    return z


def plane_fitting(points):
    X = points[:, :2]
    Y = points[:, 2]
    params = curve_fit(func, X, Y)[0]
    point = np.array([0, 0, params[2]])
    normal_vector = np.array([params[0], params[1], 1])
    return point, normal_vector/np.linalg.norm(normal_vector)
