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
Created on Wed Feb 25 14:54:37 2015

@author: Erik Herrmann
"""

import numpy as np
from math import floor, sqrt


class CatmullRomSpline(object):
    """
    Implements a Catmull-Rom Spline

    implemented based on the following resources and examples:
    #http://www.cs.cmu.edu/~462/projects/assn2/assn2/catmullRom.pdf
    #http://algorithmist.net/docs/catmullrom.pdf
    #http://www.mvps.org/directx/articles/catmull/
    #http://hawkesy.blogspot.de/2010/05/catmull-rom-spline-curve-implementation.html
    #http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
    """

    def __init__(self, control_points, verbose=False):
        self.verbose = verbose

        # http://algorithmist.net/docs/catmullrom.pdf
        # base matrix to calculate one component of a point on the spline based
        # on the influence of control points
        self._catmullrom_basematrix = np.array([[-1.0, 3.0, -3.0, 1.0],
                                             [2.0, -5.0, 4.0, -1.0],
                                             [-1.0, 0.0, 1.0, 0.0],
                                             [0.0, 2.0, 0.0, 0.0]])
        if isinstance(control_points[0], (int, float, complex)):
            self.dimensions = 1
        else:
            self.dimensions = len(control_points[0])
        self.initiated = False
        self.control_points = []
        if len(control_points) > 0:
            self._initiate_control_points(control_points)
            self.initiated = True

    def _initiate_control_points(self, control_points):
        """
        @param ordered control_points array of class accessible by control_points[index][dimension]
        """
        self.number_of_segments = len(control_points) - 1
        self.control_points = [control_points[0]] + control_points + [control_points[-1], control_points[-1]]
        if self.verbose:
            print("length of control point list ", len(self.control_points))
            print("number of segments ", self.number_of_segments)
            print("number of dimensions", self.dimensions)
        return

    def add_control_point(self, point):
        """
        Adds the control point at the end of the control point sequence
        """
        # add point replace auxiliary control points
        if self.initiated:
            del self.control_points[-2:]
            self.number_of_segments = len(self.control_points) - 1
            self.control_points += [point, point, point]
        else:
            self._initiate_control_points([point, ])
            self.initiated = True

    def clear(self):
        self.control_points = []
        self.initiated = False

    def transform_by_matrix(self, matrix):
        """
        matrix nxn transformation matrix where n is the number of dimensions of the catmull rom spline
        """
        if self.dimensions < matrix.shape[0]:
            for i in range(len(self.control_points)):
                self.control_points[i] = np.dot(
                    matrix, self.control_points[i] + [1])[:3]
    #             print "t",i
        else:
            print("Failed to transform spline by matrix", matrix.shape)
        return

    def get_last_control_point(self):
        """
        Returns the last control point ignoring the last auxiliary point
        """
        if len(self.control_points) > 0:
            return np.array(self.control_points[-1])
        else:
            print("no control points defined")
            return np.zeros((1, self.dimensions))

    def map_parameter_to_segment(self, u):
        """
        Returns the segment index associated with parameter u in range
        [0..1]
        Returns the index of the segment and the corresponding relative parameter value in this segment
        """
        #get index of the segment
        scaled_u = self.number_of_segments * u
        index = min(int(floor(scaled_u)), self.number_of_segments)
        # local_u is the remainder, e.g. number_of_segments = 10 and u = 0.62 then index = 6 and local_u is 0.02
        local_u = scaled_u - index #floor(self.number_of_segments * u)
        # increment i by 1 to ignore the first auxiliary control point
        return index + 1, local_u

    def query_point_by_parameter(self, u):
        """
        Slide 32
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        Returns a point on the curve by parametric value u in the range between
        0 and 1.
        P
        """
        segment_index, local_u = self.map_parameter_to_segment(u)
        #print u, segment_index, local_u,self.number_of_segments
        if segment_index <= self.number_of_segments:
            # defines the influence of the 4 closest control points
            weight_vector = [local_u**3, local_u**2, local_u, 1]
            control_point_vectors = self._get_control_point_vectors(segment_index)
            point = []
            d = 0
            while d < self.dimensions:
                point.append(self._query_component_by_parameter(
                             weight_vector, control_point_vectors[d]))
                d += 1
            return np.array(point)
        else:
            return self.control_points[-1]

    def _query_component_by_parameter(
            self, weight_vector, control_point_vector):
        """
        Slide 32
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        Queries a component of point on the curve based on control points
        and their weights and the _catmullrom_basematrix
        """
        transformed_control_point_vector = np.dot(
            self._catmullrom_basematrix, control_point_vector)
        value = np.dot(weight_vector, transformed_control_point_vector)
        return 0.5 * value

    def _get_control_point_vectors(self, index):
        """
        Returns the 4 control points that influence values within the i-th segment
        Note the auxiliary segments should not be queried
        """
        #assert index <= self.number_of_segments
        #index = int(index)
        d = 0
        vectors = []
        while d < self.dimensions:
            vectors.append([self.control_points[index - 1][d],
                            self.control_points[index][d],
                            self.control_points[index + 1][d],
                            self.control_points[index + 2][d]])
            d += 1
        return vectors
