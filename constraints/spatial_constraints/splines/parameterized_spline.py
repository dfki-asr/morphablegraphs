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
Created on Fri Jul 10 11:28:22 2015

@author: Erik Herrmann
"""
import numpy as np
from math import sqrt, acos
from .catmull_rom_spline import CatmullRomSpline
from .segment_list import SegmentList
from .b_spline import BSpline
from .fitted_b_spline import FittedBSpline
from scipy.optimize import minimize
from .arc_length_map import RelativeArcLengthMap
SPLINE_TYPE_CATMULL_ROM = 0
SPLINE_TYPE_BSPLINE = 1
SPLINE_TYPE_FITTED_BSPLINE = 2


class ParameterizedSpline(object):
    """ Parameterize a spline by arc length using a mapping table from parameter
    to relative arch length

    Implemented based on the following resources and examples:
    #http://www.cs.cmu.edu/~462/projects/assn2/assn2/catmullRom.pdf
    #http://algorithmist.net/docs/catmullrom.pdf
    #http://www.mvps.org/directx/articles/catmull/
    #http://hawkesy.blogspot.de/2010/05/catmull-rom-spline-curve-implementation.html
    #http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
    """

    def __init__(self, control_points,  spline_type=SPLINE_TYPE_CATMULL_ROM,
                 granularity=1000, closest_point_search_accuracy=0.001,
                 closest_point_search_max_iterations=5000, verbose=False):
        if spline_type == SPLINE_TYPE_CATMULL_ROM:
            self.spline = CatmullRomSpline(control_points, verbose=verbose)
        elif spline_type == SPLINE_TYPE_BSPLINE:
            self.spline = BSpline(control_points)
        elif spline_type == SPLINE_TYPE_FITTED_BSPLINE:
            self.spline = FittedBSpline(control_points, degree=1)
        else:
            raise NotImplementedError()
        self.granularity = granularity
        self.number_of_segments = 0
        self.arc_length_map = RelativeArcLengthMap(self.spline, granularity)
        self.full_arc_length = self.arc_length_map.full_arc_length
        self.closest_point_search_accuracy = closest_point_search_accuracy
        self.closest_point_search_max_iterations = closest_point_search_max_iterations

    def _initiate_control_points(self, control_points):
        """
        @param ordered control_points array of class accessible by control_points[index][dimension]
        """
        self.spline._initiate_control_points(control_points)
        self.full_arc_length = self.arc_length_map._update_table(self.spline)

    def add_control_point(self, point):
        """
        Adds the control point at the end of the control point sequence
        """
        if self.spline.initiated:
            self.spline.add_control_point(point)
            self.full_arc_length = self.arc_length_map._update_table(self.spline)
        else:
            self._initiate_control_points([point, ])

    def clear(self):
        self.spline.clear()
        self.full_arc_length = 0
        self.number_of_segments = 0
        self.arc_length_map.clear()

    def transform_by_matrix(self, matrix):
        """
        matrix nxn transformation matrix where n is the number of dimensions of the catmull rom spline
        """
        self.spline.transform_by_matrix(matrix)

    def get_full_arc_length(self, granularity=1000):
        """
        Apprioximate the arc length based on the sum of the finite difference using
        a step size found using the given granularity
        """
        accumulated_steps = np.arange(granularity + 1) / float(granularity)
        arc_length = 0.0
        last_point = np.zeros((self.spline.dimensions, 1))
        for accumulated_step in accumulated_steps:
            # print "sample",accumulated_step
            point = self.spline.query_point_by_parameter(accumulated_step)
            if point is not None:
                #delta = []
                #d = 0
                #while d < self.spline.dimensions:
                #    sq_k = (point[d] - last_point[d])**2
                #    delta.append(sq_k)
                #    d += 1
                arc_length += np.linalg.norm(point-last_point)#sqrt(np.sum(delta))
                last_point = point
                # print point
            else:
                raise ValueError(
                    'queried point is None at %f' %
                    (accumulated_step))

        return arc_length

    def query_point_by_parameter(self, u):
        return self.spline.query_point_by_parameter(u)

    def query_point_by_absolute_arc_length(self, absolute_arc_length):
        """
        normalize absolute_arc_length and call query_point_by_relative_arc_length
        SLIDE 30 a
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        """

        #point = np.zeros((1, self.spline.dimensions))  # source of bug
        if absolute_arc_length <= self.full_arc_length:
            # parameterize curve by arc length
            relative_arc_length = absolute_arc_length / self.full_arc_length
            return self.query_point_by_relative_arc_length(relative_arc_length)
            #point = self.query_point_by_parameter(relative_arc_length)
        else:
            # return last control point
            return self.spline.get_last_control_point()
            #raise ValueError('%f exceeded arc length %f' % (absolute_arc_length,self.full_arc_length))
        #return point

    def query_point_by_relative_arc_length(self, relative_arc_length):
        """Converts relative arc length into a spline parameter between 0 and 1
            and queries the spline for the point.
        """
        u = self.arc_length_map.map_relative_arc_length_to_parameter(relative_arc_length)
        return self.spline.query_point_by_parameter(u)

    def get_last_control_point(self):
        """
        Returns the last control point ignoring the last auxilliary point
        """
        return self.spline.get_last_control_point()

    def get_distance_to_path(self, absolute_arc_length, point):
        """ Evaluates a point with absoluteArcLength on self to get a point on the path
        then the distance between the given position and the point on the path is returned
        """
        point_on_path = self.query_point_by_absolute_arc_length(absolute_arc_length)
        return np.linalg.norm(point - point_on_path)

    def get_min_control_point(self, arc_length):
        """yields the first control point with a greater abs arclength than the
        given one"""
        min_index = 0
        num_points = len(self.spline.control_points) - 3
        index = 1
        while index < num_points:
            eval_arc_length, eval_point = self.get_absolute_arc_length_of_point(self.spline.control_points[index])
            print('check arc length', index, eval_arc_length)
            if arc_length < eval_arc_length:
                min_index = index
                break
            index += 1

        return min_index

    def get_tangent_at_arc_length(self, arc_length, eval_range=0.5):
        """
        Returns
        ------
        * dir_vector : np.ndarray
          The normalized direction vector
        * start : np.ndarry
          start of the tangent line / the point evaluated at arc length
        """
        start = self.query_point_by_absolute_arc_length(arc_length)
        magnitude = 0
        while magnitude == 0:  # handle cases where the granularity of the spline is too low
            l1 = arc_length - eval_range
            l2 = arc_length + eval_range
            p1 = self.query_point_by_absolute_arc_length(l1)
            p2 = self.query_point_by_absolute_arc_length(l2)
            dir_vector = p2 - p1
            magnitude = np.linalg.norm(dir_vector)
            eval_range += 0.1
            if magnitude != 0:
                dir_vector /= magnitude
        return start, dir_vector

    def get_direction(self):
        dir_vector = self.spline.query_point_by_parameter(1.0) - self.spline.query_point_by_parameter(0.0)
        magnitude = np.linalg.norm(dir_vector)
        if magnitude != 0:
            dir_vector /= magnitude
        return dir_vector


    def get_angle_at_arc_length_2d(self, arc_length, reference_vector):
        """
        Parameters
        ---------
        * arc_length : float
          absolute arc length for the evaluation of the spline
        * reference_vector : np.ndarray
          2D vector

        Returns
        ------
        * angle : float
          angles in degrees

        """
        assert self.spline.dimensions == 3
        start, tangent_line = self.get_tangent_at_arc_length(arc_length)
        # todo get angle with reference_frame[1]
        a = reference_vector
        b = np.array([tangent_line[0], tangent_line[2]])
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        angle = acos((a[0] * b[0] + a[1] * b[1]))
        return start, tangent_line, np.degrees(angle)

    def get_absolute_arc_length_of_point(self, point, min_arc_length=0):
        """ Finds the approximate arc length of a point on a spline
        Returns
        -------
        * arc_length : float
          arc length that leads to the closest point on the path with the given
          accuracy. If input point does not lie on the path, i.e. the accuracy
          condition can not be fullfilled -1 is returned)
        """
        u = 0.0
        step_size = 1.0 / self.granularity
        min_distance = np.inf
        min_u = None
        while u <= 1.0:
            if self.arc_length_map.get_absolute_arc_length(u) > min_arc_length:
                eval_point = self.spline.query_point_by_parameter(u)
                #delta = eval_point - point
                #distance = 0
                #for v in delta:
                #    distance += v**2
                #distance = sqrt(distance)
                distance = np.linalg.norm(eval_point-point)
                if distance < min_distance:
                    min_distance = distance
                    min_u = u
                    min_point = eval_point
            u += step_size

        if min_u is not None:
            return self.arc_length_map.get_absolute_arc_length(min_u), min_point
        else:
            return -1, None

    def find_closest_point(self, point, min_arc_length=0, max_arc_length=-1):
        """ Find closest segment by dividing the closest segments until the
            difference in the distance gets smaller than the accuracy
        Returns
        -------
        * closest_point :  np.ndarray
            point on the spline
        * distance : float
            distance to input point
        """

        eps = 0.0
        if min_arc_length >= self.full_arc_length-eps:  # min arc length was too close to full arc length
            return self.get_last_control_point(), 0.0
        else:
            segment_list = SegmentList(self.closest_point_search_accuracy, self.closest_point_search_max_iterations)
            range_large_enough = segment_list.construct_from_spline(self, min_arc_length, max_arc_length)
            if range_large_enough:
                result = segment_list.find_closest_point(point)
                if result[0] is None:
                    print("Failed to generate trajectory segments for the closest point search")
                    print(point, min_arc_length, max_arc_length)
                    return None, -1
                else:
                    return result
            else:
                return self.get_last_control_point(), 0.0

    def find_closest_point_fast(self, point, min_arc_length=0, max_arc_length=-1):
        def dist_objective(x, spline, target):
            eval_point = spline.query_point_by_parameter(x)
            #print eval_point, target
            return np.linalg.norm(eval_point-target)
        data = self, point

        #cons = (
		#	{"type": 'ineq',
		#	 "fun": lambda x: 1.0 - x})
        min_u = min_arc_length #/ self.full_arc_length0.0
        guess_t = np.array([min_arc_length]).flatten()
        if max_arc_length >0:
            max_u = max_arc_length #/ self.full_arc_length
        else:
            max_u = 1.0
        #print "bounds",min_u, max_u
        result = minimize(dist_objective, guess_t, data, method="L-BFGS-B", bounds=[(min_u, max_u)])
        #print dist_objective(result['x'], self, point)
        return self.spline.query_point_by_parameter(result['x']), result['x']


    def get_absolute_arc_length(self, u):
        return self.arc_length_map.get_absolute_arc_length(u)