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
B_SPLINE_DEGREE = 3


class FittedBSpline(object):
    def __init__(self,  points, degree=B_SPLINE_DEGREE, domain=None):
        self.points = np.array(points)
        if isinstance(points[0], (int, float, complex)):
            self.dimensions = 1
        else:
            self.dimensions = len(points[0])
        self.degree = degree
        if domain is not None:
            self.domain = domain
        else:
            self.domain = (0.0, 1.0)

        self.initiated = True
        self.spline_def = []
        points_t = np.array(points).T
        t_func = np.linspace(self.domain[0], self.domain[1], len(points)).tolist()
        for d in range(len(points_t)):
            #print d, self.dimensions
            self.spline_def.append(si.splrep(t_func, points_t[d], w=None, k=3))

    def _initiate_control_points(self):
        return

    def clear(self):
        return

    def query_point_by_parameter(self, u):
        """

        """
        point = []
        for d in range(self.dimensions):
            point.append(si.splev(u, self.spline_def[d]))
        return np.array(point)

    def get_last_control_point(self):
        return self.points[-1]