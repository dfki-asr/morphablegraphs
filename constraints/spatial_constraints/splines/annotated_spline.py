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
from .parameterized_spline import ParameterizedSpline, SPLINE_TYPE_CATMULL_ROM
from .catmull_rom_spline import CatmullRomSpline


class AnnotatedSpline(ParameterizedSpline):
    """Spline that uses the parameterization from the translation on an additional orientation spline.
       This way it is possible to query the position and orientation at a specific arc length"""
    def __init__(self, translation, orientation=None, spline_type=SPLINE_TYPE_CATMULL_ROM,
                 granularity=1000, closest_point_search_accuracy=0.001,
                 closest_point_search_max_iterations=5000, verbose=False):
        super(AnnotatedSpline, self).__init__(translation, spline_type, granularity,
                                        closest_point_search_accuracy,
                                        closest_point_search_max_iterations, verbose)
        if orientation is not None:
            assert len(translation) == len(orientation), "Error: The number of points need to be equal"
            self.orientation_spline = CatmullRomSpline(orientation)

    def query_orientation_by_absolute_arc_length(self, absolute_arc_length):
        if absolute_arc_length <= self.full_arc_length:
            relative_arc_length = absolute_arc_length / self.full_arc_length
            return self.query_orientation_by_relative_arc_length(relative_arc_length)
        else:
            return self.orientation_spline.get_last_control_point()

    def query_orientation_by_relative_arc_length(self, relative_arc_length):
        u = self.arc_length_map.map_relative_arc_length_to_parameter(relative_arc_length)
        return self.orientation_spline.query_point_by_parameter(u)

    def query_orientation_by_parameter(self, u):
        return self.orientation_spline.query_point_by_parameter(u)
