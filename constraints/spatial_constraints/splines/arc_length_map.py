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


class RelativeArcLengthMap(object):
    """ Contains a table from relative or normalized arc length in range 0 to 1
        to a spline parameter also in range 0 to 1.
    """
    LOWER_VALUE_SEARCH_FOUND_EXACT_VALUE = 0
    LOWER_VALUE_SEARCH_FOUND_LOWER_VALUE = 1
    LOWER_VALUE_SEARCH_VALUE_TOO_SMALL = 2
    LOWER_VALUE_SEARCH_VALUE_TOO_LARGE = 3

    def __init__(self, spline, granularity):
        self.granularity = granularity
        self.full_arc_length = 0
        self._relative_arc_length_map = []
        self._update_table(spline)

    def clear(self):
        self.full_arc_length = 0
        self._relative_arc_length_map = []

    def _update_table(self, spline):
        """
        creates a table that maps from parameter space of query point to relative arc length based on the given granularity in the constructor of the catmull rom spline
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        """
        self.full_arc_length = 0
        granularity = self.granularity
        accumulated_steps = np.arange(granularity + 1) / float(granularity)
        last_point = None
        number_of_evaluations = 0
        self._relative_arc_length_map = []
        for accumulated_step in accumulated_steps:
            point = spline.query_point_by_parameter(accumulated_step)
            #print point
            if last_point is not None:
                self.full_arc_length += np.linalg.norm(point-last_point)
            self._relative_arc_length_map.append([accumulated_step, self.full_arc_length])
            number_of_evaluations += 1
            last_point = point
        if self.full_arc_length == 0:
            raise ValueError("Not enough control points in trajectory constraint definition")

        # normalize values
        if self.full_arc_length > 0:
            for i in range(number_of_evaluations):
                self._relative_arc_length_map[i][1] /= self.full_arc_length
        return self.full_arc_length

    def get_absolute_arc_length(self, t):
        """Returns the absolute arc length given a parameter value
        #SLIDE 29
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        gets absolute arc length based on t in relative arc length [0..1]
        """
        step_size = 1.0 / self.granularity
        table_index = int(t / step_size)
        if table_index < len(self._relative_arc_length_map) - 1:
            t_i = self._relative_arc_length_map[table_index][0]
            t_i_1 = self._relative_arc_length_map[table_index + 1][0]
            a_i = self._relative_arc_length_map[table_index][1]
            a_i_1 = self._relative_arc_length_map[table_index + 1][1]
            arc_length = a_i + ((t - t_i) / (t_i_1 - t_i)) * (a_i_1 - a_i)
            arc_length *= self.full_arc_length
        else:
            arc_length = self._relative_arc_length_map[table_index][1] * self.full_arc_length
        return arc_length

    def map_relative_arc_length_to_parameter(self, relative_arc_length):
        """
        #see slide 30 b
         http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        #note it does a binary search so it is rather expensive to be called at every frame
        """
        floorP, ceilP, floorL, ceilL, found_exact_value = self._find_closest_values_in_relative_arc_length_map(relative_arc_length)
        if not found_exact_value:
            alpha = (relative_arc_length - floorL) / (ceilL - floorL)
            #t = floorL+alpha*(ceilL-floorL)
            return floorP + alpha * (ceilP - floorP)
        else:
            return floorP

    def _find_closest_values_in_relative_arc_length_map(
            self, relative_arc_length):
        """ Given a relative arc length between 0 and 1 it uses get_closest_lower_value
        to search the self._relative_arc_length_map for the values bounding the searched value
        Returns
        -------
        floor parameter,
        ceiling parameter,
        floor arc length,
        ceiling arc length
        and a bool if the exact value was found
        """
        found_exact_value = True
        # search for the index and a flag value, requires a getter for the
        # array
        result = RelativeArcLengthMap.get_closest_lower_value(self._relative_arc_length_map, 0,
                                                             len(self._relative_arc_length_map) - 1,
                                                             relative_arc_length,
                                                             getter=lambda A, i: A[i][1])
        # print result
        index = result[0]
        if result[1] == self.LOWER_VALUE_SEARCH_VALUE_TOO_SMALL:  # value smaller than smallest element in the array, take smallest value
            ceilP = self._relative_arc_length_map[index][0]
            floorL = self._relative_arc_length_map[index][1]
            floorP = ceilP
            ceilL = floorL
            #found_exact_value = True
        elif result[1] == self.LOWER_VALUE_SEARCH_VALUE_TOO_LARGE:  # value larger than largest element in the array, take largest value
            ceilP = self._relative_arc_length_map[index][0]
            ceilL = self._relative_arc_length_map[index][1]
            floorP = ceilP
            floorL = ceilL
            #found_exact_value = True
        elif result[1] == self.LOWER_VALUE_SEARCH_FOUND_EXACT_VALUE:  # found exact value
            floorP, ceilP = self._relative_arc_length_map[index][0], self._relative_arc_length_map[index][0]
            floorL, ceilL = self._relative_arc_length_map[index][1], self._relative_arc_length_map[index][1]
            #found_exact_value = True
        elif result[1] == self.LOWER_VALUE_SEARCH_FOUND_LOWER_VALUE:  # found lower value
            floorP = self._relative_arc_length_map[index][0]
            floorL = self._relative_arc_length_map[index][1]
            if index < len(self._relative_arc_length_map):  # check array bounds
                ceilP = self._relative_arc_length_map[index + 1][0]
                ceilL = self._relative_arc_length_map[index + 1][1]
                found_exact_value = False
            else:
                #found_exact_value = True
                ceilP = floorP
                ceilL = floorL

        # print relative_arc_length,floorL,ceilL,found_exact_value
        return floorP, ceilP, floorL, ceilL, found_exact_value

    @staticmethod
    def get_closest_lower_value(arr, left, right, value, getter=lambda A, i: A[i]):
        """
        Uses a modification of binary search to find the closest lower value
        Note this algorithm was copied from http://stackoverflow.com/questions/4257838/how-to-find-closest-value-in-sorted-array
        - left smallest index of the searched range
        - right largest index of the searched range
        - arr array to be searched
        - parameter is an optional lambda function for accessing the array
        - returns a tuple (index of lower bound in the array, flag: 0 = exact value was found, 1 = lower bound was returned, 2 = value is lower than the minimum in the array and the minimum index was returned, 3= value exceeds the array and the maximum index was returned)
        """

        delta = int(right - left)
        if delta > 1:  #test if there are more than two elements to explore
            i_mid = int(left + ((right - left) / 2))
            test_value = getter(arr, i_mid)
            if test_value > value:
                return RelativeArcLengthMap.get_closest_lower_value(arr, left, i_mid, value, getter)
            elif test_value < value:
                return RelativeArcLengthMap.get_closest_lower_value(arr, i_mid, right, value, getter)
            else:
                return i_mid, RelativeArcLengthMap.LOWER_VALUE_SEARCH_FOUND_EXACT_VALUE
        else:  # always return the lowest closest value if no value was found, see flags for the cases
            left_value = getter(arr, left)
            right_value = getter(arr, right)
            if value >= left_value:
                if value <= right_value:
                    return left, RelativeArcLengthMap.LOWER_VALUE_SEARCH_FOUND_LOWER_VALUE
                else:
                    return right, RelativeArcLengthMap.LOWER_VALUE_SEARCH_VALUE_TOO_SMALL
            else:
                return left, RelativeArcLengthMap.LOWER_VALUE_SEARCH_VALUE_TOO_LARGE
