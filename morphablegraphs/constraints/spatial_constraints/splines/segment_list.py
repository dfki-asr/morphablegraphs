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
import heapq
import numpy as np
from math import sqrt
from .spline_segment import SplineSegment


class SegmentList(object):
    def __init__(self, closest_point_search_accuracy=0.001, closest_point_search_max_iterations=5000, segments=None):
        self.segments = segments
        self.closest_point_search_accuracy = closest_point_search_accuracy
        self.closest_point_search_max_iterations = closest_point_search_max_iterations

    def construct_from_spline(self, spline, min_arc_length=0, max_arc_length=-1, granularity=1000):
        """ Constructs line segments out of the evualated points
         with the given granularity
        Returns
        -------
        * segments : list of tuples
            Each entry defines a line segment and contains
            start,center and end points
        Returns
        -------
        True if successful else if not
        """
        points = []
        step_size = 1.0 / granularity
        if max_arc_length <= 0:
            max_arc_length = spline.full_arc_length
        if abs(min_arc_length-max_arc_length) > step_size:
            u = 0
            while u <= 1.0:
                arc_length = spline.get_absolute_arc_length(u)
                # TODO make more efficient by looking up min_u
                if arc_length >= min_arc_length and arc_length <= max_arc_length:
                    point = spline.query_point_by_parameter(u)
                    points.append(point)
                u += step_size
            self.segments = []
            index = 0
            while index < len(points) - 1:
                start = np.array(points[index])
                end = np.array(points[index + 1])
                center = 0.5 * (end - start) + start
                segment = SplineSegment(start, center, end)
                self.segments.append(segment)
                index += 1
            return index > 0
        else:
            return False

    def find_closest_point(self, point):
        if self.segments is None or len(self.segments) == 0:
            return None, -1
        candidates = self.find_two_closest_segments(point)
        if len(candidates) >= 2:
            closest_point_1, distance_1 = self._find_closest_point_on_segment(candidates[0][1], point)
            closest_point_2, distance_2 = self._find_closest_point_on_segment(candidates[1][1], point)
            if distance_1 < distance_2:
                return closest_point_1, distance_1
            else:
                return closest_point_2, distance_2
        elif len(candidates) == 1:
            closest_point, distance = self._find_closest_point_on_segment(candidates[0][1], point)
            return closest_point, distance

    def find_closest_segment(self, point):
        """
        Returns
        -------
        * closest_segment : Tuple
           Defines line segment. Contains start,center and end
        * min_distance : float
          distance to this segments center
        """
        closest_segment = None
        min_distance = np.inf
        for s in self.segments:
            distance = np.linalg.norm(s.center-point)
            if distance < min_distance:
                closest_segment = s
                min_distance = distance
        return closest_segment, min_distance

    def find_two_closest_segments(self, point):
        """ Ueses a heap queue to find the two closest segments
        Returns
        -------
        * closest_segments : List of Tuples
           distance to the segment center
           Defineiation of a line segment. Contains start,center and end points

        """
        heap = []  # heap queue
        idx = 0
        while idx < len(self.segments):
            distance = np.linalg.norm(self.segments[idx].center-point)
#            print point,distance,segments[index]
#            #Push the value item onto the heap, maintaining the heap invariant.
            heapq.heappush(heap, (distance, idx))
            idx += 1

        closest_segments = []
        count = 0
        while idx-count > 0 and count < 2:
            distance, index = heapq.heappop(heap)
            segment = (distance, self.segments[index])
            closest_segments.append(segment)
            count += 1
        return closest_segments

    def _find_closest_point_on_segment(self, segment, point):
            """ Find closest point by dividing the segment until the
                difference in the distance gets smaller than the accuracy
            Returns
            -------
            * closest_point :  np.ndarray
                point on the spline
            * distance : float
                distance to input point
            """
            segment_length = np.inf
            distance = np.inf
            segment_list = SegmentList(self.closest_point_search_accuracy, self.closest_point_search_max_iterations, segment.divide())
            iteration = 0
            while segment_length > self.closest_point_search_accuracy and distance > self.closest_point_search_accuracy and iteration < self.closest_point_search_max_iterations:
                closest_segment, distance = segment_list.find_closest_segment(point)
                segment_length = np.linalg.norm(closest_segment.end-closest_segment.start)
                segment_list = SegmentList(self.closest_point_search_accuracy, self.closest_point_search_max_iterations, closest_segment.divide())
                iteration += 1
            closest_point = closest_segment.center  # extract center of closest segment
            return closest_point, distance
