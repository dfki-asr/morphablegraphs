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
@author: Han Du
"""
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from anim_utils.animation_data.motion_distance import _point_cloud_distance, _transform_invariant_point_cloud_distance

def detect_local_minima(arr):
    """ Takes an array and detects the troughs using the local minimum filter.
        Returns a boolean mask of the troughs (i.e. 1 when the pixel's value is the neighborhood minimum, 0 otherwise)
    source: 
     http://stackoverflow.com/questions/3986345/how-to-find-the-local-minima-of-a-smooth-multidimensional-array-in-numpy-efficie
     http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    background = (arr==0)
    eroded_background = morphology.binary_erosion(background, structure=neighborhood, border_value=1)
    detected_minima = local_min - eroded_background
    return local_min, np.array(np.where(detected_minima)).T

def get_global_minima(distance_matrix, candidates):
    global_minimum = np.inf
    for c in candidates:
        x = c[0]
        y = c[1]

        min = distance_matrix[x][y]
        if global_minimum > min:
            global_minimum = min

    return global_minimum

def filter_minima(distance_matrix, candidates, threshold_factor):
    # find global minimum
    global_minimum = get_global_minima(distance_matrix, candidates)
    print("global minimum", global_minimum)
    # filter local minima using the threshold
    filtered_coords = []
    for c in candidates:
        x = c[0]
        y = c[1]
        min = distance_matrix[x][y]
        if min < np.inf and min < global_minimum + (global_minimum * threshold_factor):
            filtered_coords.append([x,y])

    return filtered_coords


def extracted_filtered_minima(distance_matrix, threshold_factor):
    values, candidates = detect_local_minima(distance_matrix)
    coordinates = filter_minima(distance_matrix, candidates, threshold_factor)
    return coordinates


def argmin(values):
    min_idx = 0
    min_v = np.inf
    for idx, v in enumerate(values):
        if v < min_v:
            min_idx = idx
            min_v = v
    return min_idx


def argmin_multi(values, threshold=1.0):
    min_v = np.inf
    for idx, v in enumerate(values):
        if v < min_v:
            min_v = v

    indices = []
    for idx, v in enumerate(values):
        if v <= min_v + threshold:
            indices.append(idx)
    return indices


class KeyframeDetector(object):
    def __init__(self, skeleton):
        self._skeleton = skeleton

    def find_instance(self, point_cloud, keyframe, distance_measure=_transform_invariant_point_cloud_distance):
        distances = []
        for f in point_cloud:
            d = distance_measure(f, keyframe)
            distances.append(d)
        return argmin(distances)

    def calculate_distances(self, point_clouds, keyframe, distance_measure=_transform_invariant_point_cloud_distance):
        D = []
        for m_idx, m in enumerate(point_clouds):
            D.append([])
            for f in m:
                d = distance_measure(f, keyframe)
                D[m_idx].append(d)
        return D

    def find_instances2(self, point_clouds, keyframe):
        """
        Returns:
            result (list<Tuple>): List containing motion index and frame index
        """
        D = self.calculate_distances(point_clouds, keyframe)
        return extracted_filtered_minima(D, 5)

    def find_instances(self, point_cloud, keyframe, threshold=1.0, distance_measure=_transform_invariant_point_cloud_distance):
        distances = []
        for f in point_cloud:
            d = distance_measure(f, keyframe)
            distances.append(d)
        return argmin_multi(distances, threshold)
