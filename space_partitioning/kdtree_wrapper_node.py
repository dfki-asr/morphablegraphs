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
Created on Mon Jun 15 14:58:08 2015

@author: Erik Herrmann
"""
import uuid
import numpy as np
from .kdtree import KDTree
from . import KDTREE_WRAPPER_NODE


class KDTreeWrapper(object):
    """ Wrapper for a KDTree used as leaf of the ClusterTree.

    Parameters
    ---------
    * dim: Integer
        Number of dimensions of the data to be considered.
    """
    def __init__(self, dim):
        self.id = str(uuid.uuid1())
        self.kdtree = KDTree()
        self.dim = dim
        #self.indices = []
        self.type = KDTREE_WRAPPER_NODE

    def construct(self, data, indices):
        #self.indices = indices
        self.kdtree.construct(data[indices].tolist(), self.dim)

    def find_best_example(self, obj, data):
        return self.kdtree.find_best_example(obj, data, 1)[0]

    def find_best_example_exhaustive(self, obj, data):
        return self.kdtree.df_search(obj, data)

    def knn_interpolation(self, obj, data, k=50):
        """Searches for the k best examples and performs KNN-Interpolation
        between them to produce a new sample
        with a low objective function value.
        """
        results = self.kdtree.find_best_example(obj, data, k)
        if len(results) > 1:
            distances, points = list(zip(*results))
            weights = self._get_knn_weights(distances)
            new_point = np.zeros(len(points[0]))
            for i in range(len(weights)):
                new_point += weights[i] * np.array(points[i])
            return obj(new_point, data), new_point
        else:
            return results[0]

    def _get_knn_weights(self, distances):
        influences = []
        for distance in distances[:-1]:
            influences.append(1/distance - 1/distances[-1])
        ## calculate weight based on normalized influence
        weights = []
        sum_influence = np.sum(influences)
        for i in range(len(influences)):
            weights.append(influences[i]/sum_influence)
        return weights

    def get_desc(self):
        node_desc = dict()
        node_desc["id"] = str(self.id)
        node_desc["type"] = self.type
        node_desc["children"] = []
        node_desc["indices"] = []
        return node_desc
