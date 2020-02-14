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
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:33:42 2015

Contains a KDTree that also implements search with an objective function 
instead of a distance measure.

@author: Erik Herrmann
"""

from operator import itemgetter
import heapq
import numpy as np
from . import LEAF_NODE, INNER_NODE


class Node(object):
    """
    Node for KDTree modified from
    https://en.wikipedia.org/wiki/K-d_tree
    """
    def __init__(self, data, dim, depth=0):
        self.index = depth
        if len(data) > 1:
            self.type = INNER_NODE
            # Select axis based on depth so that axis cycles through all valid values
            axis = depth % dim

            # Sort point list and choose median as pivot element
            data.sort(key=itemgetter(axis))
            median = len(data) // 2
            left_data = data[:median]
            right_data = data[median+1:]

            # Create node and construct subtrees
            self.point = data[median]

            if len(left_data) > 0:
                self.left = Node(left_data, dim, depth+1)
            else:
                self.left = None
                
            if len(right_data) > 0:
                self.right = Node(right_data, dim, depth+1)
            else:
                self.right = None
        else:
            self.type = LEAF_NODE
            self.point = data[0]
            self.left = None
            self.right = None


class KDTree(object):
    """
    K-Dimensional space partitioning data structure modified from
    https://en.wikipedia.org/wiki/K-d_tree with additional search functions 
    based on an objective function.
    """
    def __init__(self):
        self.data = None
        self.root = None
        self.global_bb = None
              
    def construct(self, data, dim):
        self.data = data
        self.root = Node(data, dim)

    def _decide_direction_distance(self, target, left, right, distance):
        """ Chooses between left and right node allowing either to be None.
        Parameters
        ---------
        * left : Node or None
            First option.
        * right : Node or None
            Second option.
        * obj : function
            objective function returning a scalar of the form distance(x).

        Returns
        -------
        tuple containing
        * best_option: Node
            Either left or right.
        * least_cost: float
            distance for the best option.
        """
        best_option = None
        least_distance = np.inf
        if left is not None and right is not None:
            r_d = distance(target, right.point)
            l_d = distance(target, left.point)
            if l_d < r_d:
                best_option = left
                least_distance = l_d
            else:
                best_option = right
                least_distance = r_d
        elif right is not None:
            best_option = right
            least_distance = distance(target, right.point)
        elif left is not None:
            best_option = left
            least_distance = distance(target, left.point)
    #         if least_distance <= bound:
        return best_option, least_distance
    #         else:
    #             return None, 100000

    def _decide_direction_objective(self, left, right, obj, data):
        """ Chooses between left and right node allowing either to be None.
        Parameters
        ---------
        * left : Node or None
            First option.
        * right : Node or None
            Second option.
        * obj : function
            objective function returning a scalar of the form obj(x,data).
        * data : anything usually tuple
            additional parameters for the objective function.

        Returns
        -------
        * best_option: Node
            Either left or right.
        * least_cost: float
            result of obj for the best option.
        """
        best_option = None
        least_cost = np.inf
        if left is not None and right is not None:
            r_d = obj(right.point, data)
            l_d = obj(left.point, data)
            if l_d < r_d:
                best_option = left
                least_cost = l_d
            else:
                best_option = right
                least_cost = r_d
        elif right is not None:
            best_option = right
            least_cost = obj(right.point, data)
        elif left is not None:
            best_option = left
            least_cost = obj(left.point, data)
        return best_option, least_cost

    def query(self, target, k=1):
        """Finds the K nearest neighbors using a Depth First search.
            Returns
            -------
            *distances:
                list of distances to targets.
            *node:
                list of nodes.
        """
        def point_distance(a, b):
            return np.linalg.norm(a-b)
        assert k >= 1
        result_queue = [] 
        node_stack = []
        heapq.heappush(result_queue, (point_distance(self.root.point, target), self.root.point))
        node_stack.append(self.root)
        while len(node_stack) > 0:
            node = node_stack.pop(-1)
            best_option, least_distance = self._decide_direction_distance(target, node.left, node.right)
            if best_option is not None:
                heapq.heappush(result_queue, (least_distance, best_option))
        return list(zip([(n[0], n[1]) for n in result_queue[:k]]))

    def print_tree_df(self):
        """prints tree using a depth first traversal 
        """
        node_stack = []
        node_stack.append(self.root)
        element_count = 1
        while len(node_stack) > 0:
            node = node_stack.pop(-1)
            print(element_count, node.index, node.point)
            if node.type == INNER_NODE:
                if node.left is not None:
                    node_stack.append(node.left)
                if node.right is not None:
                    node_stack.append(node.right)
            element_count += 1

    def df_search(self, obj, data):
        """Depth first search to find the best of all samples
        """
        node_stack = []
        result_queue = []
        node = self.root
        heapq.heappush(result_queue, (obj(node. point, data), node.point))
        node_stack.append(self.root)
        element_count = 1
        while len(node_stack) > 0:
            node = node_stack.pop(-1)
            heapq.heappush(result_queue, (obj(node.point, data), node.point))
            if node.type == INNER_NODE:
                if node.left is not None:
                    node_stack.append(node.left)
                if node.right is not None:
                    node_stack.append(node.right)
            element_count += 1
        if len(result_queue) > 0:
            return heapq.heappop(result_queue)
        else:
            return None

    def find_best_example(self, obj, data, k=1):
        """
        Traverses the KDTree using the direction of least cost until a leaf is reached
        """
        eval_points = []
        result_queue = []
        node = self.root
        depth = 0
        eval_points.append(node.point)
        heapq.heappush(result_queue, (obj(node.point, data), depth))
   
        while node is not None and node.type == "inner":
            depth += 1
            node, least_cost = self._decide_direction_objective(node.left, node.right, obj, data)
            if node is not None:
                heapq.heappush(result_queue, (least_cost, depth))
                eval_points.append(node.point)
        return [(value, eval_points[index]) for value, index in result_queue[:k]]
