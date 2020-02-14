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
import numpy as np
import heapq
import json
import pickle as pickle
from anim_utils.utilities.log import write_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR
from .cluster_tree_node_builder import ClusterTreeNodeBuilder
from . import ROOT_NODE

DEFAULT_N_SUBDIVISIONS_PER_LEVEL = 4
DEFAULT_N_LEVELS = 4
MIN_N_SUBDIVISIONS_PER_LEVEL = 1
MIN_N_LEVELS = 1
DEFAULT_MAX_DIMENSIONS = 10


class ClusterTree(object):
    """
    Create a hiearchy of clusters using KMeans and then use a kdtree for the leafs
    Parameters
    -----------
    * n_subdivisions : Integer
        Number of subclusters/children per node in the tree. At least 2.
    * max_level : Integer
        Maximum levels in the tree. At least 1.
    
    """
    def __init__(self, n_subdivisions=DEFAULT_N_SUBDIVISIONS_PER_LEVEL, max_level=DEFAULT_N_LEVELS, dim=DEFAULT_MAX_DIMENSIONS, store_indices=False, use_kd_tree=True):
        self.n_subdivisions = max(n_subdivisions, MIN_N_SUBDIVISIONS_PER_LEVEL)
        self.max_level = max(max_level, MIN_N_LEVELS)
        self.dim = dim
        self.root = None
        self.data = None
        self.store_indices = store_indices
        self.use_kd_tree = use_kd_tree
      
    def save_to_file(self, file_name):
        fp = open(file_name+".json", "wb")
        node_desc_list = self.root.get_node_desc_list()
        node_desc_list["data_shape"] = self.data.shape
        json.dump(node_desc_list, fp, indent=4)
        fp.close()
        self.data.tofile(file_name+".data")
        
    def load_from_file(self, file_name):
        fp = open(file_name+".json", "r")
        node_desc = json.load(fp)
        fp.close()
        data_shape = node_desc["data_shape"]
        self.data = np.fromfile(file_name+".data").reshape(data_shape)
        self.dim = data_shape[1]
        node_builder = ClusterTreeNodeBuilder(self.n_subdivisions, self.max_level, self.dim, self.store_indices)
        self.root = node_builder.construct_from_node_desc_list(node_desc["root"], node_desc, self.data)

    def save_to_file_pickle(self, pickle_file_name):
        pickle_file = open(pickle_file_name, 'wb')
        pickle.dump(self, pickle_file, pickle.HIGHEST_PROTOCOL)
        pickle_file.close()
       
    def load_from_file_pickle(self, pickle_file_name):
        pickle_file = open(pickle_file_name, 'rb')
        data = pickle.load(pickle_file)
        self.data = data.data
        self.dim = self.dim
        self.root = data.root
        self.max_level = data.max_level
        self.n_subdivisions = data.n_subdivisions
        pickle_file.close()     
      
    def construct(self, data):
        self.data = data
        self.dim = min(self.data.shape[1], self.dim)
        node_builder = ClusterTreeNodeBuilder(self.n_subdivisions, self.max_level, self.dim, self.store_indices, self.use_kd_tree)
        self.root = node_builder.construct_from_data(self.data)

    def find_best_example(self, obj, data):
        return self.root.find_best_example(obj, data)

    def find_best_example_exhaustive(self, obj, data):
        return self.root.find_best_example_exhaustive(obj, data)

    def find_best_example_excluding_search(self, obj, data):
        node = self.root
        level = 0
        while level < self.max_level and not node.leaf:
            index, value = node.find_best_cluster(obj, data, use_mean=True)
            node = node.clusters[index]
            level += 1
        return node.find_best_example(obj, data)
          
    def find_best_example_excluding_search_candidates(self, obj, data, n_candidates=1):
        """ Traverses the cluster hierarchy iteratively by evaluating the means
            of the clusters at each level based on the objective function. 
            At the last level the method find_best_example for a KDTree is used.
            Multiple candidates are kept at each level in order to find the global
            optimum.
        """
        write_message_to_log("search with " + str(n_candidates) + " candidates in tree with " + str(self.n_subdivisions) + " subdivisions and "+ str(self.max_level)+ " levels", LOG_MODE_DEBUG)
        results = list()
        candidates = list()
        candidates.append((np.inf, 0, self.root))
        level = 0
        while len(candidates) > 0:
            new_candidates = list()
            for c_idx, c_data in enumerate(candidates):
                value, tie_breaker, node = c_data
                if not node.leaf:
                    good_candidates = node.find_best_cluster_candidates(obj, data, n_candidates)
                    for idx, c in enumerate(good_candidates):
                        heapq.heappush(new_candidates, (c[0], idx, c[2]))
                else:
                    v, sample = node.find_best_example(obj, data)
                    heapq.heappush(results, (v, c_idx, sample))

            candidates = new_candidates[:n_candidates]
            level += 1

        if len(results) > 0:
            r = heapq.heappop(results)
            return r[0], r[2]
        else:
            print("Error: failed to find a result")
            return np.inf, self.root.mean
        
    def find_best_example_excluding_search_candidates_boundary(self, obj, data, n_candidates=5):
        """ Traverses the cluster hierarchy iteratively by evaluating the means
            of the clusters at each level based on the objective function. 
            At the last level the method find_best_example for a KDTree is used.
            Multiple candidates are kept at each level in order to find the global
            optimum.
            Uses boundary based on maximum cost of last iteration to ignore bad candidates.
            Note requires more candidates to prevent
        """
        results = list()
        candidates = list()
        candidates.append((np.inf, self.root))
        level = 0
        while len(candidates) > 0:
            boundary = max([c[0] for c in candidates])
            print(boundary)
            new_candidates = []
            for value, node in candidates:
                
                if not node.leaf:
                    good_candidates = node.find_best_cluster_candidates(obj, data, n_candidates=n_candidates)
                    for c in good_candidates:
                         heapq.heappush(new_candidates, c)
                else:
                    kdtree_result = node.find_best_example(obj, data)
                    heapq.heappush(results, kdtree_result)

            candidates = [c for c in new_candidates[:n_candidates] if c[0] < boundary]
            if len(results) == 0 and len(candidates) == 0:
                candidates = new_candidates[:n_candidates]
            level += 1

        if len(results) > 0:
            return heapq.heappop(results)    
        else:
            print("Error: failed to find a result")
            return np.inf, self.root.mean

    def find_best_example_excluding_search_candidates_knn(self, obj, data, n_candidates=1, k=50):
        """ Traverses the cluster hierarchy iteratively by evaluating the means
            of the clusters at each level based on the objective function. 
            At the last level the method find_best_example for a KNN Interpolation is used.
            Multiple candidates are kept at each level in order to find the global
            optimum.
        """
        results = list()
        candidates = list()
        candidates.append((np.inf, self.root))
        level = 0
        while len(candidates) > 0:
            new_candidates = []
            for value, node in candidates:
                if not node.leaf:
                    good_candidates = node.find_best_cluster_candidates(obj, data, n_candidates=n_candidates)
                    for c in good_candidates:
                        heapq.heappush(new_candidates, c)
                else:
                    kdtree_result = node.find_best_example_knn(obj, data, k)
                    heapq.heappush(results, kdtree_result)

            candidates = new_candidates[:n_candidates]
            level += 1

        if len(results) > 0:
            return heapq.heappop(results)    
        else:
            print("Error: failed to find a result")
            return np.inf, self.root.mean
