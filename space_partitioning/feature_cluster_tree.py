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
from .clustering import find_clusters, all_equal
import scipy.stats.mstats
from anim_utils.utilities.log import write_message_to_log, LOG_MODE_DEBUG

MAX_SIMILARITY_CHECK = 10
PROBABILITIES = (0.25,0.75)


def _filter_outliers(values, nodes):
    """ http://www.varsitytutors.com/ap_statistics-help/how-to-find-outliers
    http://stackoverflow.com/questions/36864917/python-remove-outliers-from-data
    """

    quantiles = scipy.stats.mstats.mquantiles(values, PROBABILITIES)
    iqr = quantiles[1]-quantiles[0]
    #lower = quantiles[0]-1.5*iqr
    upper = quantiles[1]+1.5*iqr
    #indices =
    #print indices,upper, values
    # filtered_indicescandidates = candidates[filtered_indices]#callable()
    return [v for v in zip(values, nodes) if v[0] <= upper]#zip(values[indices],nodes[indices])


class FeatureClusterTree(object):
    def __init__(self, features, data, indices, options, feature_args):
        if features is not None:
            self._construct(features, data, indices, options, feature_args)


    def _construct(self, features, data, indices, options, feature_args):
        self.data = data
        self._features = features
        self._indices = indices
        self._children = []
        self._options = options
        if indices is None:
            indices = list(range(len(features)))

        if options["use_feature_mean"]:
            self._mean = np.average(features[indices], axis=0)
        else:
            self._mean = np.average(data[indices], axis=0)
        self._n_subdivisions = options["n_subdivisions"]
        self.args = feature_args
        if indices is None or len(indices) > 1:

            if indices is not None and len(indices) < MAX_SIMILARITY_CHECK:
                #all_equal = len(set(features[indices].tolist())) == 1#np.all(np.equal(vectors)) http://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
                if all_equal(features[indices]):
                    self._clusters = [[idx] for idx in indices]
                else:
                    self._clusters = find_clusters(features, indices, options)
            else:
                self._clusters = find_clusters(features, indices, options)
            print("found clusters")
            for idx, c in enumerate(self._clusters):
                print(idx, len(c))
            for c in self._clusters:
                print("create node",len(c))#,features[indices]
                if len(c) > 0:
                    if np.alltrue(c == indices):
                        for idx in c:
                            child_node = FeatureClusterTree(features, data, [idx], options, feature_args)
                            self._children.append(child_node)
                    else:
                        child_node = FeatureClusterTree(features, data, c, options, feature_args)
                        self._children.append(child_node)

    def find_best_example_excluding_search(self, obj, args):
        """ goes down the tree the path with the least cost and returns the sample at the leaf
        """
        args = list(args)+[self.args]
        if len(self._children) > 0:
            index, value = self._find_best_cluster(obj, args)#, use_mean=Truedata
            return self._children[index].find_best_example_excluding_search(obj, args)#data
        else:
            return self.data[self._indices[0]]#_mean# obj(self._mean, data)

    def _find_best_cluster(self, obj, args):
        """ Return the best cluster based on the evaluation using an objective function.

        Parameters
        ----------
        * obj: function
            Objective function of the form: scalar = obj(x,data).
        * args: Tuple
            Additional parameters for the objective function.
        """
        best_value = np.inf
        best_index = 0
        for child_index, node in enumerate(self._children):
            cluster_value = obj(node._mean, args)#args
            if cluster_value < best_value:
                best_index = child_index
                best_value = cluster_value
        return best_index, best_value

    def _find_best_cluster_candidates(self, obj, args, n_candidates):
            """Return the clusters with the least cost based on
            an evaluation using an objective
            function.

            Parameters
            ----------
            * obj: function
                Objective function of the form: scalar = obj(x,data).
            * args: Tuple
                Additional parameters for the objective function.
            * n_candidates: Integer
                Maximum number of candidates

            Returns
            -------
            * best_candidates: list of (value, ClusterTreeNode) tuples.
                List of candidates ordered using the objective function value.
            """
            result_queue = []
            for child_index, node in enumerate(self._children):
                cluster_value = obj(node._mean, args)#data
                heapq.heappush(result_queue, (cluster_value, node))
            return result_queue[:n_candidates]

    def find_best_example_excluding_search_candidates(self, obj, args, n_candidates=1):#_2_heappushs_per_canididate
        """ Traverses the cluster hierarchy iteratively by evaluating the means
            of the clusters at each level based on the objective function.
            At the last level the method find_best_example for a KDTree is used.
            Multiple candidates are kept at each level in order to find the global
            optimum.
        """
        write_message_to_log("search with " + str(n_candidates) + " candidates in tree with " +str(self._n_subdivisions) +" subdivisions ", LOG_MODE_DEBUG)
        results = list()
        candidates = list()
        candidates.append((np.inf, self))
        level = 0
        while len(candidates) > 0:
            #print "search",level,n_candidates,len(candidates)
            new_candidates = list()
            for value, node in candidates:
                if len(node._children) > 0:
                    good_candidates = node._find_best_cluster_candidates(obj, args, n_candidates)
                    for c in good_candidates:
                        heapq.heappush(new_candidates, c)
                else:
                    heapq.heappush(results, (value, node))
            candidates = new_candidates[:n_candidates]
            level += 1

        write_message_to_log("depth" +str(level), LOG_MODE_DEBUG)
        if len(results) > 0:
            value, node = heapq.heappop(results)
            write_message_to_log(str(len(node._indices)), LOG_MODE_DEBUG)#node._indices[0]
            write_message_to_log(str(len(self.data)), LOG_MODE_DEBUG)
            return value, self.data[node._indices[0]]
        else:
            print("Error: failed to find a result")
            return np.inf, self.data[self._indices[0]]

    def find_best_example_excluding_search_candidates2(self, obj, args, n_candidates=1):#2
        """ Traverses the cluster hierarchy iteratively by evaluating the means
            of the clusters at each level based on the objective function.
            At the last level the method find_best_example for a KDTree is used.
            Multiple candidates are kept at each level in order to find the global
            optimum.
        """
        print("search with ", n_candidates, "candidates in tree with ", self._n_subdivisions, " subdivisions ")
        results = list()
        candidates = list()
        candidates.append((np.inf, self))
        level = 0
        while len(candidates) > 0:
            # print "search",level,n_candidates,len(candidates)
            new_candidates = list()
            for value, node in candidates:
                if len(node._children) > 0:
                    for child_index, child in enumerate(node._children):
                        cluster_value = obj(child._mean, args + [self.args])  # data
                        # good_candidates = node._find_best_cluster_candidates(obj, args, n_candidates)
                        heapq.heappush(new_candidates, (cluster_value, child))
                else:
                    heapq.heappush(results, (value, node))
            candidates = new_candidates[:n_candidates]
            level += 1
        print("depth", level)
        if len(results) > 0:
            value, node = heapq.heappop(results)
            print(len(node._indices))  # node._indices[0]
            print(len(self.data))
            return value, self.data[node._indices[0]]
        else:
            print("Error: failed to find a result")
            return np.inf, self.data[self._indices[0]]

    def find_best_example_excluding_search_candidates3(self, obj, args,
                                                      n_candidates=1,
                                                      filter_level=1):  # _2_heappushs_per_canididate
        """ Traverses the cluster hierarchy iteratively by evaluating the means
            of the clusters at each level based on the objective function.
            At the last level the method find_best_example for a KDTree is used.
            Multiple candidates are kept at each level in order to find the global
            optimum.
        """
        print("search with", n_candidates, "candidates in tree with ", self._n_subdivisions, " subdivisions ")
        results = list()
        candidates = list()
        candidates.append((np.inf, self))
        level = 0
        while len(candidates) > 0:
            print("search", level, n_candidates, len(candidates))
            new_candidates = list()
            for value, node in candidates:
                if len(node._children) > 0:
                    good_candidates = node._find_best_cluster_candidates(obj, args, n_candidates)
                    for c in good_candidates:
                        heapq.heappush(new_candidates, c)
                else:
                    heapq.heappush(results, (value, node))
            candidates = new_candidates[:n_candidates]
            if len(new_candidates) > 0 and level >= filter_level:
                candidates = _filter_outliers(*list(zip(*candidates)))# #scipy.stats.mstats.mquantiles(wx.values(), prob)
            level += 1

        print("depth", level)
        if len(results) > 0:
            value, node = heapq.heappop(results)
            print(len(node._indices))  # node._indices[0]
            print(len(self.data))
            return value, self.data[node._indices[0]]
        else:
            print("Error: failed to find a result")
            return np.inf, self.data[self._indices[0]]

    def get_number_of_leafs(self):
        if len(self._children) == 0:
            return 1
        else:
            n_leafs = 0
            for c in self._children:
                n_leafs += c.get_number_of_leafs()
            return n_leafs

    def get_level(self, depth=0):
        if len(self._children) == 0:
            return depth+1
        depths = [-1]
        for c in self._children:
            depths.append(c.get_level())
        return max(depths)

    def get_statistics(self):
        leafs = self.get_number_of_leafs()
        level = self.get_level(0)
        return leafs, level


    @staticmethod
    def load_from_json_file(file_name):
        with open(file_name, 'rt') as infile:
            tree_data = json.load(infile)
            return FeatureClusterTree.load_from_json(tree_data)

    @staticmethod
    def load_from_json(tree_data):
        tree = FeatureClusterTree(None, None, None, None, None)
        data = np.array(tree_data["data"])
        features = np.array(tree_data["features"])
        options = tree_data["options"]
        tree.build_node_from_json(data, features, options, tree_data["root"])
        return tree

    def save_to_json_file(self, file_name):
        with open(file_name, 'wt') as oufile:
            tree_data = dict()
            tree_data["data"] = self.data.tolist()
            tree_data["features"] = self._features.tolist()
            tree_data["options"] = self._options
            tree_data["root"] = self.node_to_json()
            json.dump(tree_data, oufile)

    def node_to_json(self):
        node_data = dict()
        #node_data["args"] = None#self.args
        node_data["mean"] = self._mean.tolist()
        node_data["indices"] = self._indices
        node_data["children"] = []
        for c in self._children:
            c_data = c.node_to_json()
            node_data["children"].append(c_data)
        return node_data

    def build_node_from_json(self, data, features, options, node_data):
        self.data = data
        self.args = None
        self._features = features
        self._options = options
        self._indices = node_data["indices"]
        self._n_subdivisions = options["n_subdivisions"]
        self._mean = np.array(node_data["mean"])
        self._children = []
        for c_data in node_data["children"]:
            c_tree = FeatureClusterTree(None, None, None, None, None)
            c_tree.build_node_from_json(data, features, options, c_data)
            self._children.append(c_tree)

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

    def print_n_samples(self, level=0):
        if self._indices is not None:
            print("level", level, len(self._indices))
        for child in self._children:
            child.print_n_samples(level)



