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
Created on Wed Jul 29 16:07:11 2015

@author: Erik Herrmann
"""
import numpy as np
from sklearn import cluster
import uuid
from . import KDTREE_WRAPPER_NODE, LEAF_NODE, INNER_NODE, ROOT_NODE
from .cluster_tree_node import ClusterTreeNode
from .kdtree_wrapper_node import KDTreeWrapper


class ClusterTreeNodeBuilder(object):
    """ Creates a ClusterTreeNode based on samples. It subdivides samples using KMeans and
        creates a child node for each subdivision. Child nodes can be ClusterTreeNodes
        or KDTreeNodes depending on the current depth and the maximum depth.
        Stores the indices refering to the samples stored in ClusterTree.

    Parameters
    ---------
    * n_subdivisions: Integer
        Number of subdivisions.
    * max_level: Integer
        Maximum number of levels before.
    * dim: Integer
        Number of dimensions of the data.
    * use_kd_tree: Bool
        K-D Trees are created for levels deeper than max level
    """
    def __init__(self, n_subdivisions, max_level, dim, store_indices,use_kd_tree=True):
 
        self.n_subdivisions = n_subdivisions
        self.max_level = max_level
        self.dim = dim
        self.kmeans = None#cluster.KMeans(n_clusters=self.n_subdivisions)
        self.store_indices = store_indices
        self.use_kd_tree = use_kd_tree

    def _calculate_mean(self, data, indices):
        if indices is None:
            n_samples = len(data)
            mean = np.mean(data, axis=0)
        else:
            n_samples = len(indices)
            mean = np.mean(data[indices], axis=0)
        return mean, n_samples
        
    def _get_node_type_from_depth(self, depth):
        """Decide on on the node type based on the maximum number of levels."""
        if depth < self.max_level - 1:
            if depth == 0:
                node_type = ROOT_NODE
            else:
                node_type = INNER_NODE
        else:
            node_type = LEAF_NODE
        return node_type
        
    def _detect_clusters(self, data, indices, n_samples):
        """Use the kmeans algorithm of scipy to labels to samples according 
        to clusters.
        """
        self.kmeans = cluster.KMeans(n_clusters=self.n_subdivisions)
        if indices is None:
            labels = self.kmeans.fit_predict(data[:, :self.dim])
        else:
            labels = self.kmeans.fit_predict(data[indices, :self.dim])
        cluster_indices = [[] for i in range(self.n_subdivisions)]
        if indices is None:
            for i in range(n_samples):
                l = labels[i]
                cluster_indices[l].append(i)
        else:
            for i in range(n_samples):
                l = labels[i]
                original_index = indices[i]
                cluster_indices[l].append(original_index)
                
        return cluster_indices
        
    def construct_from_data(self, data, indices=None, depth=0):
        """ Creates a divides sample space into partitions using KMeans and creates
             a child for each space partition.
             
        Parameters
        ----------
        * data: np.ndarray
            2D array of samples
        * indices: list
            indices of data that should be considered for the subdivision
        * depth: int
            current depth used with self.max_level to decide the type of node and the type of subdivisions
        """
        clusters = []
        node_type = self._get_node_type_from_depth(depth)
        if indices is not None:
            n_samples = len(indices)
        else:
            n_samples = len(data)
        if not self.use_kd_tree and n_samples== 1:
            mean = data[indices[0]]
            is_leaf = True
        else:
            if n_samples > self.n_subdivisions and self.n_subdivisions > 1:
                is_leaf = False
                mean, clusters = self._create_subdivision(data, indices, depth)
            else:
                is_leaf = self.use_kd_tree
                mean, clusters = self._create_leafs(data,indices, depth)
        if self.store_indices:
            return ClusterTreeNode(uuid.uuid1(), depth, indices, mean, clusters, node_type, is_leaf)
        else:
            return ClusterTreeNode(uuid.uuid1(), depth, None, mean, clusters, node_type, is_leaf)

    def _create_subdivision(self,data,indices, depth):
        clusters = []
        mean, n_samples = self._calculate_mean(data, indices)
        cluster_indices = self._detect_clusters(data, indices, n_samples)
        if depth < self.max_level or not self.use_kd_tree:
            for j in range(len(cluster_indices)):
                if len(cluster_indices[j]) > 0:
                    child_node = self.construct_from_data(data, cluster_indices[j], depth + 1)
                    clusters.append(child_node)
        else:
            for j in range(len(cluster_indices)):
                if len(cluster_indices[j]) > 0:
                    child_node = KDTreeWrapper(self.dim)
                    child_node.construct(data, cluster_indices[j])
                    clusters.append(child_node)
        return mean, clusters

    def _create_leafs(self,data,indices, depth):
        mean, n_samples = self._calculate_mean(data, indices)
        clusters = []
        if self.use_kd_tree:
            #print "create kd tree b"
            child_node = KDTreeWrapper(self.dim)
            child_node.construct(data, indices)
            clusters.append(child_node)
        else:
            #print "create nodes for leafs"
            for idx in indices:
                child_node = self.construct_from_data(data, [idx], depth + 1)
                clusters.append(child_node)
        return mean, clusters


    def construct_from_node_desc_list(self,node_id, node_desc, data):
        """Recursively rebuilds the cluster tree given a dictionary containing 
           a description of all nodes and the data samples.
        
        Parameters
        ---------
        * node_id: String
            Unique identifier of the node.
        * node_desc: dict
            Dictionary containing the properties of each node. The node id is used as key.
        * data: np.ndarray
            Data samples.
        """
        desc = node_desc["nodes"][node_id]
        clusters = []
        node_type = desc["type"]
        mean = np.array(desc["mean"])
        depth = desc["depth"]
        if desc["type"] != ROOT_NODE:
            indices = desc["indices"]
        else:
            indices = []
            
        if desc["type"] != LEAF_NODE:
            is_leaf = False
            for c_id in desc["children"]:
                child_node = self.construct_from_node_desc_list(c_id, node_desc, data)
                clusters.append(child_node)
        else:
            is_leaf = True
            for c_id in desc["children"]:
                child_node = KDTreeWrapper(self.dim)
                child_node.id = c_id
                indices = node_desc["nodes"][c_id]["indices"]
                child_node.construct(data, indices)
                clusters.append(child_node)
            
        return ClusterTreeNode(node_id, depth, indices, mean, clusters, node_type, is_leaf)
