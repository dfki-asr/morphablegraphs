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
Created on Thu Jul 30 10:51:03 2015

@author: Erik Herrmann
"""
import json
import os
import time
import heapq
import numpy as np
import pickle as pickle
from anim_utils.animation_data.bvh import BVHReader
from anim_utils.animation_data.skeleton_builder import SkeletonBuilder
from anim_utils.animation_data.quaternion_frame import euler_to_quaternion
from ..space_partitioning.cluster_tree import ClusterTree
from ..space_partitioning.feature_cluster_tree import FeatureClusterTree
from ..space_partitioning.clustering import CLUSTERING_METHOD_KMEANS
from ..space_partitioning.features import map_motions_to_euclidean_pca, END_EFFECTORS2
from ..motion_model.motion_primitive_wrapper import MotionPrimitiveModelWrapper

try:
    from mgrd import Skeleton as MGRDSkeleton
    from mgrd import SkeletonNode as MGRDSkeletonNode
    has_mgrd = True
except ImportError:
    has_mgrd = False
    pass

MOTION_PRIMITIVE_FILE_ENDING = "mm.json"
MOTION_PRIMITIVE_FILE_ENDING2 = "mm_with_semantic.json"
CLUSTER_TREE_FILE_ENDING = "_cluster_tree"

TREE_TYPE_CLUSTER_TREE = 0
TREE_TYPE_FEATURE_CLUSTER_TREE = 1

FEATURE_TYPE_S_VECTOR = 0
FEATURE_TYPE_EUCLIDEAN_PCA = 1


class MGRDSkeletonBVHLoader(object):
    """ Load a Skeleton from a BVH file.

    Attributes:
        file (string): path to the bvh file
    """

    def __init__(self, file):
        self.file = file
        self.bvh = None

    def load(self):
        self.bvh = BVHReader(self.file)
        root = self.create_root()
        self.populate(root)
        return MGRDSkeleton(root)

    def create_root(self):
        return self.create_node(self.bvh.root, None)

    def create_node(self, name, parent):
        node_data = self.bvh.node_names[name]
        offset = node_data["offset"]
        if "channels" in node_data:
            angle_channels = ["Xrotation", "Yrotation", "Zrotation"]
            angles_for_all_frames = self.bvh.get_angles([(name, ch) for ch in angle_channels])
            orientation = euler_to_quaternion(angles_for_all_frames[0])
        else:
            orientation = euler_to_quaternion([0, 0, 0])
        return MGRDSkeletonNode(name, parent, offset, orientation)

    def populate(self, node):
        node_data = self.bvh.node_names[node.name]
        if "children" not in node_data:
            return
        for child in node_data["children"]:
            child_node = self.create_node(child, node)
            node.add_child(child_node)
            self.populate(child_node)


def save_data_to_pickle(data, output_filename):
    with open(output_filename, "wb") as output_file:
        pickle.dump(data, output_file, pickle.HIGHEST_PROTOCOL)


def load_data_from_pickle(input_filename):
    print("load", input_filename)
    with open(input_filename, "rb") as in_file:
        return pickle.load(in_file)


class ClusterTreeBuilder(object):
    """ Creates ClusterTrees for all motion primitives by sampling from the statistical model
        The motion primitives are assumed to be organized in a directory
        hierarchy as follows
        - model_data_root_dir
            - elementary_action_dir
                - motion_primitive_mm.json
    """
    def __init__(self, settings):
        self.morphable_model_directory = None
        self.n_samples = 10000
        self.n_subdivisions_per_level = 4
        self.n_levels = 4
        self.random_seed = None
        self.only_spatial_parameters = True
        self.mgrd_skeleton = None
        self.skeleton = None
        self.use_kd_tree = True
        self.tree_type = settings["tree_type"]
        self.feature_type = settings["feature_type"]
        self.output_mode = settings["output_mode"]
        self.animated_joints = None

    def set_config_from_file(self, config_file_path):
        config_file = open(config_file_path)
        config = json.load(config_file)
        self.set_config(config)

    def set_config(self, config):
        self.morphable_model_directory = config["model_data_dir"]
        self.n_samples = config["n_random_samples"]
        self.n_subdivisions_per_level = config["n_subdivisions_per_level"]
        self.n_levels = config["n_levels"]
        self.random_seed = config["random_seed"]
        self.only_spatial_parameters = config["only_spatial_parameters"]
        self.store_indices = config["store_data_indices_in_nodes"]
        self.use_kd_tree = config["use_kd_tree"]

    def load_skeleton(self, skeleton_path):
        bvh = BVHReader(skeleton_path)
        self.animated_joints = list(bvh.get_animated_joints())
        if has_mgrd:
            self.mgrd_skeleton = MGRDSkeletonBVHLoader(skeleton_path).load()
        self.skeleton = SkeletonBuilder().load_from_bvh(BVHReader(skeleton_path), self.animated_joints)

    def _get_samples_using_threshold(self, motion_primitive, threshold=0, max_iter_count=5):
        data = []
        count = 0
        iter_count = 0

        while count < self.n_samples and iter_count < max_iter_count:
            samples = motion_primitive.sample_low_dimensional_vectors(self.n_samples)
            for new_sample in samples:
                likelihood = motion_primitive.get_gaussian_mixture_model().score([new_sample,])[0]
                if likelihood > threshold:
                    data.append(new_sample)
                    count += 1
            iter_count += 1
        if iter_count < max_iter_count:
            return np.asarray(data)
        else:
            return None

    def _get_best_samples(self, motion_primitive):
        likelihoods = []
        data = motion_primitive.sample_low_dimensional_vectors(self.n_samples*2)
        for idx, sample in enumerate(data):
            likelihood = motion_primitive.get_gaussian_mixture_model().score([sample,])[0]
            #print i, likelihood, new_sample
            heapq.heappush(likelihoods, (-likelihood, idx))

        l, indices = list(zip(*likelihoods[:self.n_samples]))
        data = np.asarray(data)
        data = data[list(indices)]
        np.random.shuffle(data)
        return data

    def sample_data(self, motion_primitive):
        return motion_primitive.sample_low_dimensional_vectors(self.n_samples)

    def _create_training_data(self, data_dir, cluster_file_name, mp, n_training_samples):
        print("Create", self.n_samples, "good samples")
        data = None
        n_random_samples = n_training_samples
        file_name = data_dir + os.sep + cluster_file_name + "_" + "training_data" + str(n_random_samples) + ".pck"#
        if os.path.isfile(file_name):
            data = load_data_from_pickle(file_name)
        if data is None:
            print("load file")
            data = self.sample_data(mp)
            save_data_to_pickle(data, file_name)
        else:
            print("loaded file", file_name)
        return data

    def _create_space_partitioning(self, elementary_action_dir, file_name, action_name):

        index = file_name.find("_mm")
        cluster_file_name = file_name[:index]

        cluster_tree_file_name = elementary_action_dir + os.sep + cluster_file_name + CLUSTER_TREE_FILE_ENDING + ".pck"

        motion_primitive_file_name = elementary_action_dir + os.sep + file_name

        if os.path.isfile(cluster_tree_file_name) and False:
            print("Space partitioning data structure", cluster_file_name, "already exists")

        elif os.path.isfile(motion_primitive_file_name):
            print("construct space partitioning data structure", cluster_file_name, self.animated_joints)
            motion_primitive = MotionPrimitiveModelWrapper()
            motion_primitive._load_from_file(self.mgrd_skeleton, motion_primitive_file_name, self.animated_joints)

            data = self._create_training_data(elementary_action_dir, cluster_file_name, motion_primitive, self.n_samples)
            if self.tree_type == TREE_TYPE_FEATURE_CLUSTER_TREE:
                self._build_feature_tree(action_name, cluster_file_name, data, motion_primitive)
            else:
                self._build_tree(elementary_action_dir, cluster_file_name, data, motion_primitive)

        else:
            print("Could not read motion primitive", motion_primitive_file_name)

    def _build_tree(self, elementary_action_dir, cluster_file_name, data, motion_primitive):
        if self.only_spatial_parameters:
            n_dims = motion_primitive.get_n_spatial_components()
            print("maximum dimension set to", n_dims, "ignoring time parameters")
        else:
            n_dims = len(data[0])
        cluster_tree = ClusterTree(self.n_subdivisions_per_level, self.n_levels, n_dims, self.store_indices,
                                   self.use_kd_tree)
        cluster_tree.construct(data)
        cluster_tree_file_name = elementary_action_dir + os.sep + cluster_file_name + CLUSTER_TREE_FILE_ENDING +".pck"
        cluster_tree.save_to_file_pickle(cluster_tree_file_name)
        n_leafs = cluster_tree.root.get_number_of_leafs()
        print("number of leafs", n_leafs)

    def _build_feature_tree(self, action_name, model_name, data, motion_primitive):
        name = os.path.basename(model_name)[:-len("_quaternion")]
        features = self._extract_features(motion_primitive, action_name, name, data)
        options = {"n_subdivisions": self.n_subdivisions_per_level,
                   "clustering_method": CLUSTERING_METHOD_KMEANS,
                   "use_feature_mean": False}
        cluster_tree = FeatureClusterTree(features, data, None, options, [])
        n_leafs = cluster_tree.get_number_of_leafs()
        print("number of leafs", n_leafs)
        cluster_tree_file = self.morphable_model_directory + os.sep + action_name + os.sep +model_name + CLUSTER_TREE_FILE_ENDING
        if self.output_mode == "pck":
            cluster_tree.save_to_file_pickle(cluster_tree_file+".pck")
        else:
            cluster_tree.save_to_json_file(cluster_tree_file+".json")

        print("save cluster tree", cluster_tree_file)

    def _extract_features(self, motion_primitive, action_name,  mp_name, data):
        if self.feature_type == FEATURE_TYPE_EUCLIDEAN_PCA:
            filename = self.morphable_model_directory + os.sep + action_name + os.sep + mp_name + "_euclidean_pca_" + str(
                self.n_samples)
            filename += ".pck"

            print("try to load features from file", filename)
            features = None
            if os.path.isfile(filename):
                features = load_data_from_pickle(filename)
                if features is not None:
                    print("loaded features from file", filename)
                else:
                    print("file could not be loaded", filename)
            if features is None:
                motions = self._back_project(motion_primitive, data)
                features, feature_args = map_motions_to_euclidean_pca(motions, self.skeleton, END_EFFECTORS2, step=1)

                print("save training data", filename)
                save_data_to_pickle(features, filename)
        else:
            n_spatial =motion_primitive.get_n_spatial_components()
            print(n_spatial, data.shape)
            features = data[:, :n_spatial]

        return features

    def _back_project(self, mp, data):
        print("backproject data")
        splines = [mp.back_project(s) for s in data]
        motions = []
        canonical_frames = list(range(0, int(mp.get_n_canonical_frames())))
        for spline in splines:
            sample = [spline.evaluate(frame_idx) for frame_idx in canonical_frames]
            motions.append(sample)
        return np.array(motions)

    def _process_elementary_action(self, elementary_action):
        elementary_action_dir = self.morphable_model_directory + os.sep + elementary_action
        for root, dirs, files in os.walk(elementary_action_dir):
            for file_name in files:
                if file_name.endswith(MOTION_PRIMITIVE_FILE_ENDING) or file_name.endswith(MOTION_PRIMITIVE_FILE_ENDING2):
                    print(elementary_action_dir + os.sep + file_name)
                    self._create_space_partitioning(elementary_action_dir, file_name,  elementary_action)
        return True

    def build(self):
        if self.random_seed is not None:
            print("apply random seed", self.random_seed)
            np.random.seed(self.random_seed)
        if self.morphable_model_directory is not None:
            print("start construction in directory", self.morphable_model_directory)
            for elementary_action in next(os.walk(self.morphable_model_directory))[1]:
                self._process_elementary_action(elementary_action)
            return True
        else:
            return False

    def build_action(self, action_name):
        elementary_action_dir = self.morphable_model_directory
        for root, dirs, files in os.walk(elementary_action_dir):
            for file_name in files:
                if file_name.endswith(MOTION_PRIMITIVE_FILE_ENDING) or file_name.endswith(
                        MOTION_PRIMITIVE_FILE_ENDING2):
                    print(elementary_action_dir + os.sep + file_name)
                    self._create_space_partitioning(elementary_action_dir, file_name, action_name)
        return True

    def build_for_one_motion_primitive(self, motion_primitive_file, space_partition_file):
        self._create_space_partitioning(motion_primitive_file, space_partition_file)

    def build_from_path(self, elementary_action_path):
        self._process_elementary_action(elementary_action_path)
        return
