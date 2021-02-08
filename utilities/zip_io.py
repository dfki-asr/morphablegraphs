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
Created on Tue Jun 23 18:34:12 2015

@author: Erik Herrmann
"""
import zipfile
import json
import pickle
import time
import numpy as np
from anim_utils.utilities.log import write_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR


MORPHABLE_MODEL_FILE_ENDING = "mm.json"
MM_TYPE = "quaternion"
ELEMENTARY_ACTION_DIRECTORY = "elementary_action_models"
TRANSITION_MODEL_DIRECTORY = "transition_models"
GRAPH_DEFINITION_FILE = "graph_definition.json"
SKELETON_JSON_FILE = "skeleton.json"
SKELETON_BVH_FILE = "skeleton.bvh"
SKELETON_BVH_STRING_KEY = "bvh_skeleton_string"
SKELETON_JSON_KEY = "skeleton"
PFNN_DIRECTORY = "pfnn"
PFNN_WEIGHT_KEYS = ["W0", "W1", "W2", "b0", "b1", "b2"]
PFNN_MEAN_KEYS = ["Xmean", "Ymean", "Xstd", "Ystd"]


class ZipReader(object):
    def __init__(self, zip_file_path, pickle_objects=True, verbose=True):
        self.zip_file_path = zip_file_path
        self.zip_file = None
        self.pickle_objects = pickle_objects
        self.type_offset = len(MM_TYPE) + 1
        self.graph_data = None
        self.verbose = verbose
        self.elementary_action_directory = ELEMENTARY_ACTION_DIRECTORY
        self.transition_model_directory = TRANSITION_MODEL_DIRECTORY
        self.format_version = 1.0

    def get_graph_data(self):
        """ Extracts the data from the files stored in the zip file and
            returns it in a dictionary for easier parsing. The space partitioning
            data structure is also deserialized into an object.
            If pickle_objects is False the space partitioning is ignored.
        """
        write_message_to_log("Loading model data from file " + self.zip_file_path + " ...", LOG_MODE_INFO)
        self.zip_file = zipfile.ZipFile(self.zip_file_path, "r", zipfile.ZIP_DEFLATED)
        graph_data_str = self.zip_file.read(GRAPH_DEFINITION_FILE).decode("utf-8")
        data = json.loads(graph_data_str)
        if "formatVersion" in data.keys():
            self.format_version = float(data["formatVersion"])
        else:
            self.format_version = 1.0

        self.use_pickle = False
        if "usePickle" in data:
            self.use_pickle = data["usePickle"]

        write_message_to_log("Format version " + str(self.format_version), LOG_MODE_DEBUG)
        if self.format_version >= 2.0:
            structure_desc = self._read_elementary_action_file_structure_from_zip_v2()
        else:
            structure_desc = self._read_elementary_action_file_structure_from_zip_v1()
        if self.format_version <= 2.0:
            data[SKELETON_BVH_STRING_KEY] = self.zip_file.read(SKELETON_BVH_FILE).decode("utf-8")
        elif self.format_version >= 4.0:
            skeleton_data_str = self.zip_file.read(SKELETON_JSON_FILE).decode("utf-8")
            data[SKELETON_JSON_KEY] = json.loads(skeleton_data_str)
        self._construct_graph_data(structure_desc)
        data["subgraphs"] = self.graph_data
        return data

    def get_pfnn_data(self):
        data = None
        self.zip_file = zipfile.ZipFile(self.zip_file_path, "r", zipfile.ZIP_DEFLATED)
        if PFNN_DIRECTORY+"/" in self.zip_file.namelist():
            data = dict()
            data["weights"] = self.read_pfnn_weights()
            data["means"] = self.read_pfnn_means()
        return data

    def read_pfnn_weights(self):
        weights = dict()
        for key in PFNN_WEIGHT_KEYS:
            weights[key] = np.frombuffer(self.zip_file.read("pfnn/weights/"+key+".bin"),np.float32)
        return weights

    def read_pfnn_means(self):
        means = dict()
        for key in PFNN_MEAN_KEYS:
            means[key] = np.frombuffer(self.zip_file.read("pfnn/means/" + key + ".bin"), np.float32)
        return means

    def _read_elementary_action_file_structure_from_zip_v1(self):
        elementary_actions = dict()
        for name in self.zip_file.namelist():
            splitted_name = name.split("/")
            if len(splitted_name) > 1:
                action_directory = splitted_name[0]
                file_name = splitted_name[1]
                if file_name.endswith(MORPHABLE_MODEL_FILE_ENDING):
                    if action_directory not in list(elementary_actions.keys()):
                        elementary_actions[action_directory] = []
                    elementary_actions[action_directory].append(file_name[:-8])
        structure_desc = dict()
        structure_desc[self.elementary_action_directory] = elementary_actions
        structure_desc[self.transition_model_directory] = dict()
        return structure_desc

    def _read_elementary_action_file_structure_from_zip_v2(self):
        elementary_actions = dict()
        for name in self.zip_file.namelist():
            splitted_name = name.split("/")
            if len(splitted_name) > 2:
                mm_directory = splitted_name[0]
                if mm_directory == self.elementary_action_directory:
                    action_directory = splitted_name[1]
                    file_name = splitted_name[2]
                    if file_name.endswith(MORPHABLE_MODEL_FILE_ENDING):
                        if action_directory not in list(elementary_actions.keys()):
                            elementary_actions[action_directory] = []
                        elementary_actions[action_directory].append(file_name[:-8])
        structure_desc = dict()
        structure_desc[self.elementary_action_directory] = elementary_actions
        structure_desc[self.transition_model_directory] = dict()

        return structure_desc

    def _read_hand_pose_data(self):
        hand_pose_info = json.loads(self.zip_file.read("hand_poses/hand_pose_info.json").decode('utf-8'))
        hand_pose_info["skeletonStrings"] = dict()
        try:
            for file_path in self.zip_file.namelist():
                splitted_name = file_path.split("/")
                if len(splitted_name) > 1:
                    filename = splitted_name[1][:-4]
                    if splitted_name[0] == "hand_poses" and splitted_name[1][-4:] == ".bvh":
                        hand_pose_info["skeletonStrings"][filename] = self.zip_file.read(file_path)
        except:
            write_message_to_log("Error: Did not find example skeletons for hand pose in zip file ", LOG_MODE_ERROR)
            pass
        return hand_pose_info

    def _construct_graph_data(self, structure_desc):
        self.graph_data = dict()
        for structure_key in list(structure_desc[self.elementary_action_directory].keys()):
            action_data_key = structure_key.split("_")[2]
            if self.verbose:
                write_message_to_log("Load action " +str(action_data_key), LOG_MODE_INFO)
            self.graph_data[action_data_key] = {}
            self.graph_data[action_data_key]["name"] = action_data_key
            meta_info_file = self._get_meta_info_file_path(structure_key)
            if meta_info_file in self.zip_file.namelist():
                self.graph_data[action_data_key]["info"] = json.loads(self.zip_file.read(meta_info_file).decode("utf-8"))
            self.graph_data[action_data_key]["nodes"] = {}
            for mp in structure_desc[self.elementary_action_directory][structure_key]:
                self._add_motion_primitive(action_data_key, structure_key, mp)

    def _add_motion_primitive(self, action_data_key, structure_key, motion_primitive_name):
        #extract mp key name from file name
        splitted_name = (motion_primitive_name[:-self.type_offset]).split("_")
        action_offset = len(splitted_name[0])+1
        mp_data_key = motion_primitive_name[action_offset:-self.type_offset]
        self.graph_data[action_data_key]["nodes"][mp_data_key] = {}
        self.graph_data[action_data_key]["nodes"][mp_data_key]["name"] = motion_primitive_name[:-self.type_offset]
        mm_string = self.zip_file.read(self._get_motion_primitive_file_path(structure_key, motion_primitive_name)).decode("utf-8")
        mm_data = json.loads(mm_string)
        self.graph_data[action_data_key]["nodes"][mp_data_key]["mm"] = mm_data
        if self.verbose:
            write_message_to_log("\t"+ "Load motion primitive " + motion_primitive_name, LOG_MODE_INFO)
        statsfile = structure_key + "/" + (motion_primitive_name[:-self.type_offset] + ".stats")
        self._add_stats(action_data_key, mp_data_key, statsfile)
        space_partition_file = self._get_space_partitioning_file_path(structure_key, motion_primitive_name)
        #print space_partition_file
        self._add_space_partitioning_data_structure(action_data_key, mp_data_key, space_partition_file)

    def _add_stats(self, action_data_key, mp_data_key, statsfile):
        if statsfile in self.zip_file.namelist():
            stats_string = self.zip_file.read(statsfile)
            stats_data = json.loads(stats_string)
            self.graph_data[action_data_key]["nodes"][mp_data_key]["stats"] = stats_data

    def _add_space_partitioning_data_structure(self, action_data_key, mp_data_key, space_partition_file):
        if space_partition_file in self.zip_file.namelist():
            data = self.zip_file.read(space_partition_file)
            self.graph_data[action_data_key]["nodes"][mp_data_key]["space_partition"] = None
            if self.pickle_objects and self.format_version < 4.0 or self.use_pickle:
                self.graph_data[action_data_key]["nodes"][mp_data_key]["space_partition_pickle"] = pickle.loads(data)
            elif self.format_version >= 4.0:
                self.graph_data[action_data_key]["nodes"][mp_data_key]["space_partition_json"] = json.loads(data.decode("utf-8"))

    def _get_motion_primitive_file_path(self, structure_key, motion_primitive_name):
        if self.format_version >= 2.0:
            return self.elementary_action_directory + "/" + structure_key + "/" + motion_primitive_name + "_mm.json"
        else:
            return structure_key + "/" + motion_primitive_name + "_mm.json"

    def _get_space_partitioning_file_path(self, structure_key, motion_primitive_name):
        if self.format_version >= 4.0 and not self.use_pickle:
            return self.elementary_action_directory + "/" + structure_key + "/" + motion_primitive_name + "_cluster_tree.json"
        elif self.format_version >= 2.0 or self.use_pickle:
            return self.elementary_action_directory + "/" + structure_key + "/" + motion_primitive_name + "_cluster_tree.pck"
        else:
            return structure_key + "/" + motion_primitive_name + "_cluster_tree.pck"

    def _get_meta_info_file_path(self, structure_key):
        if self.format_version >= 2.0:
            return self.elementary_action_directory + "/" + structure_key + "/meta_information.json"
        else:
            return structure_key + "/meta_information.json"

def main():
    zip_path = "E:\\projects\\INTERACT\\repository\\data\\3 - Motion primitives\\motion_primitives_quaternion_PCA95.zip"
    print(zip_path)
    start = time.time()
    zip_loader = ZipReader(zip_path)
    graph_data = zip_loader.get_graph_data()
    print(list(graph_data["subgraphs"]["pick"]["nodes"].keys()))
    print("finished reading data in", time.time() - start, "seconds")
    # print  graph_data

if __name__ == "__main__":
    main()
