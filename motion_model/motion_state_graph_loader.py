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
Created on Thu Jul 16 15:57:51 2015

@author: Erik Herrmann
"""

import os
import json
from anim_utils.animation_data.bvh import BVHReader
from anim_utils.animation_data.skeleton_builder import SkeletonBuilder
from anim_utils.animation_data.quaternion_frame import euler_to_quaternion
from anim_utils.utilities.io_helper_functions import load_json_file
from anim_utils.utilities.log import write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_ERROR, LOG_MODE_INFO
from ..utilities.zip_io import ZipReader, SKELETON_BVH_STRING_KEY, SKELETON_JSON_KEY
from .motion_state_transition import MotionStateTransition
from .motion_state_graph import MotionStateGraph
from .motion_state_group import MotionStateGroup
from .motion_state_graph_node import MotionStateGraphNode
from ..motion_generator.hand_pose_generator import HandPoseGenerator
from . import TRANSITION_MODEL_DIRECTORY_NAME, NODE_TYPE_START, NODE_TYPE_STANDARD,NODE_TYPE_CYCLE_END, NODE_TYPE_END, TRANSITION_DEFINITION_FILE_NAME, TRANSITION_MODEL_FILE_ENDING, NODE_TYPE_IDLE
from ..utilities.db_interface import download_graph_from_remote_db, get_skeleton_from_remote_db, get_skeleton_model_from_remote_db, download_motion_model_from_remote_db, download_cluster_tree_from_remote_db
from ..utilities import convert_to_mgrd_skeleton


class MotionStateGraphLoader(object):
    """   Constructs a MotionPrimitiveGraph instance from a zip file or directory as data source
    """  
    def __init__(self):
        self.graph_data = None  # used to store the zip file content
        self.load_transition_models = False
        self.update_stats = False
        self.motion_state_graph_path = None
        self.ea_directory = None
        self.use_all_joints = False
        self.pfnn_data = None

    def set_data_source(self, motion_state_graph_path, load_transition_models=False, update_stats=False):
        """ Set the source which is used to load the data structure into memory.
        Parameters
        ----------
        * elementary_action_directory: string
        \tThe root directory of the morphable models of all elementary actions.
        * transition_model_directory: string
        \tThe directory of the morphable models of an elementary action.
        * transition_model_directory: string
        \tThe directory of the transition models.
        """

        self.load_transition_models = load_transition_models
        self.update_stats = update_stats
        self.motion_state_graph_path = motion_state_graph_path

    def build(self):
        graph = MotionStateGraph()
        self._build_from_zip_file(graph)
        return graph

    def build_from_database(self, db_url, skeleton_name, graph_id, frame_time=None):
        ms_graph = MotionStateGraph()
        graph_data = download_graph_from_remote_db(db_url, graph_id)
        if type(graph_data) == str:
            graph_data = json.loads(graph_data)
        skeleton_data = get_skeleton_from_remote_db(db_url, skeleton_name)
        ms_graph.skeleton = SkeletonBuilder().load_from_custom_unity_format(skeleton_data)
        if frame_time is not None:
            ms_graph.skeleton.frame_time = frame_time
        ms_graph.skeleton.skeleton_model = get_skeleton_model_from_remote_db(db_url, skeleton_name)
        ms_graph.mgrd_skeleton = convert_to_mgrd_skeleton(ms_graph.skeleton)

        ms_graph.action_definitions = dict()

        transitions = dict()
        for a in graph_data["nodes"]:

            meta_info = dict()
            start_states = []
            end_states = []
            idle_states = []
            single_states = []

            action_def = dict()
            action_def["name"] = a
            action_def["nodes"] = dict()
            action_def["constraint_slots"] = dict()
            motion_primitives = graph_data["nodes"][a]
            for model_id in motion_primitives:
                mp_name = motion_primitives[model_id]["name"]
                if mp_name.startswith("walk"):
                    mp_name = mp_name[5:]
                print("load", mp_name)
                motion_state_def = dict()
                motion_state_def["name"] = mp_name
                mp_type = motion_primitives[model_id]["type"]
                if mp_type == "start":
                    start_states.append(mp_name)
                elif mp_type == "end":
                    end_states.append(mp_name)
                elif mp_type == "idle":
                    idle_states.append(mp_name)
                elif mp_type == "single":
                    single_states.append(mp_name)

                mp_transitions = list(motion_primitives[model_id]["transitions"].keys())
                mp_transitions = [key if not key[5:].startswith("walk") else key[:5]+key[10:] for key in mp_transitions]
                mp_transitions = [key.split(":") if  ":" in key else None for key in mp_transitions]
                transitions[(a, mp_name)] = mp_transitions
                model_data_str = download_motion_model_from_remote_db(db_url, model_id)
                if model_data_str is None:
                    print("Could not load model")
                    continue
                try:
                    motion_state_def["mm"] = json.loads(model_data_str)
                except:
                    print("Could not load model")
                    continue
                # store keyframes in action definition
                if "keyframes" in motion_state_def["mm"]:
                    for key in motion_state_def["mm"]["keyframes"]:
                        action_def["constraint_slots"][key] = {"node": mp_name, "joint": "left_wrist"}
                cluster_tree_data_str = download_cluster_tree_from_remote_db(db_url, model_id)
                if cluster_tree_data_str is not None and len(cluster_tree_data_str) > 0:
                    try:
                        motion_state_def["space_partition_json"] = json.loads(cluster_tree_data_str)
                    except:
                        print("Could not load tree")
                action_def["nodes"][mp_name] = motion_state_def
            
            meta_info["start_states"] = start_states
            meta_info["end_states"] = end_states
            meta_info["idle_states"] = idle_states
            meta_info["single_states"] = single_states
            action_def["info"] = meta_info


    
          
            node_group = self.build_node_group_from_dict(action_def, ms_graph)
            ms_graph.nodes.update(node_group.nodes)
            ms_graph.node_groups[node_group.ea_name] = node_group
            if a == "walk" and len(node_group.idle_states) > 0:
                idle_mp = node_group.idle_states[0]
                ms_graph.start_node = (a, idle_mp)

            # store action definition for constraint builder
        
            action_def["node_sequence"] = []
            if len(motion_primitives) == 1:
                mp_id = list(motion_primitives.keys())[0]
                mp_name = motion_primitives[mp_id]["name"]
                action_def["node_sequence"] = [[mp_name, "single_primitive"]]
            action_def["start_states"] = start_states
            action_def["end_states"] = end_states
            action_def["idle_states"] = idle_states
            ms_graph.action_definitions[a] = action_def

        for from_node_key in transitions:
            for to_node_key in transitions[from_node_key]:
                if to_node_key is not None:
                    self._add_transition(ms_graph, from_node_key, tuple(to_node_key))
        self._update_motion_state_stats(ms_graph, recalculate=True)

        if "start_node" in graph_data:
            start_node = graph_data["start_node"]
            if start_node[1].startswith("walk"):
               start_node[1] = start_node[1][5:]
            ms_graph.start_node = tuple(start_node)
        print("set start", ms_graph.start_node)
        return ms_graph


    def _build_from_zip_file(self, ms_graph):
        zip_path = self.motion_state_graph_path+".zip"
        zip_reader = ZipReader(zip_path, pickle_objects=True)
        graph_data = zip_reader.get_graph_data()
        self.pfnn_data = zip_reader.get_pfnn_data()
        if SKELETON_BVH_STRING_KEY in graph_data.keys():
            bvh_reader = BVHReader("").init_from_string(graph_data[SKELETON_BVH_STRING_KEY])
            ms_graph.skeleton = SkeletonBuilder().load_from_bvh(bvh_reader)
        elif SKELETON_JSON_KEY in graph_data.keys():
            #use_all_joints = False
            if self.use_all_joints and "animated_joints" in graph_data[SKELETON_JSON_KEY]:
                del graph_data[SKELETON_JSON_KEY]["animated_joints"]
            ms_graph.skeleton = SkeletonBuilder().load_from_json_data(graph_data[SKELETON_JSON_KEY], use_all_joints=self.use_all_joints)
            print("load skeleton", ms_graph.skeleton.animated_joints)
        else:
            raise Exception("There is no skeleton defined in the graph file")
            return

        ms_graph.animated_joints = ms_graph.skeleton.animated_joints
        ms_graph.mgrd_skeleton = convert_to_mgrd_skeleton(ms_graph.skeleton)

        transition_dict = graph_data["transitions"]
        actions = graph_data["subgraphs"]
        for action_name in actions.keys():
            node_group = self.build_node_group_from_dict(actions[action_name], ms_graph)
            ms_graph.nodes.update(node_group.nodes)
            ms_graph.node_groups[node_group.ea_name] = node_group
            if action_name == "walk" and len(node_group.idle_states) > 0:
                idle_mp = node_group.idle_states[0]
                ms_graph.start_node = (action_name, idle_mp)

        self._set_transitions_from_dict(ms_graph, transition_dict)

        self._update_motion_state_stats(ms_graph, recalculate=False)

        if "hand_pose_info" in graph_data:
            ms_graph.hand_pose_generator = HandPoseGenerator(ms_graph.skeleton)
            ms_graph.hand_pose_generator.init_from_desc(graph_data["hand_pose_info"])

        if "actionDefinitions" in graph_data:
            ms_graph.action_definitions = graph_data["actionDefinitions"]
        if "startNode" in graph_data:
            start_node = list(graph_data["startNode"])
            if start_node[1].startswith("walk"):
               start_node[1] = start_node[1][5:]
            ms_graph.start_node = tuple(start_node)

    def _update_motion_state_stats(self, motion_state_graph, recalculate=False):
        for keys in motion_state_graph.node_groups.keys():
            motion_state_graph.node_groups[keys]._update_motion_state_stats(recalculate=recalculate)

    def _set_transitions_from_dict(self, motion_state_graph, transition_dict):
        if len(transition_dict) == 0:
            return
        split_key= "_"
        if ":" in list(transition_dict.keys())[0]:
            split_key = ":"
        for node_key in transition_dict:
            from_action_name = node_key.split(split_key)[0]
            from_mp_key = node_key.split(split_key)[1]
            from_node_key = (from_action_name, from_mp_key)
            if from_node_key in motion_state_graph.nodes.keys():
                for to_key in transition_dict[node_key]:
                    to_action_name = to_key.split(split_key)[0]
                    to_mp_name = to_key.split(split_key)[1]
                    to_node_key = (to_action_name, to_mp_name)
                    if to_node_key in motion_state_graph.nodes.keys():
                        self._add_transition(motion_state_graph, from_node_key, to_node_key)
            else:
                print("did not find", from_node_key, "in graph", len( motion_state_graph.nodes))

    def _get_transition_type(self, graph, from_node_key, to_node_key):
        t_type = "action_transition"
        if to_node_key[0] == from_node_key[0]:
            if graph.nodes[from_node_key].node_type == NODE_TYPE_IDLE:
                if graph.nodes[to_node_key].node_type == NODE_TYPE_START:
                    t_type = NODE_TYPE_START
                elif graph.nodes[to_node_key].node_type == NODE_TYPE_IDLE:
                    t_type = NODE_TYPE_IDLE
                elif graph.nodes[to_node_key].node_type == NODE_TYPE_END:
                    t_type = NODE_TYPE_END
            else:
                if graph.nodes[to_node_key].node_type == NODE_TYPE_STANDARD:
                    t_type = NODE_TYPE_STANDARD
                elif graph.nodes[to_node_key].node_type == NODE_TYPE_START:
                    t_type = NODE_TYPE_START
                elif graph.nodes[to_node_key].node_type == NODE_TYPE_CYCLE_END:
                    t_type = "cycle_end"
                elif graph.nodes[to_node_key].node_type == NODE_TYPE_IDLE:
                    t_type = NODE_TYPE_IDLE
                else:
                    t_type = NODE_TYPE_END
        return t_type

    def _add_transition(self, graph, from_key, to_key):
        transition_model = None
        transition_type = self._get_transition_type(graph, from_key, to_key)
        graph.nodes[from_key].outgoing_edges[to_key] = MotionStateTransition(from_key, to_key, transition_type, transition_model)
    
    def build_node_group_from_dict(self, action_data, graph):
        mp_node_group = MotionStateGroup(action_data["name"], None, graph)
        for mp_name in action_data["nodes"].keys():
            node_key = (action_data["name"], mp_name)
            mp_node_group.nodes[node_key] = MotionStateGraphNode(mp_node_group)
            print("init", node_key)
            mp_node_group.nodes[node_key].init_from_dict(action_data["name"], action_data["nodes"][mp_name])

        if "info" in action_data.keys():
            mp_node_group.set_meta_information(action_data["info"])
        else:
            mp_node_group.set_meta_information()

        for mp_name in action_data["nodes"]:
            if "keyframes" in action_data["nodes"][mp_name]["mm"]:
                keyframes = action_data["nodes"][mp_name]["mm"]["keyframes"]
                for label, frame_idx in keyframes.items():
                    if label not in mp_node_group.label_to_motion_primitive_map:
                        mp_node_group.label_to_motion_primitive_map[label] = list()
                    mp_node_group.label_to_motion_primitive_map[label].append(mp_name)
                if mp_name not in mp_node_group.labeled_frames:
                    mp_node_group.labeled_frames[mp_name] = dict()
                mp_node_group.labeled_frames[mp_name].update(keyframes)

        return mp_node_group
