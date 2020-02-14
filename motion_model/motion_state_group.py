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
Created on Thu Jul 16 15:57:42 2015


@author: Erik Herrmann
"""

from anim_utils.utilities.log import write_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR
from . import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END, NODE_TYPE_SINGLE,  NODE_TYPE_CYCLE_END, NODE_TYPE_IDLE
from .action_meta_info import ActionMetaInfo
import numpy as np


class MotionStateGroup(ActionMetaInfo):
    """ Contains the motion primitives of an elementary action as nodes.
    """
    def __init__(self, action_name, src_directory, motion_state_graph):
        super(MotionStateGroup, self).__init__(action_name, src_directory)
        self.motion_state_graph = motion_state_graph
        self.nodes = dict()
        self.has_transition_models = False
        self.loaded_from_dict = src_directory is None

    def set_meta_information(self, meta_information=None):
        super(MotionStateGroup, self).set_meta_information(meta_information)
        write_message_to_log("action" + str(self.ea_name), LOG_MODE_DEBUG)
        write_message_to_log("start states" + str(self.start_states), LOG_MODE_DEBUG)
        if len(self.nodes) == 1:
            node_key = list(self.nodes.keys())[0]
            self.nodes[node_key].node_type = NODE_TYPE_SINGLE
        else:
            for k in self.start_states:
                self.nodes[(self.ea_name, k)].node_type = NODE_TYPE_START
            #write_message_to_log("end states" + str(self.end_states), LOG_MODE_DEBUG)
            for k in self.end_states:
                self.nodes[(self.ea_name, k)].node_type = NODE_TYPE_END
            for k in self.cycle_states:
                 self.nodes[(self.ea_name, k)].node_type = NODE_TYPE_CYCLE_END
            for k in self.idle_states:
                self.nodes[(self.ea_name, k)].node_type = NODE_TYPE_IDLE

    def get_action_type(self):
        n_standard_nodes = 0
        for node_key in list(self.nodes.keys()):
            if self.nodes[node_key].node_type == NODE_TYPE_STANDARD:
                n_standard_nodes += 1
        if n_standard_nodes > 0:
            return "locomotion"
        else:
            return "upper body"

    def _update_motion_state_stats(self, recalculate=False):
        """  Update stats of motion states for faster lookup.
        """
        changed_meta_info = False
        if recalculate:
            changed_meta_info = True
            self.meta_information["stats"] = dict()
            for node_key in self.nodes:
                self.nodes[node_key].update_motion_stats()
                self.meta_information["stats"][node_key[1]] = {"average_step_length": self.nodes[node_key].average_step_length,
                                                               "n_standard_transitions": self.nodes[node_key].n_standard_transitions}
                write_message_to_log("n standard transitions " + str(node_key) + " " + str(self.nodes[node_key].n_standard_transitions), LOG_MODE_DEBUG)
            write_message_to_log("Updated meta information " + str(self.meta_information), LOG_MODE_DEBUG)
        else:
            if self.meta_information is None:
                self.meta_information = dict()
            if "stats" not in self.meta_information:
                self.meta_information["stats"] = dict()
            for node_key in self.nodes:
                if node_key[1] in self.meta_information["stats"]:
                    meta_info = self.meta_information["stats"][node_key[1]]
                    print(node_key, meta_info)
                    self.nodes[node_key].n_standard_transitions = meta_info["n_standard_transitions"]
                    self.nodes[node_key].average_step_length = meta_info["average_step_length"]
                else:
                    self.nodes[node_key].update_motion_stats()
                    self.meta_information["stats"][node_key[1]] = {"average_step_length": self.nodes[node_key].average_step_length,
                                                                   "n_standard_transitions": self.nodes[node_key].n_standard_transitions }
                    changed_meta_info = True
            write_message_to_log("Loaded stats from meta information file " + str(self.meta_information), LOG_MODE_DEBUG)
        if changed_meta_info and not self.loaded_from_dict:
            self.save_updated_meta_info()

    def generate_next_parameters(self, current_node_key, current_parameters, to_node_key, use_transition_model):
        """ Generate parameters for transitions.
        
        Parameters
        ----------
        * current_state: string
        \tName of the current motion primitive
        * current_parameters: np.ndarray
        \tParameters of the current state
        * to_node_key: tuple
        \t Identitfier of the action and motion primitive we want to transition to.
        \t Should have the format (action name, motionprimitive name)
        * use_transition_model: bool
        \t flag to set whether a prediction from the transition model should be made or not.
        """
        assert to_node_key[0] == self.ea_name
        if self.has_transition_models and use_transition_model:
            print("use transition model", current_node_key, to_node_key)
            next_parameters = self.nodes[current_node_key].predict_parameters(to_node_key, current_parameters)
        else:
            next_parameters = self.nodes[to_node_key].sample_low_dimensional_vector()#[0]
            print("sample from model", to_node_key, next_parameters.shape)
        return np.ravel(next_parameters)

    def get_transition_type_for_action_from_trajectory(self, graph_walk, action_constraint, travelled_arc_length, arc_length_of_end):

        #test end condition for trajectory constraints
        if not action_constraint.check_end_condition(graph_walk.get_quat_frames(), travelled_arc_length, arc_length_of_end):

            #make standard transition to go on with trajectory following
            next_node_type = NODE_TYPE_STANDARD
        else:
            # threshold was overstepped. remove previous step before
            # trying to reach the goal using a last step
            next_node_type = NODE_TYPE_END

        write_message_to_log("Generate "+ str(next_node_type) + " transition from trajectory", LOG_MODE_DEBUG)
        return next_node_type

    def get_transition_type_for_action(self, graph_walk, action_constraint):
        prev_node = graph_walk.steps[-1].node_key
        n_standard_transitions = len(self.get_n_standard_transitions(prev_node))
        if n_standard_transitions > 0:
            next_node_type = NODE_TYPE_STANDARD
        else:
            next_node_type = NODE_TYPE_END
        write_message_to_log("Generate " + str(next_node_type) + " transition without trajectory " + str(n_standard_transitions) +" " + str(prev_node), LOG_MODE_DEBUG)
        if action_constraint.cycled_next and next_node_type == NODE_TYPE_END:
            next_node_type = NODE_TYPE_CYCLE_END
        return next_node_type

    def get_n_standard_transitions(self, prev_node):
        return [e for e in list(self.nodes[prev_node].outgoing_edges.keys())
                if self.nodes[prev_node].outgoing_edges[e].transition_type == NODE_TYPE_STANDARD]

    def get_random_transition(self, graph_walk, action_constraint, travelled_arc_length, arc_length_of_end):
        """ Get next state of the elementary action based on previous iteration.
        """
        prev_node = graph_walk.steps[-1].node_key
        if action_constraint.root_trajectory is None:
            next_node_type = self.get_transition_type_for_action(graph_walk, action_constraint)
        else:
            next_node_type = self.get_transition_type_for_action_from_trajectory(graph_walk, action_constraint, travelled_arc_length, arc_length_of_end)

        to_node_key = self.nodes[prev_node].generate_random_transition(next_node_type)
        if to_node_key is not None:
            return to_node_key, next_node_type
        else:
            return None, next_node_type

    def generate_random_walk(self, state_node, number_of_steps, use_transition_model=True):
        """ Generates a random graph walk to be converted into a BVH file

        Parameters
        ----------
        * start_state: string
        \tInitial state.
        * number_of_steps: integer
        \tNumber of transitions
        * use_transition_model: bool
        \tSets whether or not the transition model should be used in parameter prediction
        """
        current_node = state_node
        assert current_node in list(self.nodes.keys())
        graph_walk = []
        count = 0
        #print "start", current_node
        current_parameters = self.nodes[current_node].sample_low_dimensional_vector()#[0]
        current_parameters = np.ravel(current_parameters)
        entry = {"node_key": current_node, "parameters": current_parameters}
        graph_walk.append(entry)

        if self.nodes[current_node].n_standard_transitions > 0:
            while count < number_of_steps:
                to_node_key = self.nodes[current_node].generate_random_transition(NODE_TYPE_STANDARD)
                s = self.generate_next_parameters(current_node, current_parameters, to_node_key, use_transition_model)
                entry = {"node_key": to_node_key, "parameters": s}
                graph_walk.append(entry)
                current_parameters = s
                current_node = to_node_key
                count += 1
        #add end node
        to_node_key = self.nodes[current_node].generate_random_transition(NODE_TYPE_END)
        if to_node_key is not None:
            next_parameters = self.generate_next_parameters(current_node, current_parameters, to_node_key, use_transition_model)
            entry = {"node_key": to_node_key, "parameters": next_parameters}
            graph_walk.append(entry)
        return graph_walk

    def has_cycle_states(self):
        return len(self.cycle_states) > 0

    def map_label_to_keyframe(self, mp_name, label):
        keyframe = None
        node_key = (self.ea_name, mp_name)
        n_canonical_frames = self.motion_state_graph.nodes[node_key].get_n_canonical_frames()
        if mp_name in self.labeled_frames.keys() and label in self.labeled_frames[mp_name].keys():
            keyframe = self.labeled_frames[mp_name][label]
            if keyframe in [-1, "lastFrame"]:
                keyframe = n_canonical_frames - 1
            elif keyframe == "middle":
                keyframe = n_canonical_frames / 2
        if keyframe is not None:
            keyframe = int(keyframe)
        return keyframe