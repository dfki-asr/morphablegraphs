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

import collections
import random
import numpy as np
from anim_utils.utilities.log import write_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR


class MotionStateGraph(object):
    """ Contains a dict of MotionStateGraphNodes and MotionStateGroups for each action,
         transitions between them are stored as outgoing edges in the nodes.
    """
     
    def __init__(self):
        """ Initializes the class
        """
        self.skeleton = None
        self.mgrd_skeleton = None
        self.node_groups = collections.OrderedDict()
        self.nodes = collections.OrderedDict()
        self.hand_pose_generator = None
        self.animated_joints = None
        self.action_definitions = None
        self.start_node = None

    def generate_random_walk(self, start_action, number_of_steps, use_transition_model=True):
        """ Generates a random graph walk
        Parameters
        ----------
        * start_action: string
            Initial action.
        * number_of_steps: integer
            Number of transitions
        * use_transition_model: bool
            Sets whether or not the transition model should be used in parameter prediction
        Returns
        -------
        *graph_walk: a list of dictionaries
            The graph walk is defined by a list of dictionaries containing entries for "action","motion primitive" and "parameters"
        """
        assert start_action in list(self.node_groups.keys())
        print("generate random graph walk for", start_action)
        start_state = self.node_groups[start_action].get_random_start_state()
        print(start_state)
        return self.node_groups[start_action].generate_random_walk(self.nodes, start_state, number_of_steps, use_transition_model)
    
    def print_information(self):
        """
        Prints out information on the graph structure and properties of the motion primitives
        """
        for s in list(self.node_groups.keys()):
            print(s)
            for n in list(self.node_groups[s].nodes.keys()):
                print("\t"+ str(n))
                print("\t"+"n canonical frames", self.nodes[n].n_canonical_frames)
                print("\t"+"n latent spatial dimensions", self.nodes[n].s_pca["n_components"])
                print("\t"+"n latent time dimensions", self.nodes[n].t_pca["n_components"])
                print("\t"+"n basis spatial ", self.nodes[n].s_pca["n_basis"])
                print("\t"+"n basis time ", self.nodes[n].t_pca["n_basis"])
                print("\t"+"n clusters", len(self.nodes[n].gaussian_mixture_model.weights_))
                print("\t"+"average length", self.nodes[n].average_step_length)
                for e in list(self.nodes[n].outgoing_edges.keys()):
                    print("\t \t to " + str(e))
                print("\t##########")       

    def get_random_action_transition(self, graph_walk, action_name, is_cycle=False):
        """ Get random start state based on edge from previous elementary action if possible
        """
        next_node = None
        if graph_walk.step_count > 0:
            prev_node_key = graph_walk.steps[-1].node_key
      
            if prev_node_key in list(self.nodes.keys()):
                next_node = self.nodes[prev_node_key].generate_random_action_transition(action_name,is_cycle)
            write_message_to_log("Generate start from transition of last action " + str(prev_node_key) + str(next_node), mode=LOG_MODE_DEBUG)
        # if there is no previous elementary action or no action transition
        #  use transition to random start state
        if next_node is None or next_node not in self.node_groups[action_name].nodes:
            #print next_node, "not in", action_name
            next_node = self.node_groups[action_name].get_random_start_state()
            write_message_to_log("Generate random start" + str(next_node), mode=LOG_MODE_DEBUG)
        return next_node

    def get_start_nodes(self, action_name):
        """
        Get all start state based on edge from previous elementary action if possible
        :param action_name:
        :return:
        """
        return self.node_groups[action_name].get_start_states()

    def get_random_start_node(self):
        """ If there are start sates defined in the graph a node key tuple is returned  (action_name, state_name)
            else None is returned.
        """
        start_node = None
        if len(self.node_groups) > 0:
            actions = list(self.node_groups.keys())
            #random_index = random.randrange(0, len(actions), 1)
            indices = list(range(len(actions)))
            np.random.shuffle(indices)
            for i in indices:
                action_name = actions[i]
                start_states = self.get_start_nodes(action_name)
                if len(start_states) > 0:
                    random_index = random.randrange(0, len(start_states), 1)
                    state_name = start_states[random_index]
                    start_node = action_name, state_name
                    break
            if start_node is None:
                print("Error: no start node", len(self.node_groups))
        return start_node
