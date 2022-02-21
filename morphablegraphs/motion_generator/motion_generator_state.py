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
import numpy as np
from ..motion_model import NODE_TYPE_END, NODE_TYPE_SINGLE, NODE_TYPE_CYCLE_END


class MotionGeneratorState(object):
    def __init__(self, algorithm_config):
        self.start_step = -1
        self.prev_action_name = None
        self.prev_mp_name = None
        self.action_start_frame = -1
        self.current_node = None
        self.current_node_type = ""
        self.temp_step = 0
        self.travelled_arc_length = 0.0
        self.debug_max_step = algorithm_config["debug_max_step"]
        self.step_start_frame = 0
        self.max_arc_length = np.inf
        self.action_cycled_next = False
        self.overstepped = False

    def initialize_from_previous_graph_walk(self, graph_walk, max_arc_length, action_cycled_next):
        self.start_step = graph_walk.step_count
        if self.start_step > 0:
            self.prev_action_name = graph_walk.steps[-1]
            self.prev_mp_name = graph_walk.steps[-1]
        else:
            self.prev_action_name = None
            self.prev_mp_name = None
        self.action_start_frame = graph_walk.get_num_of_frames()
        self.current_node = None
        self.current_node_type = ""
        self.temp_step = 0
        self.travelled_arc_length = 0.0
        self.max_arc_length = max_arc_length
        self.action_cycled_next = action_cycled_next
        self.overstepped = False

    def is_end_state(self):
        return self.is_last_node() or self.reached_debug_max_step() or self.reached_max_arc_length()

    def reached_debug_max_step(self):
        return self.start_step + self.temp_step > self.debug_max_step and self.debug_max_step > -1

    def reached_max_arc_length(self):
        return self.travelled_arc_length >= self.max_arc_length

    def is_last_node(self):
        return self.current_node_type == NODE_TYPE_END or \
                self.current_node_type == NODE_TYPE_SINGLE or\
                (self.current_node is not None and self.action_cycled_next)

    def transition(self, new_node, new_node_type, new_travelled_arc_length, new_step_start_frame):
        self.current_node = new_node
        self.current_node_type = new_node_type
        self.travelled_arc_length = new_travelled_arc_length
        self.step_start_frame = new_step_start_frame
        self.temp_step += 1
