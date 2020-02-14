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


class ActionConstraints(object):
    def __init__(self):
        self.motion_state_graph = None
        self.action_name = None
        self.prev_action_name = ""
        self.keyframe_annotations = None
        self.start_pose = None
        self.trajectory_constraints = None
        self.collision_avoidance_constraints = None
        self.annotated_trajectory_constraints = None
        self.ca_trajectory_set_constraint = None
        self.root_trajectory = None
        self.keyframe_constraints = None
        self.precision = {"pos": 1.0, "rot": 1.0, "smooth": 1.0}
        self._initialized = False
        self.contains_user_constraints = False  # any user defined keyframe constraints
        self.contains_two_hands_constraints = False  # two hand pick or place
        self.cycled_previous = False  # are there more of the same action before
        self.cycled_next = False  # are there more of the same action following
        self.group_id = ""

    def get_node_group(self):
        return self.motion_state_graph.node_groups[self.action_name]

    def get_skeleton(self):
        return self.motion_state_graph.skeleton

    def check_end_condition(self, prev_frames, travelled_arc_length, arc_length_offset):
        """
        Checks whether or not a threshold distance to the end has been reached.
        Returns
        -------
        True if yes and False if not
        """
        distance_to_end = np.linalg.norm(self.root_trajectory.get_last_control_point() - prev_frames[-1][:3])
    #    print "current distance to end: " + str(distance_to_end)
    #    print "travelled arc length is: " + str(travelled_arc_length)
    #    print "full arc length is; " + str(trajectory.full_arc_length)
    #    raw_input("go on...")
    
        continue_with_the_loop = distance_to_end > arc_length_offset/2 and \
                                 travelled_arc_length < self.root_trajectory.full_arc_length - arc_length_offset
        return not continue_with_the_loop
