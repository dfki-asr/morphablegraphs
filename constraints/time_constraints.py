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


class TimeConstraints(object):
    def __init__(self, motion_primitive_graph, graph_walk, start_step, end_step, constraint_list):
        self.start_step = start_step
        self.end_step = end_step
        self.start_keyframe = self._get_start_frame(motion_primitive_graph, graph_walk, start_step)
        self.constraint_list = constraint_list

    def _get_start_frame(self, motion_primitive_graph, graph_walk, start_step):
        if start_step <= 0:
            return 0
        start_keyframe = 0
        for i in range(0, start_step):  # until start_step - 1
            time_function = motion_primitive_graph.nodes[graph_walk.steps[i].node_key].back_project_time_function(graph_walk.steps[i].parameters)
            start_keyframe += time_function[-1]
        return start_keyframe

    def evaluate_graph_walk(self, s, motion_primitive_graph, graph_walk):
        #print "evaluate", s
        time_functions = self._get_time_functions_from_graph_walk(s, motion_primitive_graph, graph_walk)
        #get difference to desired time for each constraint
        #print "got time functions"
        frame_time = motion_primitive_graph.skeleton.frame_time
        error_sum = 0
        for time_constraint in self.constraint_list:
            error_sum += self.calculate_constraint_error(time_functions, time_constraint, frame_time)
        return error_sum

    def _get_time_functions_from_graph_walk(self, s, motion_primitive_graph, graph_walk):
        """get time functions for all steps in the graph walk.
        """
        time_functions = []
        offset = 0
        for step in graph_walk.steps[self.start_step:self.end_step]:
            #print step.node_key
            gamma = s[offset:offset+step.n_time_components]
            s_vector = np.array(step.parameters)
            s_vector[step.n_spatial_components:] = gamma
            time_function = motion_primitive_graph.nodes[step.node_key].back_project_time_function(s_vector)
            time_functions.append(time_function)
            offset += step.n_time_components
        return time_functions

    def calculate_constraint_error(self, time_functions, time_constraint, frame_time):
        constrained_step_index, constrained_keyframe_index, desired_time = time_constraint
        n_frames = self.start_keyframe #when it starts the first step start_keyframe would be 0
        temp_step_index = 0
        for time_function in time_functions:  # look back n_steps
            if temp_step_index < constrained_step_index:# go to the graph walk entry we want to constrain
                #simply add the number of frames
                n_frames += time_function[-1]
                temp_step_index += 1
            else:
                if constrained_keyframe_index >= len(time_function):
                    return 0
                warped_keyframe = int(time_function[constrained_keyframe_index]) + 1
                n_frames += warped_keyframe
                total_seconds = n_frames * frame_time
                error = (desired_time-total_seconds)**2
                #print time_function
                print("time error", constrained_keyframe_index, error, total_seconds, desired_time,n_frames,frame_time)#, mapped_keyframe, constrained_keyframe_index, n_frames
                return error
        return 10000

    def get_average_loglikelihood(self, s, motion_primitive_graph, graph_walk):
        likelihood = 0
        step_count = 0
        offset = 0
        for step in graph_walk.steps[self.start_step:self.end_step]:
            parameters = step.parameters[:step.n_spatial_components].tolist() + s[offset:offset+step.n_time_components].tolist()
            likelihood += motion_primitive_graph.nodes[step.node_key].get_gaussian_mixture_model().score(np.array(parameters).reshape(1,len(parameters)))[0]
            step_count += 1
            offset += step.n_time_components
        return likelihood/step_count

    def get_initial_guess(self, graph_walk):
        parameters = []
        for step in graph_walk.steps[self.start_step:self.end_step]:
            #print step.node_key
            parameters += step.parameters[step.n_spatial_components:].tolist()
        return parameters

    def get_error_gradient(self, motion_primitive_graph, graph_walk,frame_time):
        gradient = []
        for step in graph_walk.steps[self.start_step:self.end_step]:
            gradient += motion_primitive_graph.nodes[step.node_key].get_time_eigen_vector_matrix()[-1]*frame_time
        return 2 * np.array(gradient)
