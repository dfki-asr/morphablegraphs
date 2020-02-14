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
import threading
import time
import os
from copy import copy, deepcopy
import numpy as np
from datetime import datetime
from transformations import quaternion_multiply, quaternion_slerp, quaternion_about_axis, quaternion_matrix
from anim_utils.animation_data.motion_state import MotionState
from morphablegraphs.constraints.constraint_builder import ConstraintBuilder, MockActionConstraints
from anim_utils.animation_data.motion_concatenation import align_quaternion_frames, smooth_quaternion_frames, get_node_aligning_2d_transform, transform_quaternion_frames, get_orientation_vector_from_matrix, get_rotation_angle
from anim_utils.motion_editing import MotionEditing
from anim_utils.animation_data.motion_vector import MotionVector
from anim_utils.animation_data.skeleton_models import JOINT_CONSTRAINTS
from morphablegraphs.constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
from morphablegraphs.motion_generator.algorithm_configuration import DEFAULT_ALGORITHM_CONFIG
from morphablegraphs.motion_generator.motion_primitive_generator import MotionPrimitiveGenerator
from morphablegraphs.motion_generator.optimization.optimizer_builder import OptimizerBuilder
from morphablegraphs.motion_model import NODE_TYPE_STANDARD, NODE_TYPE_END, NODE_TYPE_START, NODE_TYPE_IDLE
from morphablegraphs.motion_model.static_motion_primitive import StaticMotionPrimitive
from anim_utils.motion_editing.cubic_motion_spline import CubicMotionSpline
from .utils import normalize, get_root_delta_q, smooth_quaternion_frames2, get_trajectory_end_direction

REF_VECTOR = [0,0,1]


class StateQueueEntry(object):
        def __init__(self, new_node, node_type, new_state, pose_buffer):
            self.node = new_node
            self.node_type = node_type
            self.state = new_state
            self.pose_buffer = pose_buffer

        def get_n_frames(self):
            return self.state.get_n_frames()


class MGStateQueue(object):
    def __init__(self, skeleton, graph, frame_time,  settings):
        self.skeleton = skeleton
        self.n_joints = len(self.skeleton.animated_joints)
        self._graph = graph
        self.frame_time = frame_time
        self.settings = settings
        self.state_queue = list()
        self.mutex = threading.Lock()
    
    def append_state_to_queue(self, new_entry):
        self.mutex.acquire()
        self.state_queue.append(new_entry)
        self.mutex.release()

    def create_state_queue_entry(self, current_node, node_type, dt, pose_buffer, new_frames, events, hold_frames, export=False):
        # build state and update pose buffer
        new_state = self.build_state(new_frames, pose_buffer, export=export)
        new_state.play = True
        new_state.events = events
        while not new_state.update(dt):
            pose_buffer.append(new_state.get_pose())
        new_state.set_frame_idx(0)
        hold_frames = list(hold_frames)
        hold_frames.sort()
        new_state.hold_frames = hold_frames
        # print("add state", len(states))
        pose_buffer = pose_buffer[-self.settings.buffer_size:]
        buffer_copy = copy(pose_buffer)
        new_entry = StateQueueEntry(current_node, node_type, new_state, buffer_copy)
        return new_entry
    
    def add_state_queue_entry(self, current_node, node_type, new_frames, pose_buffer, dt, new_state=None):
        # build state
        if new_state is None:
            new_state = self.build_state(new_frames, pose_buffer)
        new_state.play = True

        # update pose buffer
        while not new_state.update(dt):
            pose_buffer.append(new_state.get_pose())
        new_state.set_frame_idx(0)

        # print("add state", len(states))
        pose_buffer = pose_buffer[-self.settings.buffer_size:]
        buffer_copy = copy(pose_buffer)
        new_entry = StateQueueEntry(current_node, node_type, new_state, buffer_copy)
        self.append_state_to_queue(new_entry)

    def build_state(self, frames, pose_buffer, ignore_rotation=False, export=False):
        if ignore_rotation:
            pose_buffer[-1][3:7] = [1, 0, 0, 0]
        if pose_buffer is not None:
            m = get_node_aligning_2d_transform(self.skeleton, self.skeleton.aligning_root_node, pose_buffer, frames)
            #if export:
            #    print("aligning state transform",m)
            frames = transform_quaternion_frames(frames, m)
            new_frames = smooth_quaternion_frames2(pose_buffer[-1], frames, self.settings.blend_window, True)
        else:
            new_frames = frames
        if export:
            time_stamp = str(datetime.now().strftime("%d%m%y_%H%M%S"))
            mv = MotionVector()
            mv.frames = frames
            mv.n_frames = len(frames)
            mv.export(self.skeleton,  "data" + os.sep + "global"+time_stamp+"global.bvh")
        mv = MotionVector(self.skeleton)
        mv.frames = new_frames#[1:]
        mv.frame_time = self.frame_time
        mv.n_frames = len(mv.frames)
        state = MotionState(mv)
        return state

    def build_pfnn_state(self, frames, pose_buffer, aligning_transform, apply_smoothing=False):
        new_frames = transform_quaternion_frames(frames, aligning_transform)
        if apply_smoothing:
            new_frames = smooth_quaternion_frames2(pose_buffer[-1], new_frames, self.settings.blend_window, True)
        mv = MotionVector(self.skeleton)
        mv.frames = new_frames[1:]
        mv.frame_time = self.frame_time
        mv.n_frames = len(mv.frames)
        state = MotionState(mv)
        return state

    def build_state_old(self, frames, pose_buffer, ignore_rotation=False):
        if ignore_rotation:
            pose_buffer[-1][3:7] = [1, 0, 0, 0]
        m = get_node_aligning_2d_transform(self.skeleton, self.skeleton.aligning_root_node, pose_buffer, frames)
        frames = transform_quaternion_frames(frames, m)
        len_pose_buffer = len(pose_buffer)
        new_frames = smooth_quaternion_frames(np.array(pose_buffer + frames.tolist()), len_pose_buffer, 20, True)
        new_frames = new_frames[len_pose_buffer:]
        new_frames = self.apply_smoothing(new_frames, pose_buffer[-1], 0, self.settings.blend_window)

        mv = MotionVector(self.skeleton)
        mv.frames = new_frames[2:]
        mv.frame_time = self.frame_time
        mv.n_frames = len(mv.frames)
        state = MotionState(mv)
        return state

    def generate_idle_state(self, dt, pose_buffer, append_to_queue=True):
        current_node = self._graph.start_node 
        new_frames = self._graph.nodes[current_node].sample().get_motion_vector()
        if self.settings.use_all_joints:
            new_frames = self.complete_frames(current_node, new_frames)
        new_state = self.build_state(new_frames, pose_buffer)
        if pose_buffer is None:
            pose_buffer = []
        new_state.play = True
        while not new_state.update(dt):
            pose_buffer.append(new_state.get_pose())
        new_state.set_frame_idx(0)
        # print("add state", len(states))
        pose_buffer = pose_buffer[-self.settings.buffer_size:]
        buffer_copy = copy(pose_buffer)
        new_entry = StateQueueEntry(current_node, NODE_TYPE_IDLE, new_state, buffer_copy)
        if append_to_queue:
            self.append_state_to_queue(new_entry)
        return new_entry

    def apply_smoothing(self, new_frames, ref_frame, start_frame, window):
        for i in range(window):
            t = (i / window)
            for idx in range(len(self.skeleton.animated_joints)):
                o = idx*4 + 3
                indices = [o, o+1, o+2, o+3]
                old_quat = ref_frame[indices]
                new_quat = new_frames[start_frame + i, indices]
                new_frames[start_frame + i, indices] = quaternion_slerp(old_quat, new_quat, t, spin=0, shortestpath=True)
        return new_frames

    def align_frames(self, frame, ref_frame):
        for i in range(self.n_joints):
            o = i*4+3
            q = frame[o:o+4]
            frame[o:o+4] = -q if np.dot(ref_frame[o:o+4], q) < 0 else q
        return frame

    def apply_end_orientation_correction(self, end_orientation):
        state_entry = self.get_last_state()
        if state_entry is None:
            return
        if self.settings.end_target_blend_range <= 0:
            state_entry.state.mv.frames[-1, 3:7] = end_orientation
        else:
            blend_range = min(self.settings.end_target_blend_range, state_entry.state.mv.n_frames)
            start_idx = state_entry.state.mv.n_frames-blend_range
            if start_idx < 0:
                blend_range = blend_range + start_idx
                start_idx = 0
            #clipped_start_idx = max(start_idx, 0)
            #unit_q = np.array([1,0,0,0])
            n_frames = len(state_entry.state.mv.frames)
            for i in range(blend_range):
                frame_idx = start_idx+i
                if 0 < frame_idx < n_frames:
                    w = float(i) / blend_range
                    q0 = state_entry.state.mv.frames[frame_idx, 3:7]
                    q0 = normalize(q0)
                    new_q = quaternion_slerp(q0, end_orientation, w)
                    new_q = normalize(new_q)
                    state_entry.state.mv.frames[frame_idx, 3:7] = new_q
            state_entry.state.mv.frames[-1, 3:7] = end_orientation
        #print("after alignment",get_root_delta_angle(self.skeleton, pelvis_name, self.state_queue[-1].state.mv.frames, end_direction))
        if len(self.state_queue) > 0:
            print("apply end orentation correction")
            self.state_queue[-1] = state_entry

    def apply_end_pos_correction(self, step_target):
        state_entry = self.get_last_state()
        if state_entry is None:
            return
        blend_range= 0
        #step_target = np.array([0,0,0])
        if self.settings.end_target_blend_range <= 0:
           state_entry.state.mv.frames[-1, :3] = step_target
        else:
            blend_range = min(self.settings.end_target_blend_range, state_entry.state.mv.n_frames)
            delta = step_target - state_entry.state.mv.frames[-1, :3]
            weights = np.zeros((blend_range, 1))
            for i in range(1, blend_range + 1):
                weights[i - 1] = i / blend_range
            delta[1] = 0
            delta_matrix = np.dot(weights, delta.reshape((1, 3)))
            n_frames = len(state_entry.state.mv.frames)
            for i in range(blend_range):
                frame_idx = -(blend_range - i)
                #print("overwrite", i, blend_range, frame_idx, n_frames, state_entry.state.mv.n_frames)
                state_entry.state.mv.frames[frame_idx, :3] += delta_matrix[i]
        #print(state_entry.state.mv.frames[-1, :3], step_target)
        #print("apply end pos correction!!!!")
        if len(self.state_queue) > 0:
            self.state_queue[-1] = state_entry

    def apply_end_orientation_by_direction(self, end_direction, ref_vector=REF_VECTOR):
        #pelvis_name = self.skeleton.skeleton_model["joints"]["pelvis"]
        ref_vector = [0,0,1]
        #ref_vector = [0,0,1]
        pelvis_name = self.skeleton.aligning_root_node
        delta_q = get_root_delta_q(self.skeleton, pelvis_name, self.state_queue[-1].state.mv.frames, end_direction, ref_vector)
        delta_q = normalize(delta_q)
        current_q = self.state_queue[-1].state.mv.frames[-1, 3:7]
        current_q = normalize(current_q)
        end_q = quaternion_multiply(delta_q, current_q)
        end_q = normalize(end_q)
        self.apply_end_orientation_correction(end_q)

    def get_last_state(self):
        if len(self.state_queue) > 0:
            return self.state_queue[-1]
        else:
            return None

    def pop_last_state(self):
        return self.state_queue.pop(-1)

    def get_first_state(self):
        if len(self.state_queue) > 0:
            return self.state_queue[0]
        else:
            return None

    def pop_first_state(self):
        return self.state_queue.pop(0)

    def __len__(self):
        return len(self.state_queue)

    def reset(self):
        self.state_queue = list()

    def __getitem__(self, idx):
        return self.state_queue[idx]

    def complete_frames(self,current_node, new_frames):
        animated_joints = self._graph.nodes[current_node].get_animated_joints()
        #print("animated joints", animated_joints)
        new_full_frames = np.zeros((len(new_frames), self.skeleton.reference_frame_length))
        for idx, reduced_frame in enumerate(new_frames):
            new_full_frames[idx] = self.skeleton.add_fixed_joint_parameters_to_other_frame(reduced_frame,
                                                                                     animated_joints)
        return new_full_frames
