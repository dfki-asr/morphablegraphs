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
from copy import copy, deepcopy
import numpy as np
import os
import sys
import collections
from datetime import datetime
from transformations import quaternion_multiply, quaternion_slerp, quaternion_about_axis, quaternion_matrix, quaternion_from_matrix
from morphablegraphs.constraints.constraint_builder import ConstraintBuilder, MockActionConstraints
from anim_utils.animation_data.motion_concatenation import align_quaternion_frames, smooth_quaternion_frames, get_node_aligning_2d_transform, transform_quaternion_frames, get_orientation_vector_from_matrix, get_rotation_angle
from anim_utils.motion_editing import MotionEditing
from anim_utils.animation_data.skeleton import run_ccd
from anim_utils.animation_data.motion_vector import MotionVector
from anim_utils.animation_data.skeleton_models import JOINT_CONSTRAINTS, UPPER_BODY_JOINTS
from anim_utils.motion_editing.cubic_motion_spline import CubicMotionSpline
from anim_utils.animation_data.skeleton import LOOK_AT_DIR, SPINE_LOOK_AT_DIR
from anim_utils.retargeting.analytical import create_local_cos_map_from_skeleton_axes_with_map
from anim_utils.motion_editing.ik_constraints_builder import IKConstraintsBuilder
from morphablegraphs.constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
from morphablegraphs.motion_generator.algorithm_configuration import DEFAULT_ALGORITHM_CONFIG
from morphablegraphs.motion_generator.motion_primitive_generator import MotionPrimitiveGenerator
from morphablegraphs.motion_generator.optimization.optimizer_builder import OptimizerBuilder
from morphablegraphs.motion_model import NODE_TYPE_STANDARD, NODE_TYPE_END, NODE_TYPE_START, NODE_TYPE_IDLE
from morphablegraphs.motion_model.static_motion_primitive import StaticMotionPrimitive
from morphablegraphs.motion_generator.mg_state_queue import MGStateQueue
from anim_utils.motion_editing.motion_editing import KeyframeConstraint
from .utils import normalize, get_root_delta_q, smooth_quaternion_frames2, get_trajectory_end_direction

HAND_JOINTS = ["right_wrist","left_wrist"]

ANIMATED_JOINTS_CUSTOM = [
    "FK_back1_jnt",
    "FK_back2_jnt",
    "FK_back4_jnt",
    "head_jnt",
    "R_shoulder_jnt",
    "R_upArm_jnt",
    "R_lowArm_jnt",
    "R_hand_jnt",
    "L_shoulder_jnt",
    "L_upArm_jnt",
    "L_lowArm_jnt",
    "L_hand_jnt",
    "L_upLeg_jnt",
    "L_lowLeg_jnt",
    "L_foot_jnt",
    "R_upLeg_jnt",
    "R_lowLeg_jnt",
    "R_foot_jnt"
]

REF_VECTOR = np.array([0,0,1])

def stretch_points(points, scale):
    points = np.array(points)
    n_points = len(points)
    times = list(range(0, n_points))
    spline = CubicMotionSpline.fit_frames(None, times, points)
    strechted_times = np.arange(0,n_points-1, scale)
    new_points = []
    for t in strechted_times:
        #print(t, n_points)
        f = spline.evaluate(t)
        new_points.append(f)
    print("new frames", len(new_points))
    return new_points

def get_node_aligning_2d_transform_pfnn(skeleton, node_name, prev_frames, new_frames, ref_vector=REF_VECTOR):
    """from last of prev frames to first of new frames"""
    m_a = skeleton.nodes[node_name].get_global_matrix(prev_frames[-1])
    dir_vec_a = get_orientation_vector_from_matrix(m_a[:3, :3], ref_vector)
    dir_vec_b = [0,0,1]
    angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    q = quaternion_about_axis(np.deg2rad(angle), [0, 1, 0])
    m = quaternion_matrix(q)

    first_frame_pos = [new_frames[0][0], new_frames[0][1], new_frames[0][2],1.0]
    rotated_first_frame_pos = np.dot(m, first_frame_pos)[:3]
    delta = prev_frames[-1][:3] - rotated_first_frame_pos[:3]
    m[0, 3] = delta[0]
    #m[1, 3] = delta[1]
    m[2, 3] = delta[2]
    return m


def obj_spatial_error_sum_and_magnitude(s, data):
    """ Calculates the error of a low dimensional motion vector s 
    given a list of constraints.
    Note: Time parameters and time constraints will be ignored. 

    Parameters
    ---------
    * s : np.ndarray
        low dimensional motion representation
    * data : tuple
        Contains  motion_primitive, motion_primitive_constraints, prev_frames
        
    Returns
    -------
    * error: float
    """
    motion_primitive, mp_constraints, prev_frames = data
    mp_constraints.min_error = mp_constraints.evaluate(motion_primitive, s, prev_frames, use_time_parameters=False)
    #print("errors from all constraints", mp_constraints.min_error)
    return mp_constraints.min_error + np.linalg.norm(s)* 2

def get_joint_trajectory(skeleton, frames, joint_name, noise_factor):
    new_points = []
    #points = []
    prev_point = None
    for i in range(len(frames)):
        p =skeleton.nodes[joint_name].get_global_position(frames[i])
        if prev_point is None or noise_factor==0:
            new_points.append(p)
        else:
            delta = p - prev_point
            new_p = prev_point + delta + np.random.random((3)) * noise_factor *np.linalg.norm(delta)
            #new_p = p +  np.random.random((3)) * noise_factor
            new_points.append(new_p)
            #print(new_p, p, noise_factor)
        #points.append(p)
        prev_point = p
    return new_points

def reproduce_trajectory(skeleton, frames, traj, joint_name, chain_end_joint, max_iter=1):
    new_frames =  []
    constraint = KeyframeConstraint(0, joint_name, None)
    for i in range(len(frames)):
        constraint.position = traj[i]
        #print("before", skeleton.nodes[joint_name].get_global_position(frames[i]))
        f = skeleton.reach_target_position(frames[i], constraint, eps=0.01, max_iter=max_iter, chain_end_joint=chain_end_joint)
        #print(constraint.position, skeleton.nodes[joint_name].get_global_position(f))
        new_frames.append(f)
    return np.array(new_frames)


def smooth_using_moving_average(src_frames, window=4):
    """ https://www.wavemetrics.com/products/igorpro/dataanalysis/signalprocessing/smoothing.htm#MovingAverage
    """
    n_frames = len(src_frames)
    n_dims = len(src_frames[0])
    new_frames = np.zeros(src_frames.shape)
    new_frames[0,:] = src_frames[0,:]
    hw = int(window/2)
    for i in range(0, n_frames):
        for j in range(n_dims):
            start = max(0, i-hw)
            end = min(n_frames-1, i+hw)
            w = end-start
            new_frames[i, j] = np.sum(src_frames[start:end, j])/w
    return new_frames


def move_distance_to_head(skeleton, frames, traj, neck_joint, joint_name, distance_factor):
    new_points = []
    for i in range(len(frames)):
        neck_p =skeleton.nodes[neck_joint].get_global_position(frames[i])
        p  = skeleton.nodes[joint_name].get_global_position(frames[i])
        delta = p - neck_p
        delta /= np.linalg.norm(delta)
        new_p = traj[i] + delta * distance_factor
        new_points.append(new_p)
    return new_points



class MGStatePlanningSettings(object):
    def __init__(self):
        self.position_constraint_weight = 1.0
        self.direction_constraint_weight = 0.5
        self.min_target_distance = 20#50#
        self.overstepping_range = 50
        self.min_dir_distance = 50
        self.min_end_distance = 10
        self.blend_window = 20
        self.buffer_size = 100
        self.max_begin_step_length = 30
        self.max_step_length = 60
        self.add_transition_constraint = False
        self.activate_ik = True
        self.activate_grounding = True
        self.sleep_time = 0.0#0.1#5
        self.ignore_idle_rotation = False
        self.optimize_steps = True
        self.use_all_joints = False
        self.force_walk_end_targets = False
        self.look_back_range = 100
        self.end_target_blend_range = 50
        self.use_heuristic_ik = True
        self.active_joint_constraints = True
        self.use_constrained_sampling = True
        self.debug_export = False
        self.n_max_ik_iter = 10
        self.prevent_action_drift = True
        self.min_pfnn_primitive_steps = 100
        self.split_pfnn_primitives = True
        self.verbose = False
        self.min_pfnn_target_distance = 8
        self.pfnn_n_target_averaging = 1
        self.pfnn_blend_bias = 0.5
        self.walk_noise_eps = 0.1
        self.ik_resampling_factor = 1
        self.look_at_window = 40
        self.orient_spine = False
        self.restrict_number_of_constraints = True
        self.activate_joint_constraints = True
        self.enable_gesture_parameters = False
        self.gesture_ik_max_iter = 1
        self.ik_interpolation_window = 60


class MGStatePlanner(object):
    def __init__(self, state_machine, mg_state_graph, config, pfnn_wrapper=None):
        self.settings = MGStatePlanningSettings()
        self.set_config(config)
        self.algorithm_config = config["algorithm"]
        self.state_machine = state_machine
        self._graph = mg_state_graph
        self.idle_node = mg_state_graph.start_node
        self.frame_time = self._graph.skeleton.frame_time
        self.skeleton = deepcopy(self._graph.skeleton)
        self.action_constraint = None
        a = MockActionConstraints("walk", self._graph)
        self.mp_generator = MotionPrimitiveGenerator(a, config["algorithm"])
        self.mp_generator.objective = obj_spatial_error_sum_and_magnitude
        #self.mp_generator.numerical_minimizer = OptimizerBuilder(self.config["algorithm"]).build_path_following_minimizer()
        self.mp_generator.numerical_minimizer = OptimizerBuilder(config["algorithm"]).build_path_following_with_likelihood_minimizer()
        self.n_joints = len(self.skeleton.animated_joints)

        self.me = MotionEditing(self.skeleton, config["algorithm"]["inverse_kinematics_settings"])
        self.constraint_builder = ConstraintBuilder(self.skeleton, self._graph, self.settings, config["algorithm"])
        self.action_definitions = self.constraint_builder.action_definitions
        self.state_queue = MGStateQueue(self.skeleton, self._graph, self.frame_time,  self.settings)
        self.is_processing = False
        self.stop_thread = False

        self.joint_constraints = JOINT_CONSTRAINTS
        if "joint_constraints" in self.skeleton.skeleton_model:
            self.joint_constraints = self.skeleton.skeleton_model["joint_constraints"]
        if self.settings.activate_joint_constraints:
            self.me.add_constraints_to_skeleton(self.joint_constraints)

        self.use_pfnn = False
        self.pfnn_wrapper = pfnn_wrapper
        self.skeleton_cos_map = create_local_cos_map_from_skeleton_axes_with_map(self.skeleton)
        self.look_at_dir = LOOK_AT_DIR
        self.spine_look_at_dir = SPINE_LOOK_AT_DIR
        self.upper_body_indices = []
        self.upper_body_joints = []
        self.dt = 1.0/30
        if "dt" in config:
            self.dt = config["dt"]
        if self.skeleton.skeleton_model is not None:
            if "look_at_dir" in self.skeleton.skeleton_model:
                self.look_at_dir =  self.skeleton.skeleton_model["look_at_dir"]
            if "spine_look_at_dir" in self.skeleton.skeleton_model:
               self.spine_look_at_dir =  self.skeleton.skeleton_model["spine_look_at_dir"]
            for j in UPPER_BODY_JOINTS:
                if j in self.skeleton.skeleton_model["joints"]:
                    skel_j = self.skeleton.skeleton_model["joints"][j]
                    if skel_j in  self.skeleton.nodes:
                        self.upper_body_joints.append(skel_j)
                        offset = self.skeleton.nodes[skel_j].quaternion_frame_index * 4 + 3
                        self.upper_body_indices += list(range(offset,offset+4))
        print("upper body", self.upper_body_indices, self.upper_body_joints)
        self.hand_joints = []
        self.hand_chain_end_joints = dict()
        for j in HAND_JOINTS:
            if j in self.skeleton.skeleton_model["joints"]:
                skel_j = self.skeleton.skeleton_model["joints"][j]
                self.hand_joints.append(skel_j)
                if "spine_1" in self.skeleton.skeleton_model["joints"]:
                    self.hand_chain_end_joints[skel_j] = self.skeleton.skeleton_model["joints"]["spine_1"]
            
        self.collision_boundary = None
        self.hand_collision_boundary = None
        if hasattr(state_machine, "collision_boundary"):
            self.collision_boundary = state_machine.collision_boundary
        if hasattr(state_machine, "hand_collision_boundary"):
            self.hand_collision_boundary = state_machine.hand_collision_boundary
        

    def set_config(self, config):
        for key, val in config.items():
            if hasattr(self.settings, key):
                print("set", key, val)
                setattr(self.settings, key, val)
        return
        #self.mp_generator.set_algorithm_config(self.config["algorithm"])

    def generate_motion_states_from_action_sequence(self, action_sequence, start_node, start_node_type, pose_buffer, dt):
        self.dt = dt
        for idx, action_desc in enumerate(action_sequence):
            action_name = action_desc["action_name"]
            # dont transition to idle in case of multiple walk actions
            end_idle = True
            if action_name== "walk" and idx+1 < len(action_sequence) and "control_points" in action_sequence[idx+1] and len(action_sequence[idx+1]["control_points"]) > 0:
                end_idle = False
            success = self.generate_motion_states_from_action(action_name, start_node, start_node_type, pose_buffer, action_desc, end_idle)
            if not success:
                break
            state = self.state_queue.get_last_state()
            if state is not None:
                pose_buffer = state.pose_buffer
                start_node = state.node
                start_node_type = state.node_type
        
        
    def generate_motion_states_from_action(self, action_name, start_node, start_node_type, pose_buffer, action_desc, end_idle=True):
        self.is_processing = True
        success = True
        if ("control_points" in action_desc and len(action_desc["control_points"]) > 1) or ("direction" in action_desc and "n_steps" in action_desc):
            success = self.generate_locomotion(start_node, start_node_type, pose_buffer, action_desc, end_idle)

        if success and action_name != "walk" and not self.stop_thread:
            self.generate_action_motion(action_name, start_node, start_node_type, pose_buffer, action_desc)
        elif success and not self.stop_thread and end_idle:
            if len(self.state_queue) > 0:
                pose_buffer = self.state_queue[-1].pose_buffer
            self.state_queue.generate_idle_state(self.dt, pose_buffer)

        self.is_processing = False
        print("finished processing", len(self.state_queue))
        return success

    def generate_locomotion(self, start_node, start_node_type, pose_buffer, action_desc, end_idle=True):
        success = True
       
        end_direction = action_desc["end_direction"]
        step_target = None
        if "control_points" in action_desc:
            if self.settings.force_walk_end_targets:
                control_points = action_desc["control_points"]
                step_target = np.array(control_points[-1])
            control_point_end_vector = get_trajectory_end_direction(control_points)
        else:
            control_point_end_vector = action_desc["direction"]
        if self.use_pfnn:
            control_points = action_desc["control_points"]
            success, end_distance = self.generate_locomotion_from_pfnn(pose_buffer, control_points)
        else:
            if "control_points" in action_desc:
                success, end_distance = self.generate_locomotion_from_motion_primitives(start_node, start_node_type, pose_buffer, action_desc, end_idle)
            elif "direction" in action_desc and "n_steps" in action_desc:
                success, end_distance = self.generate_locomotion_from_motion_primitives_using_direction(start_node, start_node_type, pose_buffer, action_desc, end_idle)
        print("reached end", end_distance, end_idle, len(self.state_queue))
        if success and self.settings.force_walk_end_targets and len(self.state_queue) > 0 and end_idle:
            if step_target is not None and end_distance > self.settings.min_end_distance:
                self.state_queue.apply_end_pos_correction(step_target)
            print("apply end orientation", control_point_end_vector, end_direction)
            if end_direction is None:
                self.state_queue.apply_end_orientation_by_direction(control_point_end_vector)
            else:
                self.state_queue.apply_end_orientation_by_direction(end_direction)
        else:
            print("dont enforce walk end constraints")
        return success

    def generate_action_motion(self, action_name, start_node, start_node_type, pose_buffer, action_desc):
        frame_constraints = action_desc["frame_constraints"]
        body_orientation_targets = action_desc["body_orientation_targets"]
        look_at_constraints = action_desc["look_at_constraints"]
        if len(self.state_queue) > 0:
            state = self.state_queue.get_last_state()
            start_node = state.node
            start_node_type = state.node_type
            pose_buffer = state.pose_buffer
        action_start_pos = pose_buffer[-1][:3]
        action_start_q = pose_buffer[-1][3:7]
        self.constraint_builder.map_keyframe_labels_to_frame_indices(frame_constraints)
        if "n_cycles" in action_desc and action_desc["n_cycles"] >0:
            n_cycles = action_desc["n_cycles"]
            node_queue = self.generate_action_node_queue_with_cycles(action_name, start_node, start_node_type, n_cycles)
            fix_root_params = action_start_pos, action_start_q
        else:
            node_queue = self.generate_action_node_queue(action_name, start_node, start_node_type)
            fix_root_params = None
        node_constraints, node_body_orientation_targets = self.map_frame_constraints_to_nodes(node_queue, frame_constraints, body_orientation_targets, look_at_constraints)
        self.generate_action_from_motion_primitives(node_queue, node_constraints, pose_buffer, node_body_orientation_targets, fix_root_params)
        if self.settings.prevent_action_drift:
            print("prevent action drift", action_start_pos, action_start_q)
            self.state_queue.apply_end_pos_correction(action_start_pos)
            self.state_queue.apply_end_orientation_correction(action_start_q)

    def generate_action_node_queue(self, action_name, start_node, start_node_type):
        node_queue = []
        if start_node[0] == "walk" and start_node_type not in [NODE_TYPE_IDLE, NODE_TYPE_END]:
           node_queue.append((("walk", "endRightStance"), NODE_TYPE_END))

        for node_name, node_type in self.action_definitions[action_name]["node_sequence"]:
            node_queue.append(((action_name, node_name), node_type))
        
        # add extra idle state except if action is idle
        if action_name != "idle":
            node_queue.append((self.idle_node, NODE_TYPE_IDLE))
        return node_queue
    
    def map_frame_constraints_to_nodes(self, node_queue, frame_constraints, body_orientation_targets=None, look_at_constraints=False):
        node_constraints = dict()
        for frame_constraint in frame_constraints:
            frame_constraint.look_at = look_at_constraints
            key = (frame_constraint.node, frame_constraint.cycle)
            if key not in node_constraints:
                node_constraints[key] = list()
            node_constraints[key].append(frame_constraint)
            #frame_constraints.remove(frame_constraint)
        node_body_orientation_targets = dict()
        for key in node_constraints:
            if key not in node_body_orientation_targets:
                if body_orientation_targets is not None:
                    orientation_target  = list(body_orientation_targets)
                else:
                    orientation_target  = [None, None]
                if look_at_constraints and len(node_constraints) > 0:
                    orientation_target[0] = node_constraints[key][0].position
                    orientation_target[1] = node_constraints[key][0].position # assume 2d projection
                node_body_orientation_targets[key] = tuple(orientation_target)
        return node_constraints, node_body_orientation_targets

    def generate_action_node_queue_with_cycles(self, action_name, start_node, start_node_type, n_cycles):
        node_queue = []
        if start_node[0] == "walk" and start_node_type not in [NODE_TYPE_IDLE, NODE_TYPE_END]:
           node_queue.append((("walk", "endRightStance"), NODE_TYPE_END))


        node_name, node_type = self.action_definitions[action_name]["cycle_start"]
        node_queue.append(((action_name, node_name), node_type))

        for n in range(n_cycles-2):
            node_name, node_type = self.action_definitions[action_name]["cycle_node"]
            node_queue.append(((action_name, node_name), node_type))
        if n_cycles > 1:
            node_name, node_type = self.action_definitions[action_name]["cycle_end"]
            node_queue.append(((action_name, node_name), node_type))
        
        # add extra idle state except if action is idle
        if action_name != "idle":
            node_queue.append((self.idle_node, NODE_TYPE_IDLE))
        return node_queue

    def generate_action_from_motion_primitives(self, node_queue, node_constraints, pose_buffer, node_body_orientation_targets, fix_root_params=None):
        """ Generate a motion primitive sequence based on action constraints """
        #print("generate action", node_constraints.keys(), look_at_target)
        cycle_count = 0
        while len(node_queue) > 0 and not self.stop_thread:
            current_node, node_type = node_queue[0]
            constraints = []
            key = (current_node, cycle_count)
            if key in node_constraints:
                constraints = node_constraints[key]
            else:
                print("ignore",key)
            body_orientation_targets = None
            if key in node_body_orientation_targets:
                body_orientation_targets = node_body_orientation_targets[key]
            new_frames, events, hold_frames = self.generate_constrained_frames(current_node, constraints, pose_buffer, body_orientation_targets)

            new_entry = self.state_queue.create_state_queue_entry(current_node, node_type, self.dt, pose_buffer, new_frames,
                                                      events, hold_frames, export=False)

            self.state_queue.append_state_to_queue(new_entry)
            if fix_root_params is not None:
                start_pos, start_q = fix_root_params
                self.state_queue.apply_end_pos_correction(start_pos)
                self.state_queue.apply_end_orientation_correction(start_q)
            time.sleep(self.settings.sleep_time)
            node_queue = node_queue[1:]
            action = current_node[0]
            if action in self.action_definitions:
                if "cycle_nodes" in self.action_definitions[action]:
                    if current_node[1] in self.action_definitions[action]["cycle_nodes"]:
                        cycle_count+=1


    def generate_constrained_frames(self, current_node, constraints, pose_buffer, body_orientation_targets=None):
        #aligning_transform = self.get_aligning_transform(current_node, pose_buffer)
        #print("aligining constraint transform", aligning_transform)

        # prepare constraints
        if pose_buffer is None:
            aligning_transform = np.eye(4)
        else:
            aligning_transform = self.get_aligning_transform(current_node, pose_buffer)
        mp_constraints, apply_ik = self.constraint_builder.generate_motion_primitive_constraints(current_node, aligning_transform, constraints, pose_buffer)


        new_frames = self.generate_constrained_motion_primitive(current_node, mp_constraints, pose_buffer)

        # self.me.apply_joint_constraints(new_frames, 0, len(new_frames))
        if apply_ik:
            if self.settings.debug_export:
                self.export_frames(new_frames, current_node[1], "before_ik.bvh")
            new_frames = self.apply_ik_constraints(new_frames, current_node, constraints, body_orientation_targets, pose_buffer)
            if self.settings.debug_export:
                self.export_frames(new_frames, current_node[1],  "after_ik.bvh")
        elif body_orientation_targets is not None and len(body_orientation_targets) == 2 and body_orientation_targets[0] is not None:
            new_frames = self.apply_body_orientation(new_frames, current_node, body_orientation_targets[0], body_orientation_targets[1], pose_buffer)
        else:
            print("do not apply ik")


        hold_frames = set()
        events = dict()
        for c in constraints:
            if c.keyframe not in events:
                events[c.keyframe] = []
            events[c.keyframe] += c.keyframe_events
            if c.hold_frame:
                hold_frames.add(c.keyframe)
        return new_frames, events, hold_frames

    def get_next_node_type(self, current_node_type, step_distance, end_idle=True):
        """ Returns next node type taking step_distance into account """
        next_node_type = current_node_type
        if current_node_type == NODE_TYPE_START:
            next_node_type = NODE_TYPE_STANDARD
        elif current_node_type == NODE_TYPE_STANDARD:
            if step_distance > 0 or not end_idle: # keep walking 
                next_node_type = NODE_TYPE_STANDARD
            else:
                print("SET END NODE", end_idle, step_distance)
                print("SET END NODE", end_idle, step_distance)
                next_node_type = NODE_TYPE_END
        elif current_node_type == NODE_TYPE_END:
            if step_distance > 0:
                next_node_type = NODE_TYPE_START
            else:
                next_node_type = NODE_TYPE_IDLE
        elif current_node_type == NODE_TYPE_IDLE:
            if step_distance > 0:
                next_node_type = NODE_TYPE_START
            else:
                next_node_type = NODE_TYPE_IDLE
        return next_node_type


    def generate_locomotion_from_motion_primitives(self, start_node, start_node_type, pose_buffer, action_desc, end_idle=True):
        """generate a sequence of walk motion primitives based on a sequence of target points """
        success = True
        control_points = action_desc["control_points"]
        end_direction = action_desc["end_direction"]
        body_orientation_targets = action_desc["body_orientation_targets"]
        upper_body_gesture = action_desc["upper_body_gesture"]
        velocity_factor = action_desc["velocity_factor"]

        upper_body_state = None
        if upper_body_gesture is not None:
            upper_body_state = self.generate_upper_body_state(upper_body_gesture)
        path_state = dict()
        current_node = start_node
        node_type = start_node_type
        path_state["distance"] = np.inf
        path_state["prev_direction_vector"] = None
        path_state["prev_distance"] = np.inf
        path_state["prev_target"] = control_points[0]
        path_state["current_position"] = np.array(pose_buffer[-1][:3])
        print("start processing", control_points)
        add_noise = False
        while len(control_points) > 0 and not self.stop_thread:
            direction_vector, step_distance, path_state, control_points = self.get_direction_from_control_points(control_points, node_type, path_state, pose_buffer, end_direction, add_noise)
            if direction_vector is None:
                print("target is none", current_node,  len(control_points) )
                raise Exception("Error: direction_vector is None",current_node)
                
            if self.collision_boundary is not None and step_distance > 0:
                points = [np.array(pose_buffer[-1][:3])]
                points.append(points[0]+step_distance*direction_vector)
                if self.collision_boundary.check_trajectory(points, 0.0001):
                    print("stop due to collision")
                    control_points = list()
                    step_distance = 0
                    success = False
                    self.state_queue.pop_last_state()
                    last_state = self.state_queue.get_last_state()
                    if last_state is not None:
                        pose_buffer = self.state_queue.get_last_state().pose_buffer
            elif step_distance > 0:
                print("no collision boundary defined")
            new_frames, current_node, node_type = self.generate_locomotion_step(current_node, node_type, direction_vector, step_distance, pose_buffer, body_orientation_targets, upper_body_state, velocity_factor, end_idle)
            self.state_queue.add_state_queue_entry(current_node, node_type, new_frames, pose_buffer, self.dt)
            path_state["prev_direction_vector"] = direction_vector
            path_state["prev_distance"] = path_state["distance"] 
            time.sleep(self.settings.sleep_time)
            print("generated state")
        return success, np.linalg.norm(path_state["current_position"]  - path_state["prev_target"])

    def generate_locomotion_from_motion_primitives_using_direction(self, start_node, start_node_type, pose_buffer, action_desc, end_idle=True):
        """generate a sequence of walk motion primitives based on a direction and a number of steps """
        success = True
        direction = action_desc["direction"]
        n_remaining_steps = action_desc["n_steps"]
        body_orientation_targets = action_desc["body_orientation_targets"]
        upper_body_gesture = action_desc["upper_body_gesture"]
        velocity_factor = action_desc["velocity_factor"]
        step_distance = 80
        if "step_distance" in action_desc:
            step_distance = action_desc["step_distance"]

        upper_body_state = None
        if upper_body_gesture is not None:
            upper_body_state = self.generate_upper_body_state(upper_body_gesture)
        current_node = start_node
        node_type = start_node_type
        print("start processing direction", direction)
        while n_remaining_steps > 0 and not self.stop_thread:
            direction_vector = direction
            if n_remaining_steps == 1:
                step_distance = 0
            if self.collision_boundary is not None and step_distance > 0:
                points = [np.array(pose_buffer[-1][:3])]
                points.append(points[0]+step_distance*direction_vector)
                if self.collision_boundary.check_trajectory(points, 0.0001):
                    print("stop due to collision")
                    step_distance = 0
                    success = False
                    self.state_queue.pop_last_state()
                    last_state = self.state_queue.get_last_state()
                    if last_state is not None:
                        pose_buffer = self.state_queue.get_last_state().pose_buffer
            elif step_distance > 0:
                print("no collision boundary defined")
            new_frames, current_node, node_type = self.generate_locomotion_step(current_node, node_type, direction_vector, step_distance, pose_buffer, body_orientation_targets, upper_body_state, velocity_factor, end_idle)
            self.state_queue.add_state_queue_entry(current_node, node_type, new_frames, pose_buffer, self.dt)
            n_remaining_steps-=1
            time.sleep(self.settings.sleep_time)
            print("generated state")
        return success, 0


    def generate_upper_body_state(self, upper_body_gesture):
        action_name = upper_body_gesture["name"]
        velocity_factor = 1.0
        noise_factor = 0.0
        distance_factor = 0.0
        if "velocityFactor" in upper_body_gesture:
            velocity_factor = upper_body_gesture["velocityFactor"]
            velocity_factor = max(velocity_factor,0)
        if "noiseFactor" in upper_body_gesture:
            noise_factor = upper_body_gesture["noiseFactor"]
        if "distanceToHeadFactor" in upper_body_gesture:
            distance_factor = upper_body_gesture["distanceToHeadFactor"]
        upper_body_state = None
        if action_name in self.action_definitions:
            node_seq = []
            for node_name, node_type in self.action_definitions[action_name]["node_sequence"]:
                node_seq.append((action_name, node_name))
            
            upper_body_frames = []
            for upper_body_node in node_seq:
                sample = self._graph.nodes[upper_body_node].sample(False)
                #upper_body_frames += self.me.resample_motion(sample.get_motion_vector(), 1/velocity_factor).tolist()
                upper_body_frames += sample.get_motion_vector(velocity_factor).tolist()
    
            if len(upper_body_frames) > 0 and self.settings.use_all_joints:
                upper_body_frames = np.array(upper_body_frames)
                upper_body_frames = self.complete_frames(upper_body_node, upper_body_frames)

            if self.settings.enable_gesture_parameters and (noise_factor > 0 or distance_factor > 0):
                #noise_factor = 0
                joint_name = self.skeleton.skeleton_model["joints"]["right_wrist"]
                chain_end_joint = self.skeleton.skeleton_model["joints"]["right_shoulder"]
                neck_joint = self.skeleton.skeleton_model["joints"]["neck"]
                # get trajectory of hand and modify trajectory based on noise and distance to head
                traj = get_joint_trajectory(self.skeleton, upper_body_frames, joint_name, noise_factor)
                if distance_factor > 0:
                    traj = move_distance_to_head(self.skeleton,  upper_body_frames, traj, neck_joint, joint_name, distance_factor)
               
                #traj = add_noise_to_trajectory(traj, noise_factor)
                # reproduce using ik
                upper_body_frames = reproduce_trajectory(self.skeleton, upper_body_frames, traj, joint_name, chain_end_joint, self.settings.gesture_ik_max_iter)
                #n_window_size = 10
                #upper_body_frames = smooth_using_moving_average(upper_body_frames, n_window_size)
            
            upper_body_state = dict()
            upper_body_state["frames"] = np.array(upper_body_frames)
            upper_body_state["frame_idx"] = 0
        return upper_body_state



    def get_direction_from_control_points(self, control_points, node_type, path_state, pose_buffer, end_direction=None, add_noise=False):
        """ get direction vector from control points and handle overstepping """
        direction_vector = None
        path_state["current_position"] = np.array(pose_buffer[-1][:3])
        found_target = False
        in_target_range = False
        while not found_target and len(control_points) > 0 and not self.stop_thread:
            direction_vector = control_points[0] - path_state["current_position"]
            distance = np.linalg.norm(direction_vector)

            if len(control_points) == 1 and distance < self.settings.overstepping_range:
                in_target_range = True

            if distance > path_state["prev_distance"] and len(control_points) == 1 and in_target_range:
                print("abort due to overstepping", distance,path_state["prev_distance"])
                self.state_queue.pop_last_state()
                pose_buffer = self.state_queue.get_last_state().pose_buffer
                direction_vector /= distance
                direction_vector = path_state["prev_direction_vector"]
                distance = 0
                found_target = False
                prev_target = control_points.pop(0)
            elif distance > self.settings.min_target_distance:
                # point has not yet been reached
                if len(pose_buffer) > 0: # make sure we did not  overstep the target
                    points = [f[:3] for f in pose_buffer[-self.settings.look_back_range:]]
                    min_distance = min([ np.linalg.norm(p - control_points[0]) for p in points])
                    print("overstepped target",min_distance, distance)
                else:
                    min_distance = distance
                if min_distance > self.settings.min_target_distance:
                    direction_vector /= distance
                    found_target = True
                else:
                    # point has already been reached
                    print("reached point with distance", min_distance)
                    direction_vector = None
                    path_state["distance"] = np.inf
                    path_state["prev_target"] = control_points.pop(0)
            else:
                # point has already been reached
                print("reached point with distance", distance)
                direction_vector = None
                path_state["distance"] = np.inf
                path_state["prev_target"] = control_points.pop(0)

        if direction_vector is None:
            print("abort due to min distance")
            distance = 0
            direction_vector = path_state["prev_direction_vector"]
        if end_direction is not None and len(control_points) == 1 and distance< self.settings.min_dir_distance:
            direction_vector = end_direction

        if node_type == NODE_TYPE_IDLE:
            step_distance = min(distance, self.settings.max_begin_step_length)
        else:
            step_distance = min(distance, self.settings.max_step_length)
        step_distance = max(0, step_distance)

        if direction_vector is not None and add_noise:
            dir_noise = np.random.rand(3) * self.settings.walk_noise_eps
            dir_noise[1] = 0
            direction_vector += dir_noise
        return direction_vector, step_distance, path_state, control_points

    
    def generate_locomotion_step(self, current_node, current_node_type, direction_vector, step_distance, pose_buffer, body_orientation_targets=None, upper_body_state=None, velocity_factor=1.0, end_idle=True):
        next_node_type = self.get_next_node_type(current_node_type, step_distance, end_idle)
        options = self.get_node_options(current_node, next_node_type)
        if len(options) ==0:
            print(current_node, options, next_node_type)
            print("Error no transitions were defined")
            print("Reset to idle")
            current_node = self.idle_node 
            next_node_type = NODE_TYPE_IDLE
            options = self.get_node_options(current_node, next_node_type)
        #apply larger step distance to keep walking
        if not end_idle:
            step_distance = self.settings.min_dir_distance
        current_node, s, mp_constraints = self.select_best_option(current_node, options, direction_vector, step_distance, pose_buffer)
        node_type = next_node_type
        if self.settings.optimize_steps:
            s = self._optimize_step(current_node, s, mp_constraints, pose_buffer)

        spline = self._graph.nodes[current_node].back_project(s, use_time_parameters=False)
        new_frames = spline.get_motion_vector(velocity_factor)

        if self.settings.use_all_joints:
            new_frames = self.complete_frames(current_node, new_frames)
            
        if upper_body_state is not None:
            upper_body_state, new_frames = self.combine_frames_with_other_motion_primitive(new_frames, upper_body_state)

        if body_orientation_targets is not None and len(body_orientation_targets)==2:
            if body_orientation_targets[0] is not None or body_orientation_targets[1] is not None:
                # recalculate aligning transform
                m = get_node_aligning_2d_transform(self.skeleton, self.skeleton.aligning_root_node, pose_buffer, new_frames)
                #print("aligining constraint transform", m)        
                aligning_transform = np.linalg.inv(m)
                look_at_target = body_orientation_targets[0]
                if look_at_target is not None:
                    look_at_target = np.dot(aligning_transform, [look_at_target[0], look_at_target[1], look_at_target[2], 1])[:3]
                spine_target = body_orientation_targets[1]
                if spine_target is not None:
                    spine_target = np.dot(aligning_transform, [spine_target[0], spine_target[1], spine_target[2], 1])[:3]
                look_start, look_end = 0, len(new_frames)
                #print("set look at range to", look_start, look_end, look_at_target)
                new_frames = self.me.edit_motion_to_look_at_target(new_frames, look_at_target, spine_target, look_start, look_end, self.settings.orient_spine, self.look_at_dir, self.spine_look_at_dir)
                new_frames = self.me.apply_joint_constraints(new_frames, look_start, look_end)
        return new_frames, current_node, node_type

    def combine_frames_with_other_motion_primitive(self, frames, upper_body_state):
        n_frames = len(frames)
        upper_body_frames = upper_body_state["frames"]
        upper_body_frame_idx = upper_body_state["frame_idx"]
        n_upper_body_frames = len(upper_body_frames)
        for frame_idx in range(n_frames):
            frames[frame_idx][self.upper_body_indices] = upper_body_frames[upper_body_frame_idx][self.upper_body_indices]
            upper_body_frame_idx+=1
            upper_body_frame_idx %= n_upper_body_frames
        upper_body_state["frame_idx"] = upper_body_frame_idx
        return upper_body_state, frames

    def get_pose_orientation(self, frame, ref_vector):
        node_name = self.skeleton.aligning_root_node
        m_a = self.skeleton.nodes[node_name].get_global_matrix(frame)
        dir_vec = get_orientation_vector_from_matrix(m_a[:3, :3], ref_vector)
        angle = get_rotation_angle(dir_vec, ref_vector)
        return [dir_vec[0],0,dir_vec[1]], np.deg2rad(angle)

    def rotate_point(self, point, angle):
        r = np.radians(angle)
        s = np.sin(r)
        c = np.cos(r)
        point = np.array(point, float)
        point[0] = c * point[0] - s * point[2]
        point[2] = s * point[0] + c * point[2]
        return point

    def get_avg_direction(self, pos, points):
        n_points = min(self.settings.pfnn_n_target_averaging, len(points))
        avg_point = np.zeros(3)
        for idx in range(n_points):
            avg_point += points[idx]
        avg_point /= n_points
        target_dir = avg_point - pos
        target_dir = np.array([target_dir[0], 0, target_dir[2]])
        return target_dir

    def generate_locomotion_from_pfnn(self, frame_buffer, control_points):
        """generate a walk motion based on a sequence of target points
        frame_buffer: a list of frames containing pose parameters of the previous frames
        control_points: a list of positions
        """
        success = True
        self.pfnn_wrapper.controller.traj.blend_bias = self.settings.pfnn_blend_bias
        aligning_transform = self.get_node_aligning_start_transform_pfnn(frame_buffer)
        aligning_transform_inv = np.linalg.inv(aligning_transform)
        scale_factor = 1.0 / 10
        local_control_points = []
        for p in control_points:
            local_p = np.dot(aligning_transform_inv, [p[0],0, p[2], 1])[:3]
            local_control_points.append(local_p * scale_factor)

        current_position = np.array([0, 0, 0])
        local_control_points = [current_position] + local_control_points + [local_control_points[-1]]

        print("stretch control points", len(local_control_points))
        local_control_points = stretch_points(local_control_points, 0.1)
        print("generate motion", len(control_points), len(local_control_points))
        start_position = np.array(frame_buffer[-1][:3])

        self.pfnn_wrapper.reset(current_position, 0, np.array([0, 0, 1]))

        n_primitives = 0
        new_frames = []
        while len(local_control_points) > 0 and not self.stop_thread:
            # get direction vector from control points
            target_dir = self.get_avg_direction(current_position, local_control_points)
            distance = np.linalg.norm(target_dir)
            target_dir /= distance
            if distance < self.settings.min_pfnn_target_distance:
                # point has been reached
                local_control_points.pop(0)
                print(len(local_control_points), "points remaining")
                continue
            frame = self.pfnn_wrapper.get_next_frame(target_dir, self.settings.verbose)
            current_position = frame[:3]
            #print("step_controller", current_position, target_pos, self.pfnn_wrapper.phase, distance)
            new_frames.append(frame)

            if len(new_frames) > self.settings.min_pfnn_primitive_steps and self.settings.split_pfnn_primitives:
                # convert the result of the pfnn into a state queue entry
                scaled_frames = np.array(new_frames)
                if len(new_frames) > 0:
                    scaled_frames[:, :3] *= 1 / scale_factor
                if n_primitives > 0:
                    current_node = ("walk", "right_step")
                    node_type = NODE_TYPE_STANDARD
                    apply_smoothing = False
                else:
                    current_node = ("walk", "right_end_step")
                    node_type = NODE_TYPE_END
                    apply_smoothing = True
                new_state = self.state_queue.build_pfnn_state(scaled_frames, frame_buffer, aligning_transform, apply_smoothing)
                self.state_queue.add_state_queue_entry(current_node, node_type, scaled_frames, frame_buffer, self.dt, new_state=new_state)
                new_frames = []
                frame_buffer = np.array(self.state_queue.get_last_state().state.mv.frames).tolist()
                n_primitives+=1

        # convert the result of the pfnn into a state queue entry
        new_frames = np.array(new_frames)
        if len(new_frames) >0:
            current_node = ("walk", "right_end_step")
            node_type = NODE_TYPE_END
            new_frames[:,:3] *= 1/scale_factor
            #new_frames[:,:3] += start_position
            apply_smoothing = True
            if n_primitives > 0:
                apply_smoothing = False
            new_state = self.state_queue.build_pfnn_state(new_frames, frame_buffer, aligning_transform, apply_smoothing=apply_smoothing)
            if new_state.mv.n_frames > 0:
                self.state_queue.add_state_queue_entry(current_node, node_type, new_frames, frame_buffer, self.dt, new_state=new_state)
        return success, np.linalg.norm(start_position - control_points[-1])


    def _optimize_step(self, current_node, s, mp_constraints, pose_buffer):
        mp_constraints.constraints = [c for c in mp_constraints.constraints if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION]
        if len(mp_constraints.constraints) > 0:
            data = (self._graph.nodes[current_node], mp_constraints, pose_buffer, 1, 1, 1.0)
            self.mp_generator.numerical_minimizer.set_objective_function_parameters(data)
            return self.mp_generator.numerical_minimizer.run(s)
        else:
            return s

    def get_node_options(self, current_node, next_node_type):
        edges = self._graph.nodes[current_node].outgoing_edges
        options = [edge_key for edge_key in list(edges.keys()) if edges[edge_key].transition_type == next_node_type]
        return options

    def select_best_option(self, current_node, options, direction_vector, step_distance, pose_buffer):
        errors = np.empty(len(options))
        s_vectors = []
        constraints = []
        index = 0
        for node_name in options:
            # print "option", node_name
            m = self.get_aligning_transform(node_name, pose_buffer)
            # find generate frames based on constraints
            self.mp_generator.action_name = current_node[0]
            mp_constraints = self.constraint_builder.generate_walk_constraints(node_name, m, direction_vector, step_distance,
                                                            pose_buffer)
            s = self.mp_generator.generate_constrained_sample(self._graph.nodes[node_name], mp_constraints)
            errors[index] = mp_constraints.min_error
            s_vectors.append(s)
            constraints.append(mp_constraints)
            index += 1
        min_idx = np.argmin(errors)
        next_node = options[min_idx]
        s = s_vectors[min_idx]
        c = constraints[min_idx]
        print("options", current_node, min_idx, next_node, errors, options)
        return next_node, s, c

    def get_aligning_transform(self, current_node, pose_buffer):
        """ uses a random sample of the morphable model to find an aligning transformation to bring constraints into the local coordinate system"""
        sample = self._graph.nodes[current_node].sample(False)
        frames = sample.get_motion_vector()
        m = get_node_aligning_2d_transform(self.skeleton, self.skeleton.aligning_root_node, pose_buffer, frames)
        m = np.linalg.inv(m)
        return m

    def get_node_aligning_start_transform_pfnn(self, pose_buffer):
        """ Calculate transform to align two frame sequences.
        """
        self.pfnn_wrapper.reset(np.array([0, 0, 0]), 0, np.array([0, 0, 1]))
        frames = self.pfnn_wrapper.get_random_frames(10)
        m = get_node_aligning_2d_transform_pfnn(self.skeleton, self.skeleton.aligning_root_node, pose_buffer, frames)
        return m

    def get_alining_transform_from_frames(self, frames, pose_buffer):
        """ uses a random sample of the morphable model to find an aligning transformation to bring constraints into the local coordinate system"""
        m = get_node_aligning_2d_transform(self.skeleton, self.skeleton.aligning_root_node, pose_buffer, frames)
        return np.linalg.inv(m)

    def complete_frames(self,current_node, new_frames):
        animated_joints = self._graph.nodes[current_node].get_animated_joints()
        #print("animated", animated_joints)
        if len(animated_joints) == 0:
            animated_joints = ANIMATED_JOINTS_CUSTOM
        #print("animated joints", animated_joints)
        new_full_frames = np.zeros((len(new_frames), self.skeleton.reference_frame_length))
        for idx, reduced_frame in enumerate(new_frames):
            new_full_frames[idx] = self.skeleton.add_fixed_joint_parameters_to_other_frame(reduced_frame,
                                                                                     animated_joints)
        return new_full_frames

    def generate_constrained_motion_primitive(self, current_node, mp_constraints, pose_buffer=None):

        if self.settings.use_constrained_sampling and len(mp_constraints.constraints) > 0: #and current_node[1] != "placeRaceway":
            if len(mp_constraints.constraints) > 2 and self.settings.restrict_number_of_constraints:
                mp_constraints.constraints = [mp_constraints.constraints[0]]
                print("throw away constraints")
            self.mp_generator.action_name = current_node[0]
            s = self.mp_generator.generate_constrained_sample(self._graph.nodes[current_node], mp_constraints)
        else:
            print("get random sample")
            s = self._graph.nodes[current_node].sample_low_dimensional_vector()#[0]
            #new_frames = self._graph.nodes[current_node].sample(use_time=False).get_motion_vector()
        #print(s)
        spline = self._graph.nodes[current_node].back_project(s, use_time_parameters=False)
        new_frames = spline.get_motion_vector()
        if self.settings.use_all_joints:
            new_frames = self.complete_frames(current_node, new_frames)
       
        return new_frames

    def apply_ik_constraints(self, frames, node, constraints, body_orientation_targets, pose_buffer):
        print("apply ik constraints")

        # recalculate aligning transform
        m = get_node_aligning_2d_transform(self.skeleton, self.skeleton.aligning_root_node, pose_buffer, frames)

        aligning_transform = np.linalg.inv(m)
        mp_constraints, apply_ik = self.constraint_builder.generate_motion_primitive_constraints(node, None, constraints, pose_buffer)
        if self.settings.ik_resampling_factor < 1 and self.settings.ik_resampling_factor > 0:
            # resample frames
            frames = self.me.resample_motion(frames, self.settings.ik_resampling_factor)
            for c in mp_constraints.constraints:
                c.canonical_keyframe = int(c.canonical_keyframe*self.settings.ik_resampling_factor)
                if c.canonical_end_keyframe is not None:
                    c.canonical_end_keyframe = int(c.canonical_end_keyframe*self.settings.ik_resampling_factor)

        frames = transform_quaternion_frames(frames, m)

        if body_orientation_targets is not None and len(body_orientation_targets) == 2 and body_orientation_targets[0] is not None:

            look_start, look_end = self.find_look_at_frame_range(mp_constraints.constraints, len(frames), self.settings.look_at_window)
            print("set look at range to", look_start, look_end)
            head_target = body_orientation_targets[0]
            spine_target = body_orientation_targets[1]
            if self.settings.debug_export:
                self.export_frames(frames, node[1],  "before_look.bvh")
            frames = self.me.edit_motion_to_look_at_target(frames, head_target, spine_target, look_start, look_end, self.settings.orient_spine, self.look_at_dir, self.spine_look_at_dir)
            if self.settings.debug_export:
                self.export_frames(frames, node[1],  "after_look.bvh")

        mp_constraints.action_name = node[0]
        time_function = None  # TODO get time_function
        if self.settings.use_heuristic_ik:
            frames = self.apply_heuristic_ik(frames, node, mp_constraints, time_function)
        else:
            frames = self.apply_legacy_ik(frames, node, mp_constraints, time_function)
        action_name = node[0]

        frames = self.handle_collision2(frames, action_name, self.hand_joints, self.hand_chain_end_joints)


        if self.settings.ik_resampling_factor < 1 and self.settings.ik_resampling_factor > 0:
            frames = self.me.resample_motion(frames, 1/self.settings.ik_resampling_factor)
        return transform_quaternion_frames(frames, aligning_transform)

    def export_frames(self, frames, prefix, suffix):
        time_stamp = str(datetime.now().strftime("%d%m%y_%H%M%S"))
        mv = MotionVector()
        mv.frames = frames
        mv.n_frames = len(frames)
        mv.export(self.skeleton, "data" + os.sep + prefix+time_stamp+suffix)


    def apply_body_orientation(self, frames, node, look_at_target, spine_target, pose_buffer):
        print("apply look at constraint")
        m = get_node_aligning_2d_transform(self.skeleton, self.skeleton.aligning_root_node, pose_buffer, frames)
        aligning_transform = np.linalg.inv(m)
        frames = transform_quaternion_frames(frames, m)
        look_start = 0
        look_end = len(frames)-1
        frames = self.me.edit_motion_to_look_at_target(frames, look_at_target, spine_target, look_start, look_end, self.settings.orient_spine, self.look_at_dir, self.spine_look_at_dir)
        return transform_quaternion_frames(frames, aligning_transform)

    def apply_legacy_ik(self, frames, node, mp_constraints, time_function):
        ik_settings = self.algorithm_config["inverse_kinematics_settings"]
        ik_constraints = mp_constraints.convert_to_ik_constraints(self._graph, 0, time_function=time_function,
                                                                    version=2)
        frames = self.me.edit_motion_using_displacement_map(frames, ik_constraints, influence_range=40, plot=False)
        return frames

    def get_root_joint(self, action_name):
        root_joint = None
        if "ik_root_joint" in self.action_definitions[action_name]:
            std_root_joint = self.action_definitions[action_name]["ik_root_joint"]
            if std_root_joint in self.skeleton.skeleton_model["joints"] and self.skeleton.skeleton_model["joints"][std_root_joint] is not None:
                root_joint = self.skeleton.skeleton_model["joints"][std_root_joint]
                print("set ik root to ", root_joint)
        return root_joint

    def apply_heuristic_ik(self, frames, node, mp_constraints, time_function):
        print("use heuristic")
        builder = IKConstraintsBuilder2(self.skeleton)
        ik_constraints = builder.convert_to_ik_constraints_with_relative(frames, mp_constraints.constraints,
                                                                        0, time_function, constrain_orientation=True)
        action_name = node[0]
        root_joint = self.get_root_joint(action_name)
        max_ik_iter = self.settings.n_max_ik_iter
        if len(ik_constraints) > 1:
            frames = self.me.edit_motion_using_ccd(frames, ik_constraints, n_max_iter=max_ik_iter, root_joint=root_joint)
        else:
            frames = self.me.edit_motion_using_displacement_map_and_ccd(frames, ik_constraints, n_max_iter=max_ik_iter, root_joint=root_joint, influence_range=self.settings.ik_interpolation_window)

        frames = self.me.set_global_joint_orientations(frames, mp_constraints.constraints)
        return frames

    def find_look_at_frame_range(self, constraints, n_frames, window_size):
        if len(constraints) >1:
            start_idx = n_frames
            end_idx = 0
            for c in constraints:
                if start_idx > c.canonical_keyframe:
                    start_idx = c.canonical_keyframe
                if end_idx < c.canonical_keyframe:
                    end_idx = c.canonical_keyframe

            start_idx = max(0, start_idx-window_size)
            end_idx = min(n_frames, end_idx+ window_size)
        else:
            start_idx = 0
            end_idx = n_frames
        return start_idx, end_idx

    def handle_collision(self, frames, action_name, dt=0.0001):
        if self.hand_collision_boundary is not None:
            joint_name = self.hand_collision_boundary.joint_name
            if joint_name in self.skeleton.nodes:
                points = [self.skeleton.nodes[joint_name].get_global_position(frame) for frame in frames]
                frame_idx, pos, normal = self.hand_collision_boundary.check_trajectory(points,dt)
                print("found collision ", frame_idx)
                if frame_idx > 0:
                    max_ik_iter = self.settings.n_max_ik_iter
                    root_joint = self.get_root_joint(action_name)
                    ik_constraints = collections.OrderedDict()
                    #root_pos = frames[frame_idx, :3]
                    #delta = root_pos - pos
                    #delta /= np.linalg.norm(delta)
                    pos = np.array(pos)
                    print("##apply collision avoidance constraint", pos, normal)
                    pos -= np.array(normal) * 20
                    ik_constraints[frame_idx] = {joint_name: KeyframeConstraint(frame_idx, joint_name, pos, None, False, None)}
                    before = np.array(frames)
                    frames = self.me.edit_motion_using_ccd(frames, ik_constraints, n_max_iter=max_ik_iter, root_joint=root_joint)
                    print("change", np.linalg.norm(before-frames))
        return frames
        
    def handle_collision2(self, frames, action_name, joint_names, chain_end_joints, dt=0.0001):
        new_frames = np.array(frames)
        if self.hand_collision_boundary is not None:
            _has_collision = False
            ik_constraints = collections.OrderedDict() 
            for joint_name in joint_names:
                self.hand_collision_boundary.joint_name = joint_name
                if joint_name in self.skeleton.nodes:
                    matrices = [self.skeleton.nodes[joint_name].get_global_matrix(frame) for frame in frames]
                    joint_positions = [m[:3,3] for m in matrices]
                    orientations = [quaternion_from_matrix(m) for m in matrices]
                    has_collision, delta_trajectory = self.hand_collision_boundary.get_delta_trajectory(joint_positions,dt)
                    if has_collision:
                        _has_collision = True
                        max_ik_iter = self.settings.n_max_ik_iter
                        root_joint = self.get_root_joint(action_name)
                        root_joint = None
                        for frame_idx, d in delta_trajectory.items():
                            if frame_idx < len(frames):
                                orientation = None#orientations[frame_idx]
                                c = KeyframeConstraint(frame_idx, joint_name, joint_positions[frame_idx]+d, orientation, False, None)
                                if frame_idx not in ik_constraints:
                                    ik_constraints[frame_idx] = dict()
                                ik_constraints[frame_idx][joint_name] = c
            if _has_collision:
                #frames = self.me.edit_motion_using_displacement_map_and_ccd(frames, ik_constraints, n_max_iter=5, root_joint=root_joint)
                for frame_idx in ik_constraints:
                    new_frames[frame_idx] = self.skeleton.reach_target_positions(frames[frame_idx], list(ik_constraints[frame_idx].values()) , chain_end_joints, n_max_iter=1, verbose=False)
                print("change", np.linalg.norm(frames-new_frames))
        return new_frames


