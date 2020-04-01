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
import os
import collections
from transformations import quaternion_slerp
from anim_utils.animation_data.skeleton_models import *
from anim_utils.motion_editing.utils import generate_root_constraint_for_one_foot, smooth_root_translation_at_end, smooth_root_translation_at_start
from anim_utils.motion_editing.footplant_constraint_generator import guess_ground_height
from anim_utils.motion_editing.motion_grounding import MotionGroundingConstraint, generate_ankle_constraint_from_toe, create_ankle_constraint_from_toe_and_heel
from anim_utils.motion_editing.analytical_inverse_kinematics import AnalyticalLimbIK

LEFT_FOOT = "LeftFoot"
RIGHT_FOOT = "RightFoot"
RIGHT_TOE = "RightToeBase"
LEFT_TOE = "LeftToeBase"
RIGHT_KNEE = "RightLeg"
LEFT_KNEE = "LeftLeg"
RIGHT_HIP = "RightUpLeg"
LEFT_HIP = "LeftUpLeg"
LEFT_HEEL = "LeftHeel"
RIGHT_HEEL = "RightHeel"

MP_CONFIGURATIONS = collections.OrderedDict()

MP_CONFIGURATIONS["leftStance"] = dict()
MP_CONFIGURATIONS["leftStance"]["start_stance_foot"] = "right"
MP_CONFIGURATIONS["leftStance"]["stance_foot"] = "right"
MP_CONFIGURATIONS["leftStance"]["swing_foot"] = "left"
MP_CONFIGURATIONS["leftStance"]["end_stance_foot"] = "left"
MP_CONFIGURATIONS["leftStance"]["stance_mode"] = "toe"
MP_CONFIGURATIONS["leftStance"]["start_window_size"] = 10
MP_CONFIGURATIONS["leftStance"]["end_window_size"] = 10

MP_CONFIGURATIONS["rightStance"] = dict()
MP_CONFIGURATIONS["rightStance"]["start_stance_foot"] = "left"
MP_CONFIGURATIONS["rightStance"]["stance_foot"] = "left"
MP_CONFIGURATIONS["rightStance"]["swing_foot"] = "right"
MP_CONFIGURATIONS["rightStance"]["end_stance_foot"] = "right"
MP_CONFIGURATIONS["rightStance"]["stance_mode"] = "toe"
MP_CONFIGURATIONS["rightStance"]["start_window_size"] = 10
MP_CONFIGURATIONS["rightStance"]["end_window_size"] = 10

MP_CONFIGURATIONS["beginLeftStance"] = dict()
MP_CONFIGURATIONS["beginLeftStance"]["start_stance_foot"] = "both"
MP_CONFIGURATIONS["beginLeftStance"]["stance_foot"] = "right"
MP_CONFIGURATIONS["beginLeftStance"]["swing_foot"] = "left"
MP_CONFIGURATIONS["beginLeftStance"]["end_stance_foot"] = "left"
MP_CONFIGURATIONS["beginLeftStance"]["stance_mode"] = "toe"
MP_CONFIGURATIONS["beginLeftStance"]["start_window_size"] = 10
MP_CONFIGURATIONS["beginLeftStance"]["end_window_size"] = 10

MP_CONFIGURATIONS["beginRightStance"] = dict()
MP_CONFIGURATIONS["beginRightStance"]["start_stance_foot"] = "both"
MP_CONFIGURATIONS["beginRightStance"]["stance_foot"] = "left"
MP_CONFIGURATIONS["beginRightStance"]["swing_foot"] = "right"
MP_CONFIGURATIONS["beginRightStance"]["end_stance_foot"] = "right"
MP_CONFIGURATIONS["beginRightStance"]["stance_mode"] = "toe"
MP_CONFIGURATIONS["beginRightStance"]["start_window_size"] = 10
MP_CONFIGURATIONS["beginRightStance"]["end_window_size"] = 10

MP_CONFIGURATIONS["endRightStance"] = dict()
MP_CONFIGURATIONS["endRightStance"]["start_stance_foot"] = "left"
MP_CONFIGURATIONS["endRightStance"]["stance_foot"] = "left"
MP_CONFIGURATIONS["endRightStance"]["swing_foot"] = "right"
MP_CONFIGURATIONS["endRightStance"]["end_stance_foot"] = "both"
MP_CONFIGURATIONS["endRightStance"]["swing_foot"] = "right"
MP_CONFIGURATIONS["endRightStance"]["stance_mode"] = "none"
MP_CONFIGURATIONS["endRightStance"]["start_window_size"] = 10
MP_CONFIGURATIONS["endRightStance"]["end_window_size"] = 10

MP_CONFIGURATIONS["endLeftStance"] = dict()
MP_CONFIGURATIONS["endLeftStance"]["start_stance_foot"] = "right"
MP_CONFIGURATIONS["endLeftStance"]["stance_foot"] = "right"
MP_CONFIGURATIONS["endLeftStance"]["swing_foot"] = "left"
MP_CONFIGURATIONS["endLeftStance"]["end_stance_foot"] = "both"
MP_CONFIGURATIONS["endLeftStance"]["swing_foot"] = "left"
MP_CONFIGURATIONS["endLeftStance"]["stance_mode"] = "none"
MP_CONFIGURATIONS["endLeftStance"]["start_window_size"] = 10
MP_CONFIGURATIONS["endLeftStance"]["end_window_size"] = 10


MP_CONFIGURATIONS["turnLeftRightStance"] = dict()
MP_CONFIGURATIONS["turnLeftRightStance"]["start_stance_foot"] = "both"
MP_CONFIGURATIONS["turnLeftRightStance"]["stance_foot"] = "left"
MP_CONFIGURATIONS["turnLeftRightStance"]["swing_foot"] = "right"
MP_CONFIGURATIONS["turnLeftRightStance"]["end_stance_foot"] = "right"
MP_CONFIGURATIONS["turnLeftRightStance"]["stance_mode"] = "none"
MP_CONFIGURATIONS["turnLeftRightStance"]["start_window_size"] = 20
MP_CONFIGURATIONS["turnLeftRightStance"]["end_window_size"] = 20

MP_CONFIGURATIONS["turnRightLeftStance"] = dict()
MP_CONFIGURATIONS["turnRightLeftStance"]["start_stance_foot"] = "both"
MP_CONFIGURATIONS["turnRightLeftStance"]["stance_foot"] = "right"
MP_CONFIGURATIONS["turnRightLeftStance"]["swing_foot"] = "left"
MP_CONFIGURATIONS["turnRightLeftStance"]["end_stance_foot"] = "left"
MP_CONFIGURATIONS["turnRightLeftStance"]["stance_mode"] = "none"
MP_CONFIGURATIONS["turnRightLeftStance"]["start_window_size"] = 20
MP_CONFIGURATIONS["turnRightLeftStance"]["end_window_size"] = 20


def generate_ankle_constraint_from_toe_without_orientation(skeleton, frames, frame_idx, ankle_joint_name, toe_joint_name, target_ground_height, toe_pos = None):
    """ create a constraint on the ankle position based on the toe constraint position"""
    if toe_pos is None:
        ct = skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])
        ct[1] = target_ground_height  # set toe constraint on the ground
    else:
        ct = toe_pos
    a = skeleton.nodes[ankle_joint_name].get_global_position(frames[frame_idx])
    t = skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])

    target_toe_offset = a - t  # difference between unmodified toe and ankle at the frame
    ca = ct + target_toe_offset  # move ankle so toe is on the ground
    return MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, None, None)


def blend_between_frames(skeleton, frames, start, end, joint_list, window):
    for joint in joint_list:
        idx = skeleton.animated_joints.index(joint) * 4 + 3
        j_indices = [idx, idx + 1, idx + 2, idx + 3]
        start_q = frames[start][j_indices]
        end_q = frames[end][j_indices]
        print(joint, window)
        for i in range(window):
            t = float(i) / window
            slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
            frames[start + i][j_indices] = slerp_q
            #print "blend at frame", start + i, slerp_q


def apply_constraint(skeleton, frames, frame_idx, c, blend_start, blend_end, blend_window=5):
    ik_chain = skeleton.skeleton_model["ik_chains"][c.joint_name]
    ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
    #print "b",c.joint_name,frame_idx,c.position, skeleton.nodes[c.joint_name].get_global_position(frames[frame_idx])
    frames[frame_idx] = ik.apply2(frames[frame_idx], c.position, c.orientation)
    #print "a",c.joint_name,frame_idx,c.position, skeleton.nodes[c.joint_name].get_global_position(frames[frame_idx])
    joint_list = [ik_chain["root"], ik_chain["joint"], c.joint_name]
    if blend_start < blend_end:
        blend_between_frames(skeleton, frames, blend_start, blend_end, joint_list, blend_window)


def move_to_ground(skeleton,frames, foot_joints, target_ground_height,start_frame=0, n_frames=5):
    print("n_frames",len(frames),start_frame, n_frames)
    source_ground_height = guess_ground_height(skeleton, frames, start_frame, n_frames, foot_joints)
    print("ground height", source_ground_height)
    for f in frames:
        f[1] += target_ground_height - source_ground_height


def ground_both_feet(skeleton, frames, target_height, frame_idx):
    constraints = []
    stance_foot = skeleton.skeleton_model["joints"]["right_ankle"]
    heel_joint = skeleton.skeleton_model["joints"]["right_heel"]
    toe_joint = skeleton.skeleton_model["joints"]["right_toe"]
    heel_offset = skeleton.skeleton_model["heel_offset"]
    c1 = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot, heel_joint, toe_joint, heel_offset, target_height)
    constraints.append(c1)

    stance_foot = skeleton.skeleton_model["joints"]["left_ankle"]
    toe_joint = skeleton.skeleton_model["joints"]["left_toe"]
    heel_joint = skeleton.skeleton_model["joints"]["left_heel"]
    c2 = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot, heel_joint, toe_joint, heel_offset, target_height)
    constraints.append(c2)
    return constraints


def ground_right_stance(skeleton, frames, target_height, frame_idx):
    constraints = []
    stance_foot = skeleton.skeleton_model["joints"]["right_ankle"]
    heel_joint = skeleton.skeleton_model["joints"]["right_heel"]
    toe_joint = skeleton.skeleton_model["joints"]["right_toe"]
    heel_offset = skeleton.skeleton_model["heel_offset"]
    c1 = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot, heel_joint, toe_joint, heel_offset,
                           target_height)
    constraints.append(c1)

    swing_foot = skeleton.skeleton_model["joints"]["left_ankle"]
    toe_joint = skeleton.skeleton_model["joints"]["left_toe"]
    heel_joint = skeleton.skeleton_model["joints"]["left_heel"]
    c2 = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, swing_foot, heel_joint, toe_joint, target_height)
    constraints.append(c2)
    return constraints


def ground_left_stance(skeleton, frames, target_height, frame_idx):
    constraints = []
    stance_foot = skeleton.skeleton_model["joints"]["left_ankle"]
    heel_joint = skeleton.skeleton_model["joints"]["left_heel"]
    toe_joint = skeleton.skeleton_model["joints"]["left_toe"]
    heel_offset = skeleton.skeleton_model["heel_offset"]
    c1 = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot, heel_joint, toe_joint, heel_offset, target_height)
    constraints.append(c1)

    swing_foot = skeleton.skeleton_model["joints"]["right_ankle"]
    toe_joint = skeleton.skeleton_model["joints"]["right_toe"]
    heel_joint = skeleton.skeleton_model["joints"]["right_heel"]
    c2 = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, swing_foot,heel_joint, toe_joint, target_height)
    constraints.append(c2)
    return constraints


def ground_first_frame(skeleton, frames, target_height, window_size, stance_foot="right", first_frame=0):
    if stance_foot == "both":
        constraints = ground_both_feet(skeleton, frames, target_height, first_frame)
    elif stance_foot == "right":
        constraints = ground_right_stance(skeleton, frames, target_height, first_frame)
    else:
        constraints = ground_left_stance(skeleton, frames, target_height, first_frame)

    c1 = constraints[0]
    c2 = constraints[1]
    root_pos = generate_root_constraint_for_two_feet(skeleton, frames[first_frame], c1, c2)
    if root_pos is not None:
        frames[first_frame][:3] = root_pos
        print("change root at frame", first_frame)
        smooth_root_translation_at_start(frames, first_frame, window_size)
    for c in constraints:
        apply_constraint(skeleton, frames, first_frame, c, first_frame, first_frame + window_size, window_size)


def ground_last_frame(skeleton, frames, target_height, window_size, stance_foot="left", last_frame=None):
    if last_frame is None:
        last_frame = len(frames) - 1
    if stance_foot == "both":
        constraints = ground_both_feet(skeleton, frames, target_height, last_frame)
        c1 = constraints[0]
        c2 = constraints[1]
        root_pos = generate_root_constraint_for_two_feet(skeleton, frames[last_frame], c1, c2)
    elif stance_foot == "left":
        constraints = ground_left_stance(skeleton, frames, target_height, last_frame)
        c1 = constraints[0]
        root_pos = generate_root_constraint_for_one_foot(skeleton, frames[last_frame], c1)
    else:
        constraints = ground_right_stance(skeleton, frames, target_height, last_frame)
        c1 = constraints[0]
        root_pos = generate_root_constraint_for_one_foot(skeleton, frames[last_frame], c1)
    if root_pos is not None:
        frames[last_frame][:3] = root_pos
        print("change root at frame", last_frame)
        smooth_root_translation_at_end(frames, last_frame, window_size)
    for c in constraints:
        apply_constraint(skeleton, frames, last_frame, c, last_frame - window_size, last_frame, window_size)


def ground_initial_stance_foot_unconstrained(skeleton, frames, target_height, stance_foot="right", mode="toe", start_frame=0, end_frame=None):
    if end_frame is None:
        end_frame = len(frames)
    foot_joint = skeleton.skeleton_model["joints"][stance_foot+"_foot"]
    toe_joint = skeleton.skeleton_model["joints"][stance_foot+"_toe"]
    heel_joint = skeleton.skeleton_model["joints"][stance_foot+"_heel"]
    heel_offset = skeleton.skeleton_model["heel_offset"]

    toe_pos = None
    heel_pos = None
    for frame_idx in range(start_frame, end_frame):
        if toe_pos is None:
            toe_pos = skeleton.nodes[toe_joint].get_global_position(frames[frame_idx])
            toe_pos[1] = target_height
            heel_pos = skeleton.nodes[heel_joint].get_global_position(frames[frame_idx])
            heel_pos[1] = target_height
        if mode == "toe":
            c = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, foot_joint, toe_joint, target_height, toe_pos)
        else:
            c = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, foot_joint, heel_joint, toe_joint, heel_offset, target_height, heel_pos, toe_pos)
        root_pos = generate_root_constraint_for_one_foot(skeleton, frames[frame_idx], c)
        if root_pos is not None:
            frames[frame_idx][:3] = root_pos
        apply_constraint(skeleton, frames, frame_idx, c, frame_idx, frame_idx)


def ground_initial_stance_foot(skeleton, frames, target_height, stance_foot="right", swing_foot="left", stance_mode="toe", start_frame=0, end_frame=None):
    if end_frame is None:
        end_frame = len(frames)
    stance_foot_joint = skeleton.skeleton_model["joints"][stance_foot + "_ankle"]
    stance_toe_joint = skeleton.skeleton_model["joints"][stance_foot + "_toe"]
    stance_heel_joint = skeleton.skeleton_model["joints"][stance_foot + "_heel"]
    swing_foot_joint = skeleton.skeleton_model["joints"][swing_foot + "_ankle"]
    swing_toe_joint = skeleton.skeleton_model["joints"][swing_foot + "_toe"]
    swing_heel_joint = skeleton.skeleton_model["joints"][swing_foot + "_heel"]
    heel_offset = skeleton.skeleton_model["heel_offset"]

    stance_toe_pos = None
    stance_heel_pos = None
    for frame_idx in range(start_frame, end_frame):
        temp_stance_toe_pos = skeleton.nodes[stance_toe_joint].get_global_position(frames[frame_idx])
        temp_stance_heel_pos = skeleton.nodes[stance_heel_joint].get_global_position(frames[frame_idx])

        if stance_toe_pos is None:
            stance_toe_pos = skeleton.nodes[stance_toe_joint].get_global_position(frames[frame_idx])
            stance_toe_pos[1] = target_height
            stance_heel_pos = skeleton.nodes[stance_heel_joint].get_global_position(frames[frame_idx])
            stance_heel_pos[1] = target_height
        if stance_mode == "toe" and temp_stance_heel_pos[1] >= target_height:
                #stance_c = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, stance_foot_joint, stance_heel_joint, stance_toe_joint,
                #                                       target_height, stance_toe_pos)

                stance_heel_pos[1] = temp_stance_heel_pos[1]
                stance_c = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot_joint, stance_heel_joint, stance_toe_joint, heel_offset, target_height, stance_heel_pos, stance_toe_pos, is_swinging=True)
                #stance_heel_pos = skeleton.nodes[stance_heel_joint].get_global_position(frames[frame_idx])
                #stance_c = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot_joint, stance_heel_joint,
                #                             stance_toe_joint, heel_offset,
                #                             target_height, stance_heel_pos, stance_toe_pos)
                #print("toe",stance_toe_pos)

        else:
            stance_c = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot_joint, stance_heel_joint, stance_toe_joint, heel_offset,
                                  target_height, stance_heel_pos, stance_toe_pos)

        swing_heel_pos = skeleton.nodes[swing_heel_joint].get_global_position(frames[frame_idx])
        swing_toe_pos = skeleton.nodes[swing_toe_joint].get_global_position(frames[frame_idx])
        swing_c = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, swing_foot_joint, swing_heel_joint, swing_toe_joint, heel_offset,
                                     target_height, swing_heel_pos, swing_toe_pos)

        #print "swing_c",swing_c.position,swing_heel_pos,swing_toe_pos
        #root_pos = generate_root_constraint_for_two_feet(skeleton, frames[frame_idx], stance_c, swing_c)
        root_pos = generate_root_constraint_for_one_foot(skeleton, frames[frame_idx], stance_c)
        if root_pos is not None:
            frames[frame_idx][:3] = root_pos
        #print "toe pos before ", frame_idx, skeleton.nodes[stance_toe_joint].get_global_position(frames[frame_idx])

        apply_constraint(skeleton, frames, frame_idx, stance_c, frame_idx, frame_idx)
        apply_constraint(skeleton, frames, frame_idx, swing_c, frame_idx, frame_idx)
        #print "toe pos after ",frame_idx, skeleton.nodes[stance_toe_joint].get_global_position(frames[frame_idx])



def align_xz_to_origin(skeleton, frames):
    root = skeleton.aligning_root_node
    tframe = frames[0]
    offset = np.array([0, 0, 0]) - skeleton.nodes[root].get_global_position(tframe)
    for frame_idx in range(0, len(frames)):
        frames[frame_idx, 0] += offset[0]
        frames[frame_idx, 2] += offset[2]


class MotionPrimitiveGrounding(object):
    def __init__(self, skeleton, mp_configs=MP_CONFIGURATIONS, target_height=0):
        self.skeleton = skeleton
        self.mp_configs = mp_configs
        self.target_height = target_height
        self.foot_joints = self.skeleton.skeleton_model["foot_joints"]

    def move_motion_to_ground(self, mv, step_offset, step_length):
        search_window_start = step_offset + int(step_length / 2)
        move_to_ground(self.skeleton, mv.frames, self.foot_joints, self.target_height, start_frame=search_window_start, n_frames=search_window_start+step_length-search_window_start)
        return mv

    def ground_feet(self, mv, mp_type, step_offset, step_length):
        config = self.mp_configs[mp_type]
        start_stance_foot = config["start_stance_foot"]
        stance_foot = config["stance_foot"]
        swing_foot = config["swing_foot"]
        end_stance_foot = config["end_stance_foot"]
        stance_mode = config["stance_mode"]
        start_window_size = config["start_window_size"]
        end_window_size = config["end_window_size"]

        ground_first_frame(self.skeleton, mv.frames, self.target_height, start_window_size, start_stance_foot, first_frame=step_offset)
        ground_last_frame(self.skeleton, mv.frames, self.target_height, end_window_size, end_stance_foot, last_frame=step_offset+step_length-1)
        if stance_mode is not "none":
            ground_initial_stance_foot(self.skeleton, mv.frames, self.target_height, stance_foot, swing_foot, stance_mode, start_frame=step_offset, end_frame=step_offset+step_length)
            # ground_initial_stance_foot_unconstrained(skeleton, mv.frames, target_height, stance_foot, stance_mode)
        return mv

    def align_to_origin(self, mv):
        align_xz_to_origin(self.skeleton, mv.frames)
        return mv
