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
import json
from transformations import quaternion_slerp
from anim_utils.animation_data.bvh import BVHReader
from anim_utils.animation_data.motion_vector import MotionVector
from anim_utils.animation_data.skeleton import Skeleton
from anim_utils.animation_data.motion_blending import smooth_quaternion_frames_using_slerp_
import numpy as np


class HandPose(object):
    def __init__(self):
        self.pose_vectors = dict()
        self.hand_skeletons = None


class HandPoseGenerator(object):
    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.pose_map = dict()
        self.status_change_map = dict()
        self.left_hand_skeleton = dict()
        self.right_hand_skeleton = dict()
        self.initialized = False
        return

    def init_from_desc(self, hand_pose_info):
        self.status_change_map = hand_pose_info["status_change_map"]
        self.right_hand_skeleton = hand_pose_info["right_hand_skeleton"]
        self.left_hand_skeleton = hand_pose_info["left_hand_skeleton"]
        self.right_hand_skeleton["indices"] = self.skeleton.get_joint_indices(hand_pose_info["right_hand_skeleton"]["joint_names"])
        self.left_hand_skeleton["indices"] = self.skeleton.get_joint_indices(hand_pose_info["left_hand_skeleton"]["joint_names"])
        for pose in list(hand_pose_info["poses"].keys()):
            hand_pose = HandPose()
            hand_pose.hand_skeletons = dict()
            hand_pose.hand_skeletons["RightHand"] = self.right_hand_skeleton
            hand_pose.hand_skeletons["LeftHand"] = self.left_hand_skeleton
            hand_pose.pose_vectors["LeftHand"] = np.asarray(hand_pose_info["poses"][pose]["LeftHand"])
            hand_pose.pose_vectors["RightHand"] = np.asarray(hand_pose_info["poses"][pose]["RightHand"])
            self.pose_map[pose] = hand_pose
        #for key in hand_pose_info["skeletonStrings"]:
        #    bvh_reader = BVHReader("").init_from_string(hand_pose_info["skeletonStrings"][key])
        #    skeleton = Skeleton(bvh_reader)
        #    self._add_hand_pose(key, skeleton)
        self.initialized = True

    def init_generator_from_directory(self, motion_primitive_dir):
        """
        creates a dicitionary for all possible hand poses
        TODO define in a file
        :param directory_path:
        :return:
        """
        hand_pose_directory = motion_primitive_dir+os.sep+"hand_poses"
        hand_pose_info_file = hand_pose_directory + os.sep + "hand_pose_info.json"
        if os.path.isfile(hand_pose_info_file):
            with open(hand_pose_info_file, "r") as in_file:
                hand_pose_info = json.load(in_file)
                self.init_from_desc(hand_pose_info)
                #for root, dirs, files in os.walk(hand_pose_directory):
                #    for file_name in files:
                #        if file_name[-4:] == ".bvh":
                #            print file_name[:-4]
                #            bvh_reader = BVHReader(root+os.sep+file_name)
                #            skeleton = Skeleton(bvh_reader)
                #            self._add_hand_pose(file_name[:-4], skeleton)
        else:
            print("Error: Could not load hand poses from", hand_pose_directory)

    def _add_hand_pose(self, name, skeleton):

         hand_pose = HandPose()
         hand_pose.hand_skeletons = dict()
         hand_pose.hand_skeletons["RightHand"] = self.right_hand_skeleton
         hand_pose.hand_skeletons["LeftHand"] = self.left_hand_skeleton
         hand_pose.pose_vector = skeleton.reference_frame
         self.pose_map[name] = hand_pose

    def _is_affecting_hand(self, hand, event_desc):
        if hand == "RightHand":
            return "RightToolEndSite" in event_desc["parameters"]["joint"] or\
                    "RightHand" in event_desc["parameters"]["joint"] or\
                   "RightToolEndSite" == event_desc["parameters"]["joint"] or\
                   "RightHand" == event_desc["parameters"]["joint"]

        elif hand == "LeftHand":
            return "LeftToolEndSite" in event_desc["parameters"]["joint"] or\
                    "LeftHand" in event_desc["parameters"]["joint"] or\
                   "LeftToolEndSite" == event_desc["parameters"]["joint"] or\
                   "LeftHand" == event_desc["parameters"]["joint"]

    def generate_hand_poses(self, motion_vector):
        if self.initialized:
            right_status = "standard"
            left_status = "standard"
            left_hand_events = []
            right_hand_events = []
            for frame_idx in range(motion_vector.n_frames):
                if frame_idx in list(motion_vector.keyframe_event_list.keyframe_events_dict["events"].keys()):
                    for event_desc in motion_vector.keyframe_event_list.keyframe_events_dict["events"][frame_idx]:
                        if event_desc["event"] != "transfer" and event_desc["event"] != "rotate":
                            if self._is_affecting_hand("RightHand", event_desc):
                                right_status = self.status_change_map[event_desc["event"]]
                                print("change right hand status to", right_status)
                                right_hand_events.append(frame_idx)
                            if self._is_affecting_hand("LeftHand", event_desc):
                                left_status = self.status_change_map[event_desc["event"]]
                                print("change left hand status to", left_status)
                                left_hand_events.append(frame_idx)
                        elif event_desc["event"] == "transfer":
                            right_hand_events.append(frame_idx)
                            left_hand_events.append(frame_idx)
                            tmp = right_status
                            right_status = left_status
                            left_status = tmp

                self.set_pose_in_frame("RightHand", right_status, motion_vector.frames[frame_idx])
                self.set_pose_in_frame("LeftHand", left_status, motion_vector.frames[frame_idx])

            quat_frames = np.array(motion_vector.frames)
            self.smooth_state_transitions(quat_frames, left_hand_events, self.left_hand_skeleton["indices"])
            self.smooth_state_transitions(quat_frames, right_hand_events, self.right_hand_skeleton["indices"])
            motion_vector.frames = quat_frames.tolist()

    def set_pose_in_frame(self, hand, status, pose_vector):
        """
        Overwrites the parameters in the given pose vector with the hand pose
        :param pose_vector:
        :return:
        """
        for src_idx, target_idx in enumerate(self.pose_map[status].hand_skeletons[hand]["indices"]):
            param_index = target_idx*4 + 3 #translation is ignored
            src_vector_idx = src_idx*4
            pose_vector[param_index:param_index+4] = self.pose_map[status].pose_vectors[hand][src_vector_idx:src_vector_idx+4]

    def smooth_state_transitions(self, quat_frames, events, indices, window=30):
        for event_frame in events:
            for i in indices:
                index = i*4+3
                smooth_quaternion_frames_using_slerp_(quat_frames, list(range(index, index+4)), event_frame, window)


    def nlerp(self, start, end, t):
        """http://physicsforgames.blogspot.de/2010/02/quaternions.html
        """
        dot = start[0]*end[0] + start[1]*end[1] + start[2]*end[2] + start[3]*end[3]
        result = np.array([0.0, 0.0, 0.0, 0.0])
        inv_t = 1.0 - t
        if dot < 0.0:
            temp = []
            temp[0] = -end[0]
            temp[1] = -end[1]
            temp[2] = -end[2]
            temp[3] = -end[3]
            result[0] = inv_t*start[0] + t*temp[0]
            result[1] = inv_t*start[1] + t*temp[1]
            result[2] = inv_t*start[2] + t*temp[2]
            result[3] = inv_t*start[3] + t*temp[3]

        else:
            result[0] = inv_t*start[0] + t*end[0]
            result[1] = inv_t*start[1] + t*end[1]
            result[2] = inv_t*start[2] + t*end[2]
            result[3] = inv_t*start[3] + t*end[3]

        return result/np.linalg.norm(result)
