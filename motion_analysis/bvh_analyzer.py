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
from anim_utils.animation_data import SkeletonBuilder, SKELETON_NODE_TYPE_END_SITE, LEN_EULER, LEN_ROOT,\
                                      LEN_QUAT
import numpy as np
from transformations import euler_matrix, euler_from_matrix
from .motion_plane import Plane
from anim_utils.animation_data.utils import pose_orientation_euler, check_quat, convert_quat_frame_value_to_array,\
     euler_to_quaternion, convert_euler_frames_to_quaternion_frames
from anim_utils.utilities.custom_math import angle_between_vectors


class BVHAnalyzer():

    def __init__(self, bvhreader):
        self.skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
        self.bvhreader = bvhreader
        self.quat_frames = []
        self.euler_frames = bvhreader.frames
        self.n_frames = len(self.euler_frames)
        self.body_plane = None

    def get_global_pos(self, joint_name, frame_index):
        joint_chain = self.get_joint_chain(joint_name)
        global_trans = np.eye(4)
        global_trans[:3, 3] = self.euler_frames[frame_index][:LEN_ROOT]
        for joint in joint_chain:
            offset = joint.offset
            if 'EndSite' in joint.node_name: # end site joint
                rot_mat = np.eye(4)
                rot_mat[:3, 3] = offset
            else:
                rot_angles_euler = self.get_relative_orientation_euler(joint.node_name, frame_index)
                rot_angles_rad = np.deg2rad(rot_angles_euler)
                rot_mat = euler_matrix(rot_angles_rad[0],
                                       rot_angles_rad[1],
                                       rot_angles_rad[2],
                                       'rxyz')
                rot_mat[:3, 3] = offset
            global_trans = np.dot(global_trans, rot_mat)
        return global_trans[:3, 3]

    def get_global_joint_positions(self, joint_name):
        '''
        Get joint positions for the sequence of frames
        :param joint_name: str
        :return: numpy.array<3d>
        '''
        joint_pos = np.zeros((self.n_frames, LEN_ROOT))
        for i in range(self.n_frames):
            joint_pos[i] = self.get_global_pos(joint_name, i)
        return joint_pos

    def get_relative_joint_position(self, joint_name, frame_index):
        """
        relative joint position to Hips
        :param joint_name: str
        :param frame_index: int
        :return:
        """
        joint_global_pos = self.get_global_pos(joint_name, frame_index)
        root_global_pos = self.get_global_pos('Hips', frame_index)
        return joint_global_pos - root_global_pos

    def get_filtered_joint_index(self, joint_name):
        return self.skeleton.node_name_frame_map.keys().index(joint_name)

    def get_parent_joint_name(self, joint_name):
        node = self.get_joint_by_joint_name(joint_name)
        if node.parent is not None:
            return node.parent.node_name
        else:
            return None

    def get_filtered_joint_param_range(self, joint_name):
        reduced_joint_index = self.get_filtered_joint_index(joint_name)
        start_index = LEN_ROOT + reduced_joint_index * LEN_QUAT
        end_index = LEN_ROOT + (reduced_joint_index + 1) * LEN_QUAT
        return start_index, end_index

    def get_joint_speed_at_frame_each_dim(self, joint_name, frame_idx):
        assert frame_idx != 0, ("Index starts from 1")
        return self.get_global_pos(joint_name, frame_idx) - self.get_global_pos(joint_name, frame_idx-1)

    def get_joint_speed_each_dim(self, joint_name):
        speed = [np.zeros(3)]
        for i in range(1, self.n_frames):
            speed.append(self.get_joint_speed_at_frame_each_dim(joint_name, i))
        return np.asarray(speed)

    def get_joint_speed(self, joint_name):
        speed = []
        for i in range(1, self.n_frames):
            speed.append(self.get_joint_speed_at_frame(joint_name, i))
        return np.asarray(speed)

    def get_joint_speed_at_frame(self, joint_name, frame_idx):
        assert frame_idx != 0, ("Index starts from 1")
        return np.linalg.norm(self.get_global_pos(joint_name, frame_idx) - self.get_global_pos(joint_name, frame_idx-1))

    def get_joint_acceleration_at_frame(self, joint_name, frame_idx):
        assert frame_idx != self.n_frames - 1 and frame_idx != 0, ("frame index is out of range!")
        return self.get_global_pos(joint_name, frame_idx + 1) + self.get_global_pos(joint_name, frame_idx - 1) - \
               2 * self.get_global_pos(joint_name, frame_idx)

    def get_joint_acceleration(self, joint_name):
        acc = [np.zeros(3)]
        for i in range(1, self.n_frames-1):
            acc.append(self.get_joint_acceleration_at_frame(joint_name, i))
        acc.append(np.zeros(3))
        return np.asarray(acc)

    def get_global_pos_for_all_frames(self, joint_name):
        pos = np.zeros((self.n_frames, 3))
        for i in range(self.n_frames):
            pos[i] = self.get_global_pos(joint_name, i)
        return pos

    def get_joint_chain(self, joint_name):
        joint = self.get_joint_by_joint_name(joint_name)
        joint_chain = []
        while joint.parent is not None:
            joint_chain.append(joint)
            joint = joint.parent
        joint_chain.append(joint)
        joint_chain.reverse()
        return joint_chain

    def get_relative_pos(self, joint_name, frame_index):
        joint_chain = self.get_joint_chain(joint_name)
        if len(joint_chain) == 1:
            raise ValueError('Root joint has no relative position')
        pos = self.get_global_pos(joint_name, frame_index)
        parent_pos = self.get_global_pos(joint_chain[-2].node_name, frame_index)
        return pos - parent_pos

    def get_joint_offset(self, joint_name):
        return self.skeleton.nodes[joint_name].offset

    def _get_nodes_without_endsite(self):
        animated_nodes = self.skeleton.nodes.values()
        nodes_without_endsite = [node for node in animated_nodes if node.node_type != SKELETON_NODE_TYPE_END_SITE]
        return nodes_without_endsite

    def get_relative_orientation_euler(self, joint_name, frame_index):
        # assert frame_index in range(self.n_frames), ('Frame index is invalid!')
        nodes_without_endsite = self._get_nodes_without_endsite()
        # assert (len(nodes_without_endsite)+1) * 3 == len(self.euler_frames[0]), \
        #     ('The length of euler frame is not corresponding to length of modeled joints')
        joint = self.get_joint_by_joint_name(joint_name)
        assert joint in nodes_without_endsite, ("The joint is not modeled!")
        joint_index = nodes_without_endsite.index(joint)
        start_channel_index = joint_index * 3 + LEN_ROOT
        end_channel_index = start_channel_index + LEN_EULER
        return self.euler_frames[frame_index][start_channel_index: end_channel_index]

    def get_global_transform(self, joint_name, frame_index):
        joint_chain = self.get_joint_chain(joint_name)
        global_trans = np.eye(4)
        global_trans[:3, 3] = self.euler_frames[frame_index][:LEN_ROOT]
        for joint in joint_chain:
            offset = joint.offset
            if 'EndSite' in joint.node_name: # end site joint
                rot_mat = np.eye(4)
                rot_mat[:3, 3] = offset
            else:
                rot_angles_euler = self.get_relative_orientation_euler(joint.node_name, frame_index)
                rot_angles_rad = np.deg2rad(rot_angles_euler)
                rot_mat = euler_matrix(rot_angles_rad[0],
                                       rot_angles_rad[1],
                                       rot_angles_rad[2],
                                       'rxyz')
                rot_mat[:3, 3] = offset
            global_trans = np.dot(global_trans, rot_mat)
        return global_trans

    def get_global_orientation_euler(self, joint_name, frame_index):
        joint_chain = self.get_joint_chain(joint_name)
        global_trans = np.eye(4)
        global_trans[:3, 3] = self.euler_frames[frame_index][:LEN_ROOT]
        for joint in joint_chain:
            offset = joint.offset
            rot_angles_euler = self.get_relative_orientation_euler(joint.node_name, frame_index)
            rot_angles_rad = np.deg2rad(rot_angles_euler)
            rot_mat = euler_matrix(rot_angles_rad[0],
                                   rot_angles_rad[1],
                                   rot_angles_rad[2],
                                   'rxyz')
            rot_mat[:3, 3] = offset
            global_trans = np.dot(global_trans, rot_mat)
        global_angles_rad = euler_from_matrix(global_trans,
                                              'rxyz')
        return np.rad2deg(global_angles_rad)

    def get_global_orientation_quat(self, joint_name, frame_index):
        return euler_to_quaternion(self.get_global_orientation_euler(joint_name,
                                                                     frame_index))

    def set_relative_orientation_euler(self, joint_name, frame_index, euler_angles):
        """

        :param joint_name: str
        :param frame_index: int
        :param euler_angles: array<float> degree
        :return:
        """
        # assert frame_index in range(self.n_frames), ('Frame index is invalid!')
        animated_nodes = self.skeleton.nodes.values()
        nodes_without_endsite = [node for node in animated_nodes if node.node_type != SKELETON_NODE_TYPE_END_SITE]
        assert (len(nodes_without_endsite)+1) * 3 == len(self.euler_frames[0]), \
            ('The length of euler frame is not corresponding to length of modeled joints')
        joint_index = 0
        for node in nodes_without_endsite:
            if node.node_name == joint_name:
                break
            else:
                joint_index += 1
        start_channel_index = (joint_index + 1) * 3
        end_channel_index = start_channel_index + LEN_EULER
        self.euler_frames[frame_index][start_channel_index: end_channel_index] = euler_angles

    def get_joint_index(self, joint_name):
        joint_name_list = self.skeleton.nodes.keys()
        if joint_name not in joint_name_list:
            raise ValueError('joint name is not found!')
        return joint_name_list.index(joint_name)

    def set_joint_offset(self, joint_name, offset):
        assert len(offset) == 3, ('The length of joint is not correct')
        joint = self.get_joint_by_joint_name(joint_name)
        joint.offset = [offset[0], offset[1], offset[2]]

    def get_joint_by_joint_name(self, joint_name):
        if joint_name not in self.skeleton.nodes.keys():
            print(joint_name)
            raise KeyError('Joint name is not found!')
        return self.skeleton.nodes[joint_name]

    def to_quaternion(self, filter_joints=True):
        self.quat_frames = np.array(convert_euler_frames_to_quaternion_frames(self.bvhreader,
                                                                              self.euler_frames,
                                                                              filter_joints))

    def get_joint_channel_in_full_euler_frame(self, joint):
        """

        :param joint: str, joint name
        :return:
        """
        return self.skeleton.node_channels.index((joint, 'Xrotation'))

    def get_closure_kinematic_chain(self, joint):
        joint_chain = []
        if joint.parent is not None:
            joint_chain.append(joint)
        return joint_chain.reverse()

    def get_body_plane(self, frame_idx):
        body_plane_joints = ['Hips', 'Spine', 'LeftShoulder', 'RightShoulder', 'LeftUpLeg', 'RightUpLeg']
        points = []
        for joint in body_plane_joints:
            points.append(self.get_relative_joint_position(joint, frame_idx))
        points = np.asarray(points)
        return Plane(points)

    def get_left_elbow_angle(self, frame_idx):
        left_arm_pos = self.get_global_pos('LeftArm', frame_idx)
        left_forearm_pos = self.get_global_pos('LeftForeArm', frame_idx)
        left_hand_pos = self.get_global_pos('LeftHand', frame_idx)
        upper_arm = left_forearm_pos - left_arm_pos
        lower_arm = left_forearm_pos - left_hand_pos
        theta = np.arccos(np.dot(upper_arm, lower_arm)/(np.linalg.norm(upper_arm) * np.linalg.norm(lower_arm)))
        theta = np.rad2deg(theta)
        return theta

    def get_left_elbow_angles(self):
        left_elbow_anlges = []
        for i in range(self.n_frames):
            left_elbow_anlges.append(self.get_left_elbow_angle(i))
        return left_elbow_anlges

    def get_right_elbow_angle(self, frame_idx):
        right_arm_pos = self.get_global_pos('RightArm', frame_idx)
        right_forearm_pos = self.get_global_pos('RightForeArm', frame_idx)
        right_hand_pos = self.get_global_pos('RightHand', frame_idx)
        upper_arm = right_forearm_pos - right_arm_pos
        lower_arm = right_forearm_pos - right_hand_pos
        theta = np.arccos(np.dot(upper_arm, lower_arm)/(np.linalg.norm(upper_arm) * np.linalg.norm(lower_arm)))
        theta = np.rad2deg(theta)
        return theta

    def get_right_elbow_anlges(self):
        right_elbow_angles = []
        for i in range(self.n_frames):
            right_elbow_angles.append(self.get_right_elbow_angle(i))
        return right_elbow_angles

    def right_hand_forward(self):
        relative_right_hand_pos = np.zeros((self.n_frames, 3))
        for i in range(self.n_frames):
            relative_right_hand_pos[i] = self.get_global_pos('RightHand', i) - self.get_global_pos('Hips', i)
        moving_offsets = relative_right_hand_pos[1:] - relative_right_hand_pos[:-1]
        annotation = [False]
        for i in range(self.n_frames-1):
            body_dir = pose_orientation_euler(self.euler_frames[i+1])
            if np.dot(body_dir, np.array([moving_offsets[i, 0], moving_offsets[i, 2]])) > 0.5:
                annotation.append(True)
            else:
                annotation.append(False)
        return annotation

    def left_hand_forward(self):
        left_hand_pos = np.zeros((self.n_frames, 3))
        for i in range(self.n_frames):
            left_hand_pos[i] = self.get_global_pos('LeftHand', i)
        moving_offsets = left_hand_pos[1:] - left_hand_pos[:-1]
        annotation = [False]
        for i in range(self.n_frames-1):
            body_dir = pose_orientation_euler(self.euler_frames[i+1])
            if np.dot(body_dir, np.array([moving_offsets[i, 0], moving_offsets[i, 2]])) > 0.1:
                annotation.append(True)
            else:
                annotation.append(False)
        return annotation

    def feet_distance_on_ground(self):
        left_foot_pos = self.get_global_joint_positions('LeftFoot')
        right_foot_pos = self.get_global_joint_positions('RightFoot')
        feet_distance = []
        for i in range(self.n_frames):
            feet_distance.append(np.linalg.norm(left_foot_pos[i, [0, 2]] - right_foot_pos[i, [0, 2]]))
        return np.asarray(feet_distance)

    def rfoot_behind_lleg(self, frame_index, jointlist=['LeftUpLeg', 'RightUpLeg', 'LeftFoot', 'RightFoot']):
        """
        involved joints: Hips, LeftUpLeg, LeftFoot, RightLeg
        :return:
        """
        points = []
        for joint in jointlist:
            points.append(self.get_global_pos(joint, frame_index))
        # determine the last point is before the body plane defined by the other three joints or behind
        # reverse the list of joints, because the direction of the plane is decided by the right-hand rule
        body_plane = Plane(points[:3])
        return not body_plane.is_before_plane(points[-1])

    def lfoot_behind_rleg(self, frame_index, jointlist=['LeftUpLeg', 'RightUpLeg', 'RightFoot', 'LeftFoot']):
        """
        involve joints: Hips, RightUpLeg, RightFoot, LeftLeg
        :param frame_index:
        :return:
        """
        points = []
        for joint in jointlist:
            points.append(self.get_global_pos(joint, frame_index))
        body_plane = Plane(points[:3])
        return not body_plane.is_before_plane(points[-1])

    def rhand_moving_forwards(self, frameIndex):
        """
        involved joints: body plane and RightHand
        :param frameIndex:
        :return:
        """
        if self.body_plane is None:
            self.get_body_plane(frameIndex)
        if frameIndex == self.n_frames - 1:
            return False
        else:
            current_distance = self.joint_disntace_to_body('RightHand', frameIndex)
            next_distance = self.joint_disntace_to_body('RightHand', frameIndex + 1)
            if next_distance - current_distance > 0.1:
                return True
            else:
                return False

    def lhand_moving_forwards(self, frameIndex):
        """
        involved joints: body plane and LeftHand
        :param frameIndex:
        :return:
        """
        if self.body_plane is None:
            self.get_body_plane(frameIndex)
        left_hand_pos = self.get_relative_joint_position('LeftHand', frameIndex)
        if frameIndex == self.n_frames - 1:
            return False
        else:
            next_pos = self.get_relative_joint_position('LeftHand', frameIndex + 1)
            current_distance = self.body_plane.distance(left_hand_pos)
            next_distance = self.body_plane.distance(next_pos)
            if next_distance - current_distance > 0.1:
                return True
            else:
                return False

    def lhand_moving_forwards_one_frame(self, frameIndex):
        threshold = 0.1
        if frameIndex <= 0:
            return False
        else:
            current_pos = self.get_relative_joint_position('LeftHand', frameIndex)
            previous_pos = self.get_relative_joint_position('LeftHand', frameIndex)
            if self.body_plane is None:
                self.get_body_plane(frameIndex)
            current_dist = self.body_plane.distance(current_pos)
            previous_dist = self.body_plane.distance(previous_pos)
            if current_dist - previous_dist > threshold:
                return True
            else:
                return False

    def lhand_moving_forwards2(self, frameIndex, windowSize=10):
        if frameIndex < windowSize:
            max_frame = frameIndex
        elif self.n_frames - frameIndex < windowSize:
            max_frame = self.n_frames - frameIndex - 1
        else:
            max_frame = windowSize
        w = 1
        while w <= max_frame:
            prev_frame = self.lhand_moving_forwards_one_frame(frameIndex - w)
            next_frame = self.lhand_moving_forwards_one_frame(frameIndex + w)
            if prev_frame and next_frame:
                return 1
            elif not prev_frame and not next_frame:
                return -1
            else:
                w += 1
        return 0

    def joint_disntace_to_body(self, jointname, frameIndex):
        body_plane = self.get_body_plane(frameIndex)
        joint_pos = self.get_relative_joint_position(jointname, frameIndex)
        return body_plane.distance(joint_pos)

    def rhand_moving_forwards_one_frame(self, frameIndex):
        threshold = 0.1
        if frameIndex <= 0:
            return False
        else:
            current_dist = self.joint_disntace_to_body('RightHand', frameIndex)
            previous_dist = self.joint_disntace_to_body('RightHand', frameIndex - 1)
            # print('current distance: ', current_dist)
            # print('previous distance: ', previous_dist)
            if current_dist - previous_dist > threshold:
                return True
            else:
                return False

    def rhand_moving_forwards2(self, frameIndex, windowSize=10):
        if frameIndex < windowSize:
            max_frame = frameIndex
        elif self.n_frames - frameIndex < windowSize:
            max_frame = self.n_frames - frameIndex - 1
        else:
            max_frame = windowSize
        # print("test1 max_frame: ", max_frame)
        w = 1
        while w <= max_frame:
            prev_frame = self.rhand_moving_forwards_one_frame(frameIndex - w)
            next_frame = self.rhand_moving_forwards_one_frame(frameIndex + w)
            # print("w: ", w)
            # print("prev_frame: ", prev_frame)
            # print("next_frame: ", next_frame)
            if prev_frame and next_frame:
                return 1
            elif not prev_frame and not next_frame:
                return -1
            else:
                w += 1
        return 0

    def lknee_angle(self, frameIndex):
        """
        involved joints: LeftUpLeg, LeftLeg, LeftFoot
        :param frameIndex:
        :return:
        """
        leftUpLeg_position = self.get_relative_joint_position('LeftUpLeg', frameIndex)
        leftLeg_position = self.get_relative_joint_position('LeftLeg', frameIndex)
        leftFoot_position = self.get_relative_joint_position('LeftFoot', frameIndex)
        upLegBone = leftLeg_position - leftUpLeg_position
        lowLegBone = leftFoot_position - leftLeg_position
        return angle_between_vectors(upLegBone, lowLegBone)

    def rknee_angle(self, frameIndex):
        """
        involved joints: RightUpLeg, RightLeg, RightFoot
        :param frameIndex:
        :return:
        """
        rightUpLeg_position = self.get_relative_joint_position('RightUpLeg', frameIndex)
        rightLeg_position = self.get_relative_joint_position('RightLeg', frameIndex)
        rightFoot_position = self.get_relative_joint_position('RightFoot', frameIndex)
        upLegBone = rightLeg_position - rightUpLeg_position
        lowLegBone = rightFoot_position - rightLeg_position
        return angle_between_vectors(upLegBone, lowLegBone)

    def lleg_bending(self, frameIndex):
        """
        involved joints: LeftUpLeg, LeftLeg, LeftFoot
        :param frameIndex:
        :param w (int): window size
        :return:
        reverse indexing is not supported
        """
        angle_threshold = 0.001
        if frameIndex <= 0:
            return False
        else:
            previous_angle = self.lknee_angle(frameIndex - 1)
            angle = self.lknee_angle(frameIndex)
            if angle - previous_angle < -angle_threshold:
                return True
            else:
                return False

    def lleg_stretching(self, frameIndex):
        """
        involved joints: LeftUpLeg, LeftLeg, LeftFoot
        :param frameIndex:
        :param w (int): window size
        :return:
        reverse indexing is not supported
        """
        angle_threshold = 0.01
        if frameIndex <= 0:
            return False
        else:
            previous_angle = self.lknee_angle(frameIndex - 1)
            angle = self.lknee_angle(frameIndex)
            if angle - previous_angle >angle_threshold:
                return True
            else:
                return False

    def rleg_bending(self, frameIndex):
        """
        involved joints: RightUpLeg, RightLeg, RightFoot
        :param frameIndex:
        :param w (int): window size
        :return:
        reverse indexing is not supported
        """
        angle_threshold = 0.001
        if frameIndex <= 0:
            return False
        else:
            previous_angle = self.rknee_angle(frameIndex - 1)
            angle = self.rknee_angle(frameIndex)
            if angle - previous_angle < -angle_threshold:
                return True
            else:
                return False

    def rleg_stretching(self, frameIndex):
        """
        involved joints: RightUpLeg, RightLeg, RightFoot
        :param frameIndex:
        :param w (int): window size
        :return:
        reverse indexing is not supported
        """
        angle_threshold = 0.01
        if frameIndex <= 0:
            return False
        else:
            previous_angle = self.rknee_angle(frameIndex - 1)
            angle = self.rknee_angle(frameIndex)
            if angle - previous_angle > angle_threshold:
                return True
            else:
                return False

    def rtoe_before_lleg(self, frameIndex):
        """
        involved joints: Hips, LeftUpLeg, LeftLeg, Bip01_R_Toe0
        :param frameIndex:
        :return:
        """
        jointList = ['Hips', 'LeftUpLeg', 'LeftLeg', 'Bip01_R_Toe0']
        points = []
        for joint in jointList:
            points.append(self.get_relative_joint_position(joint, frameIndex))
        points.reverse()
        relative_plane = Plane(points[1:])
        return relative_plane.is_before_plane(points[0])

    def ltoe_before_rleg(self, frameIndex):
        """
        involved joints: Hips, RightUpLeg, RightLeg, Bip01_L_Toe0
        :param frameIndex:
        :return:
        """
        jointlist = ['Hips', 'RightUpLeg', 'RightLeg', 'Bip01_L_Toe0']
        points = []
        for joint in jointlist:
            points.append(self.get_relative_joint_position(joint, frameIndex))
        relative_plane = Plane(points[:3])
        return relative_plane.is_before_plane(points[-1])

    def spine_horizontal(self, frameIndex):
        """
        involved joints:
        :param frameIndex:
        :return:
        """
        pass

    def feet_moving_towards_each_other(self):
        '''
        Feature: Distance between two feet on the ground
        involved joints:
        :return Boolean: status
        '''
        pass

    def process(self, frame_idx):
        '''
        use a list of signal processor to process given frame
        :return:
        '''
        pass