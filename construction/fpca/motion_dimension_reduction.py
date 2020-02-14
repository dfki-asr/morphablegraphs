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
Created on Sun Aug 02 13:15:01 2015

@author: Han Du
"""

import numpy as np
import os
import json
import copy
from collections import OrderedDict
from transformations import quaternion_multiply
from .fpca_temporal_data import FPCATemporalData
from .fpca_spatial_data import FPCASpatialData
from anim_utils.utilities.custom_math import areQuatClose, diff_quat, quat_to_logmap, normalize_quaternion, logmap_to_quat
from ..utils import get_data_analysis_folder
from anim_utils.utilities import load_json_file, write_to_json_file
from anim_utils.animation_data import BVHReader
from anim_utils.animation_data.utils import convert_euler_frames_to_reduced_euler_frames
from anim_utils.animation_data.quaternion_frame import convert_euler_frames_to_quaternion_frames
LEN_QUATERNION = 4
LEN_ROOT_POSITION = 3
LEN_LOGMAP = 3


class MotionDimensionReduction(object):

    def __init__(self, skeleton_bvh, params):
        """
        * motion_data: dictionary
        \t{'filename': {'frames': euler frames, 'warping_index': warping frame index}}
        :param motion_data (dic): frames and warping indices
        :param skeleton_bvh (BVHReader)
        :param params:
        :return:
        """
        self.params = params

        self.repo_dir = r'C:\repo'
        self.spatial_data = OrderedDict()
        self.temporal_data = OrderedDict()
        self.skeleton_bvh = skeleton_bvh
        self.fdata = {}
        self.fdata['motion_type'] = self.params.elementary_action + '_' + \
            self.params.motion_primitive
        self.fdata['n_dim_spatial'] = 79
        self.fdata['n_basis'] = self.params.n_basis_functions_spatial
        self.fpca_temporal = None
        self.fpca_spatial = None
        self.quat_frames = OrderedDict()
        self.rescaled_quat_frames = OrderedDict()
        self.data_analysis_folder = get_data_analysis_folder(params.elementary_action,
                                                             params.motion_primitive,
                                                             self.repo_dir)

    def load_motion_data(self, motion_data):
        self.motion_data = motion_data
        for filename, data in self.motion_data.items():
            self.spatial_data[filename] = data['frames']
            self.temporal_data[filename] = data['warping_index']
        self.fdata['n_frames'] = len(self.spatial_data[list(self.spatial_data.keys())[0]])

    def load_temporal_data(self, temporal_data):
        self.temporal_data = temporal_data

    def load_spatial_data(self, spatial_data):
        self.spatial_data = spatial_data
        self.fdata['n_frames'] = len(self.spatial_data[list(self.spatial_data.keys())[0]])

    def use_fpca_on_temporal_params(self):
        if not self.temporal_data:
            raise ValueError('temporal data is not loaded!')
        self.fpca_temporal = FPCATemporalData(
            self.temporal_data,
            self.params.n_basis_functions_temporal,
            self.params.npc_temporal)
        self.fpca_temporal.fpca_on_temporal_data()

    def use_fpca_on_spatial_params(self):
        if not self.spatial_data:
            raise ValueError('spatial data is not loaded!')
        self.convert_euler_to_quat()
        self.scale_rootchannels()
        smoothed_quat_frames = self.smooth_quat_frames(self.rescaled_quat_frames)
        # smoothed_quat_frames = self.rescaled_quat_frames
        print("smoothing is done!")
        self.fpca_spatial = FPCASpatialData(
            n_basis=self.params.n_basis_functions_spatial,
            fraction=self.params.fraction)
        self.fpca_spatial.fit_motion_dictionary(smoothed_quat_frames)

    def use_fpca_on_spatial_params_without_scale(self):
        self.convert_euler_to_quat()
        smoothed_quat_frames = self.smooth_quat_frames(self.quat_frames)
        self.fpca_spatial = FPCASpatialData(
            n_basis=self.params.n_basis_functions_spatial,
            fraction=self.params.fraction
        )
        self.fpca_spatial.fit_motion_dictionary(smoothed_quat_frames)

    def convert_euler_to_quat(self):
        for filename, frames in self.spatial_data.items():
            self.quat_frames[filename] = convert_euler_frames_to_quaternion_frames(self.skeleton_bvh, frames)

    @staticmethod
    def get_mean_motion(smoothed_quat_frames):
        """
        Calculate average motion by averaging quaternion values
        :return:
        """
        smoothed_quat_values = np.asarray(list(smoothed_quat_frames.values()))
        average_quat_value = np.average(smoothed_quat_values, axis=0)
        n_frames, n_dims = average_quat_value.shape
        n_quats = (n_dims - LEN_ROOT_POSITION)/LEN_QUATERNION
        for i in range(n_frames):
            for j in range(n_quats):
                average_quat_value[i, LEN_ROOT_POSITION + j * LEN_QUATERNION : LEN_ROOT_POSITION + (j + 1) * LEN_QUATERNION] =\
                normalize_quaternion(average_quat_value[i, LEN_ROOT_POSITION + j * LEN_QUATERNION : LEN_ROOT_POSITION + (j + 1) * LEN_QUATERNION])
        return average_quat_value

    @staticmethod
    def centralize_motion_data(smoothed_quat_frames):
        """
        Centrailize the motion data, for root translation, subtract mean root translation. For orientation, the rotation
        from current quaternion to mean quaternion is calculated.
        :param smoothed_quat_frames (dic):
        :return:
        """
        mean_moton = MotionDimensionReduction.get_mean_motion(smoothed_quat_frames)
        centrailized_motion_data = {}
        for key, value in smoothed_quat_frames.items():
            centrailized_motion_data[key] = MotionDimensionReduction.quat_motion_subtraction(value, mean_moton).tolist()
        return centrailized_motion_data, mean_moton.tolist()

    @staticmethod
    def quat_motion_subtraction(quat_motion_a, quat_motion_b):
        quat_motion_a = np.asarray(quat_motion_a)
        quat_motion_b = np.asarray(quat_motion_b)
        assert quat_motion_a.shape == quat_motion_b.shape
        res_motion = np.zeros(quat_motion_a.shape)
        for i in range(len(quat_motion_a)):
            res_motion[i] = MotionDimensionReduction.quat_frame_subtraction(quat_motion_a[i],
                                                                            quat_motion_b[i])
        return res_motion

    @staticmethod
    def quat_motion_addition(quat_motion_a, quat_motion_b):
        quat_motion_a = np.asarray(quat_motion_a)
        quat_motion_b = np.asarray(quat_motion_b)
        assert quat_motion_a.shape == quat_motion_b.shape
        res_motion = np.zeros(quat_motion_a.shape)
        for i in range(len(quat_motion_a)):
            res_motion[i] = MotionDimensionReduction.quat_frame_addition(quat_motion_a[i],
                                                                         quat_motion_b[i])
        return res_motion

    def get_full_euler_frames(self, load_data=True, export_data=False):
        full_euler_frames_filename = os.path.join(self.data_analysis_folder, 'full_euler_frames.json')
        if load_data and os.path.exists(full_euler_frames_filename):
            full_euler_frames_dic = load_json_file(full_euler_frames_filename)['data']
        else:
            full_euler_frames_dic = {}
            for filename, frames in self.spatial_data.items():
                full_euler_frames_dic[filename] = frames
        if export_data:
            output_data = {'data': full_euler_frames_dic}
            write_to_json_file(full_euler_frames_filename, output_data)
        return full_euler_frames_dic

    def get_reduced_euler_frames(self, load_data=True, export_data=False):
        reduced_euler_frames_filename = os.path.join(self.data_analysis_folder, 'reduced_euler_frames.json')
        if load_data and os.path.exists(reduced_euler_frames_filename):
            reduced_euler_frames_dic = load_json_file(reduced_euler_frames_filename)['data']
        else:
            bvhreader = BVHReader(self.skeleton_bvh)
            reduced_euler_frames_dic = {}
            for filename, frames in self.spatial_data.items():
                reduced_euler_frames_dic[filename] = convert_euler_frames_to_reduced_euler_frames(bvhreader, frames)
        if export_data:
            output_data = {'data': reduced_euler_frames_dic}
            write_to_json_file(reduced_euler_frames_filename, output_data)
        return reduced_euler_frames_dic

    @staticmethod
    def get_root_trajectory_data(smoothed_quat_frames):
        root_trajectory = OrderedDict()
        for key, value in smoothed_quat_frames.items():
            root_trajectory[key] = np.asarray(value)[:, :LEN_ROOT_POSITION]
        return root_trajectory

    @staticmethod
    def get_pose_data(smoothed_quat_frames):
        pose_data = OrderedDict()
        for key, value in smoothed_quat_frames.items():
            pose_data[key] = np.asarray(value)[:, LEN_ROOT_POSITION:]
        return pose_data

    @staticmethod
    def smooth_quat_frames(quat_frames):
        smoothed_quat_frames = {}
        filenames = list(quat_frames.keys())
        smoothed_quat_frames_data = np.asarray(copy.deepcopy(list(quat_frames.values())))
        print(('quaternion frame data shape: ', smoothed_quat_frames_data.shape))
        assert len(smoothed_quat_frames_data.shape) == 3, ('The shape of quaternion frames is not correct!')
        n_samples, n_frames, n_dims = smoothed_quat_frames_data.shape
        assert n_dims == 79, ('The length of dimension is not correct!')
        n_joints = (n_dims - 3)/4
        for i in range(n_joints):
            ref_quat = smoothed_quat_frames_data[0, 0, i*LEN_QUATERNION + LEN_ROOT_POSITION : (i+1)*LEN_QUATERNION + LEN_ROOT_POSITION]
            for j in range(1, n_samples):
                test_quat = smoothed_quat_frames_data[j, 0, i*LEN_QUATERNION + LEN_ROOT_POSITION : (i+1)*LEN_QUATERNION + LEN_ROOT_POSITION]
                if not areQuatClose(ref_quat, test_quat):
                    smoothed_quat_frames_data[j, :, i*LEN_QUATERNION + LEN_ROOT_POSITION : (i+1)*LEN_QUATERNION + LEN_ROOT_POSITION] *= -1
        for i in range(len(filenames)):
            smoothed_quat_frames[filenames[i]] = smoothed_quat_frames_data[i]
        return smoothed_quat_frames

    def reshape_quat_frame(self, quat_frame_values):
        n_channels = len(quat_frame_values)
        quat_channels = n_channels - 1
        n_dims = quat_channels * LEN_QUATERNION + LEN_ROOT_POSITION
        self.fdata['n_dim_spatial'] = n_dims
        # in order to use Functional data representation from Fda(R), the
        # input data should be a matrix of shape (n_frames * n_samples *
        # n_dims)
        quat_frame_value_array = []
        for item in quat_frame_values:
            if not isinstance(item, list):
                item = list(item)
            quat_frame_value_array += item
        assert isinstance(quat_frame_value_array, list) and len(
            quat_frame_value_array) == n_dims, \
            ('The length of quaternion frame is not correct! ')
        return quat_frame_value_array

    @classmethod
    def check_quat(cls, test_quat, ref_quat):
        """check test quat needs to be filpped or not
        """
        test_quat = np.asarray(test_quat)
        ref_quat = np.asarray(ref_quat)
        dot = np.dot(test_quat, ref_quat)
        if dot < 0:
            test_quat = - test_quat
        return test_quat.tolist()

    def scale_rootchannels(self):
        """ Scale all root channels in the given frames.
        It scales the root channel by taking its absolut maximum
        (max_x, max_y, max_z) and devide all values by the maximum,
        scaling all positions between -1 and 1
        """
        max_x = 0
        max_y = 0
        max_z = 0
        for key, value in self.quat_frames.items():
            tmp = np.asarray(value)
            max_x_i = np.max(np.abs(tmp[:, 0]))
            max_y_i = np.max(np.abs(tmp[:, 1]))
            max_z_i = np.max(np.abs(tmp[:, 2]))
            if max_x < max_x_i:
                max_x = max_x_i
            if max_y < max_y_i:
                max_y = max_y_i
            if max_z < max_z_i:
                max_z = max_z_i

        # max_x = 1.0
        # max_y = 1.0
        # max_z = 1.0
        for key, value in self.quat_frames.items():
            value = np.array(value)
            value[:, 0] /= max_x
            value[:, 1] /= max_y
            value[:, 2] /= max_z
            self.rescaled_quat_frames[key] = value.tolist()
        self.fdata['scale_vector'] = [max_x, max_y, max_z]

    def gen_data_for_modeling(self):
        print("fpca on temporal parameters")
        self.use_fpca_on_temporal_params()
        print('fpca on spatial parameters')
        self.use_fpca_on_spatial_params()
        self.fdata['spatial_parameters'] = self.fpca_spatial.fpcaobj.low_vecs
        self.fdata['file_order'] = self.fpca_spatial.fileorder
        self.fdata['spatial_eigenvectors'] = self.fpca_spatial.fpcaobj.eigenvectors
        self.fdata['mean_motion'] = self.fpca_spatial.fpcaobj.mean
        self.fdata['temporal_pcaobj'] = self.fpca_temporal.temporal_pcaobj

    def gen_spatial_data_for_modeling(self):
        print('fpca on spatial parameters')
        self.use_fpca_on_spatial_params()
        self.fdata['spatial_parameters'] = self.fpca_spatial.fpcaobj.low_vecs
        self.fdata['file_order'] = self.fpca_spatial.fileorder
        self.fdata['spatial_eigenvectors'] = self.fpca_spatial.fpcaobj.eigenvectors
        self.fdata['mean_motion'] = self.fpca_spatial.fpcaobj.mean

    @staticmethod
    def quat_frame_subtraction(frame_a, frame_b):
        """
        frame_a minus frame_b using quaternion division
        :param frame_a (numpy.array):  the first part is root translation
        :param frame_b (numpy.array):  the first part is root translation
        :return:
        """
        assert len(frame_a) == len(frame_b), ('The frames should have the same length!')
        assert (len(frame_a) - LEN_ROOT_POSITION)%LEN_QUATERNION == 0
        n_quats = (len(frame_a) - LEN_ROOT_POSITION)/LEN_QUATERNION
        res = np.zeros(len(frame_a))
        res[:LEN_ROOT_POSITION] = frame_a[:LEN_ROOT_POSITION] - frame_b[:LEN_ROOT_POSITION]
        for i in range(n_quats):
            res[LEN_ROOT_POSITION + i*LEN_QUATERNION : LEN_ROOT_POSITION + (i+1)*LEN_QUATERNION] = \
                diff_quat(frame_a[LEN_ROOT_POSITION + i*LEN_QUATERNION : LEN_ROOT_POSITION + (i+1)*LEN_QUATERNION],
                          frame_b[LEN_ROOT_POSITION + i*LEN_QUATERNION : LEN_ROOT_POSITION + (i+1)*LEN_QUATERNION])
        return res

    @staticmethod
    def quat_frame_addition(frame_a, frame_b):
        assert len(frame_a) == len(frame_b), ('The frames should have the same length!')
        assert (len(frame_a) - LEN_ROOT_POSITION)%LEN_QUATERNION == 0
        n_quats = (len(frame_a) - LEN_ROOT_POSITION)/LEN_QUATERNION
        res = np.zeros(len(frame_a))
        res[:LEN_ROOT_POSITION] = frame_a[: LEN_ROOT_POSITION] + frame_b[: LEN_ROOT_POSITION]
        for i in range(n_quats):
            res[LEN_ROOT_POSITION + i*LEN_QUATERNION : LEN_ROOT_POSITION + (i+1)*LEN_QUATERNION] = \
                quaternion_multiply(frame_a[LEN_ROOT_POSITION + i*LEN_QUATERNION : LEN_ROOT_POSITION + (i+1)*LEN_QUATERNION],
                                    frame_b[LEN_ROOT_POSITION + i*LEN_QUATERNION : LEN_ROOT_POSITION + (i+1)*LEN_QUATERNION])
        return res

    def convert_data_to_exp_map(self):
        self.convert_euler_to_quat()
        smoothed_quat_frames = self.smooth_quat_frames(self.quat_frames)
        mean_motion_vector = MotionDimensionReduction.cal_mean_motion()

    @staticmethod
    def convert_data_to_logmap(quat_motions):
        """

        :param quat_frames numpy.array<2d>: first three elements of each frame is root translation
        :return:
        """
        res = {}
        for key, value in quat_motions.items():
            res[key] = MotionDimensionReduction.logmap_quat_motion(value).tolist()
        return res

    @staticmethod
    def logmap_quat_motion(quat_frames):
        res = []
        for i in range(len(quat_frames)):
            res.append(MotionDimensionReduction.logmap_quat_frame(quat_frames[i]))
        return np.asarray(res)

    @staticmethod
    def logmap_quat_frame(quat_frame):
        assert (len(quat_frame) - LEN_ROOT_POSITION)%LEN_QUATERNION == 0
        n_quats = (len(quat_frame) - LEN_ROOT_POSITION)/LEN_QUATERNION
        res = np.zeros(LEN_ROOT_POSITION + n_quats * LEN_LOGMAP)
        res[:LEN_ROOT_POSITION] = quat_frame[:LEN_ROOT_POSITION]
        for i in range(n_quats):
            res[LEN_ROOT_POSITION + i * LEN_LOGMAP : LEN_ROOT_POSITION + (i + 1) * LEN_LOGMAP] =\
              quat_to_logmap(quat_frame[LEN_ROOT_POSITION + i * LEN_QUATERNION : LEN_ROOT_POSITION + (i + 1) * LEN_QUATERNION])
        return res

    @staticmethod
    def cal_mean_motion(smoothed_quat_frames):
        quat_data = np.asarray(list(smoothed_quat_frames.values()))

    @staticmethod
    def logmap_to_quat_frame(logmap_vec):
        """
        Map a logmap frame back to quaternion
        :param logmap_vec: first three elements are root translation
        :return: quaternion frame
        """
        assert (len(logmap_vec) - LEN_ROOT_POSITION)%LEN_LOGMAP == 0
        n_quats = (len(logmap_vec) - LEN_ROOT_POSITION)/LEN_LOGMAP
        quat_frame = np.zeros(LEN_ROOT_POSITION + n_quats * LEN_QUATERNION)
        quat_frame[:LEN_ROOT_POSITION] = logmap_vec[:LEN_ROOT_POSITION]
        for i in range(n_quats):
            quat_frame[LEN_ROOT_POSITION + i * LEN_QUATERNION : LEN_ROOT_POSITION + (i + 1) * LEN_QUATERNION] =\
            logmap_to_quat(logmap_vec[LEN_ROOT_POSITION + i * LEN_LOGMAP : LEN_ROOT_POSITION + (i + 1) * LEN_LOGMAP])
        return quat_frame

    @staticmethod
    def logmap_to_quat_motion(logmap_mat):
        res = []
        for i in range(len(logmap_mat)):
            res.append(MotionDimensionReduction.logmap_to_quat_frame(logmap_mat[i]))
        return np.asarray(res)

    def save_data(self, save_filename):
        mean_fd = self.fpca_temporal.temporal_pcaobj[self.fpca_temporal.temporal_pcaobj.name.index('meanfd')]
        harms = self.fpca_temporal.temporal_pcaobj[self.fpca_temporal.temporal_pcaobj.name.index('harmonics')]
        output_data = {'n_basis_spatial': self.fdata['n_dim_spatial'],
                       'eigen_vector_spatial': self.fpca_spatial.fpcaobj.eigenvectors.tolist(),
                       'translation_maxima': self.fdata['scale_vector'],
                       'mean_vector_spatial': self.fpca_spatial.fpcaobj.mean.tolist(),
                       'spatial_params': self.fpca_spatial.fpcaobj.low_vecs,
                       'motion_type': self.fdata['motion_type'],
                       'n_canonical_frames': self.fdata['n_frames'],
                       'eigen_vector_temopral': harms[harms.names.index('coefs')]}
        with open(save_filename, 'w') as outfile:
            json.dump(output_data, outfile)
