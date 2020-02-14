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
import glob
import numpy as np
from anim_utils.utilities import get_aligned_data_folder, load_json_file, load_bvh_files_from_folder
from ..preprocessing import create_low_level_semantic_annotation
from ..fpca import FPCATimeSemantic, MotionDimensionReduction, FPCASpatialData
from anim_utils.animation_data.quaternion_frame import convert_euler_frames_to_quaternion_frames


class MotionPrimitiveConstructor(object):

    def __init__(self,
                 elementary_action,
                 motion_primitive):
        self.elementary_action = elementary_action
        self.motion_primitive = motion_primitive
        self.dimension_reduction_method = 'FPCA'

    def config_setting(self, repo_dir, bvhreader, n_basis_spatial=7, n_basis_temporal=8, n_components_spatial=None,
                       precision_spatial=0.95, n_components_temporal=None, precision_temporal=0.99,
                       temporal_semantic_weight=0.1):
        self.repo_dir = repo_dir
        self.skeleton_bvh = bvhreader
        self.n_components_spatial = n_components_spatial
        self.precision_spatial = precision_spatial
        self.n_basis_spatial = n_basis_spatial
        self.n_basis_temporal = n_basis_temporal
        self.n_components_temporal = n_components_temporal
        self.precision_temporal = precision_temporal
        self.temporal_semantic_weight = temporal_semantic_weight

    def load_data_from_aligned_motions(self):
        '''
        Load motion data from aligned bvh files
        :return:
        '''
        aligned_data_folder = get_aligned_data_folder(self.elementary_action,
                                                      self.motion_primitive,
                                                      self.repo_dir)
        self.motion_data_dic = load_bvh_files_from_folder(aligned_data_folder)
        if not os.path.exists(os.path.join(aligned_data_folder, 'timewarping.json')):
            raise ImportError('Cannot find timewarping file!')
        else:
            self.timewarping = load_json_file(os.path.join(aligned_data_folder, 'timewarping.json'))
        semantic_annotation_filename = '_'.join([self.elementary_action, self.motion_primitive, 'semantic',
                                                 'annotation.json'])
        if not os.path.exists(os.path.join(aligned_data_folder, semantic_annotation_filename)):
            create_low_level_semantic_annotation(self.motion_primitive,
                                                 self.elementary_action,
                                                 self.repo_dir)
        else:
            self.semantic_annotation = load_json_file(os.path.join(aligned_data_folder,
                                                                   semantic_annotation_filename))

    def convert_euler_to_quaternion(self):
        motion_data_quaternion_dic = {}
        for filename, frames in self.motion_data_dic.items():
            motion_data_quaternion_dic[filename] = convert_euler_frames_to_quaternion_frames(self.skeleton_bvh,
                                                                                             frames)
        self.motion_data_quaternion_dic = MotionDimensionReduction.smooth_quat_frames(motion_data_quaternion_dic)

    def reduce_spatial_dimensionality(self):
        if self.dimension_reduction_method == 'FPCA':
            self.reduce_dimensionality_using_fpca()
        elif self.dimension_reduction_method == 'normalized FPCA':
            self.reduce_dimensionality_using_normalized_fpca()
        elif self.dimension_reduction_method == 'scaled FPCA':
            self.reduce_dimensionality_using_scaled_fpca()
        else:
            raise NotImplementedError

    def reduce_temporal_semantic_dimensionalty(self):
        self.fpca_temporal_semantic = FPCATimeSemantic(n_basis=self.n_basis_temporal,
                                                       n_components_temporal=self.n_components_temporal,
                                                       precision_temporal=self.precision_temporal)
        self.fpca_temporal_semantic.load_time_warping_data_from_dic(self.timewarping)
        self.fpca_temporal_semantic.load_semantic_annotation_from_dic(self.semantic_annotation)
        self.fpca_temporal_semantic.merge_temporal_semantic_data()
        self.fpca_temporal_semantic.functional_pca()
        print((self.fpca_temporal_semantic.file_order))
        print((self.fpca_temporal_semantic.lowVs.shape))

    def combine_spatial_temporal_semantic_low_dimensional_data(self):
        self.weight_temporal_semantic_data()
        n_samples_spatial, n_components_spatial = self.fpca_spatial.low_vecs.shape
        n_samples_temporal, n_components_temporal = self.fpca_temporal_semantic.shape
        assert n_samples_spatial == n_components_temporal
        combined_low_vecs = np.zeros((n_samples_spatial, n_components_temporal + n_components_spatial))
        spatial_filelist = self.fpca_spatial.fileorder
        temporal_filelist = self.fpca_temporal_semantic.file_order
        for i in range(n_samples_spatial):
            if not spatial_filelist[i] in temporal_filelist:
                raise KeyError('Cannot find ' + spatial_filelist[i] + ' in temporal semantic data!')
            temporal_index = temporal_filelist.index(spatial_filelist[i])
            combined_low_vecs[i] = np.concatenate((self.fpca_spatial.low_vecs[i],
                                                   self.fpca_temporal_semantic.lowVs[temporal_index]))
        self.motion_parameters = combined_low_vecs

    def reduce_dimensionality_using_fpca(self):
        self.fpca_spatial = FPCASpatialData(n_basis=self.n_basis_spatial)
        self.fpca_spatial.fit(np.asarray(list(self.motion_data_quaternion_dic.values())))
        self.fpca_spatial.fileorder = list(self.motion_data_quaternion_dic.keys())
        print((self.fpca_spatial.low_vecs.shape))

    def reduce_dimensionality_using_normalized_fpca(self):
        pass

    def reduce_dimensionality_using_scaled_fpca(self):
        pass

    def weight_temporal_semantic_data(self):
        self.fpca_temporal_semantic.eigenvectors *= self.temporal_semantic_weight