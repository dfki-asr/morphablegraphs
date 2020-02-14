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
import numpy as np
from anim_utils.utilities.io_helper_functions import write_to_json_file, load_json_file
from ..utils import get_cubic_b_spline_knots, get_data_analysis_folder
from ...motion_analysis.prepare_data import get_smoothed_quat_frames, create_quat_functional_data, \
                                                         reshape_data_for_PCA, scale_root_channels
from ..fpca.scaled_fpca import ScaledFunctionalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from . import LEN_QUATERNION, LEN_ROOT, LEN_EULER
DIMENSION_REDUCTION_METHODS = ['pca', 'lle', 'iosmap', 'GPLVM']
DATA_TYPE = ['discrete', 'functional']
SCALING = ['normalization', 'scaled_root', 'normal', 'scaled']
PARAMETERIZATION = ['quaternion', 'Euler']


class MotionDimensionReductor(object):

    def __init__(self, elementary_action, motion_primitive, data_repo_path, data_type='discrete', scaling='normal',
                 method='pca', parameterization='quaternion'):
        self.elementary_action = elementary_action
        self.motion_primitive = motion_primitive
        self.data_repo_path = data_repo_path
        self.data_type = data_type
        self.scaling = scaling
        self.method = method
        self.parameterization = parameterization

    def set_scaling(self, scaling_method):
        if not scaling_method in SCALING:
            raise KeyError(scaling_method + ' is not supported.')
        self.scaling = scaling_method

    def set_data_type(self, data_type):
        if not data_type in DATA_TYPE:
            raise KeyError(data_type + ' is not supported.')
        self.data_type = data_type

    def set_dimension_reduction_method(self, method):
        if not method in DIMENSION_REDUCTION_METHODS:
            raise KeyError(method + ' is not supported.')
        self.method = method

    def set_parameterization_method(self, method):
        if not method in PARAMETERIZATION:
            raise KeyError(method + ' is not supported.')
        self.parameterization = method

    def load_data(self, n_basis=None):
        '''
        load the data for specified setting, the data will be loaded from data analysis folder if it is precomputed and
        stored, otherwise it will be loaded directly from alignment folder
        :return:
        '''
        data_analysis_folder = get_data_analysis_folder(self.elementary_action,
                                                        self.motion_primitive,
                                                        self.data_repo_path)
        if self.data_type == 'discrete':
            if self.parameterization == 'quaternion':
                prestored_filename = os.path.join(data_analysis_folder, 'smoothed_quat_frames.json')
                if not os.path.isfile(prestored_filename):
                    discrete_motion_data = get_smoothed_quat_frames(self.elementary_action,
                                                                    self.motion_primitive,
                                                                    self.data_repo_path)

                    write_to_json_file(prestored_filename, discrete_motion_data)
                else:
                    discrete_motion_data = load_json_file(prestored_filename)
                return discrete_motion_data
            elif self.parameterization == 'Euler':
                raise NotImplementedError
            else:
                raise KeyError('Motion Parameterization type is not supported')
        elif self.data_type == 'functional':
            if self.parameterization == 'quaternion':
                prestored_filename = os.path.join(data_analysis_folder, 'functional_quat_data.json')
                if not os.path.isfile(prestored_filename):
                    functional_quat_data = create_quat_functional_data(self.elementary_action,
                                                                       self.motion_primitive,
                                                                       self.data_repo_path,
                                                                       n_basis)
                    write_to_json_file(prestored_filename, functional_quat_data)
                else:
                    functional_quat_data = load_json_file(prestored_filename)
                return functional_quat_data
            elif self.parameterization == 'Euler':
                raise NotImplementedError
            else:
                raise KeyError('Motion Parameterization type is not supported')

    def load_discrete_data(self):
        data_analysis_folder = get_data_analysis_folder(self.elementary_action,
                                                        self.motion_primitive,
                                                        self.data_repo_path)
        if self.parameterization == 'quaternion':
            prestored_filename = os.path.join(data_analysis_folder, 'smoothed_quat_frames.json')
            if not os.path.isfile(prestored_filename):
                discrete_motion_data = get_smoothed_quat_frames(self.elementary_action,
                                                                self.motion_primitive,
                                                                self.data_repo_path)

                write_to_json_file(prestored_filename, discrete_motion_data)
            else:
                discrete_motion_data = load_json_file(prestored_filename)
        elif self.parameterization == 'Euler':
            raise NotImplementedError
        else:
            raise KeyError('Motion Parameterization type is not supported')
        return discrete_motion_data

    def load_functional_data(self, n_basis):
        data_analysis_folder = get_data_analysis_folder(self.elementary_action,
                                                        self.motion_primitive,
                                                        self.data_repo_path)
        if self.parameterization == 'quaternion':
            prestored_filename = os.path.join(data_analysis_folder, 'functional_quat_data.json')
            if not os.path.isfile(prestored_filename):
                functional_quat_data = create_quat_functional_data(self.elementary_action,
                                                                   self.motion_primitive,
                                                                   self.data_repo_path,
                                                                   n_basis)
                write_to_json_file(prestored_filename, functional_quat_data)
            else:
                functional_quat_data = load_json_file(prestored_filename)
        elif self.parameterization == 'Euler':
            raise NotImplementedError
        else:
            raise KeyError('Motion Parameterization type is not supported')
        return functional_quat_data

    def scale_data(self, motion_data_matrix):
        '''
        scale motion data and reshape 3d data to 2d for dimension reduction
        :param motion_data_matrix (numpy.narray<3d>): n_samples * n_frames(n_knots) * n.dims
        :return (numpy.narray<3d>): n_samples * (n_frames * n_dims)
        '''
        assert len(motion_data_matrix.shape) == 3
        if self.scaling == 'normalization':
            motion_data_2d = reshape_data_for_PCA(motion_data_matrix)
            scaler = StandardScaler().fit(motion_data_2d)
            normalized_motion_data_2d = scaler.transform(motion_data_2d)
            scale_params = {'mean': scaler.mean_.tolist(),
                            'std': scaler.std_.tolist()}
            return normalized_motion_data_2d, scale_params
        elif self.scaling == 'scaled_root':
            scaled_root_motion_data, scale_root_vector = scale_root_channels(motion_data_matrix)
            scale_params = {'translation_maxima': scale_root_vector}
            scaled_root_motion_data_2d = reshape_data_for_PCA(scaled_root_motion_data)
            return scaled_root_motion_data_2d, scale_params
        elif self.scaling == 'scaled':
            data_analysis_folder = get_data_analysis_folder(self.elementary_action,
                                                            self.motion_primitive,
                                                            self.data_repo_path)
            optimized_weights_filename = os.path.join(data_analysis_folder, '_'.join([self.elementary_action,
                                                                                      self.motion_primitive,
                                                                                      'optimization',
                                                                                      'result.json']))
            if not os.path.isfile(optimized_weights_filename):
                raise IOError('Cannot find weight file for scaling features')
            else:
                optimal_weights_dic = load_json_file(optimized_weights_filename)
        else:
            return motion_data_matrix, None

    def normalize_motion_data(self, motion_data_matrix):
        assert len(motion_data_matrix.shape) == 3
        motion_data_2d = reshape_data_for_PCA(motion_data_matrix)
        scaler = StandardScaler().fit(motion_data_2d)
        normalized_motion_data_2d = scaler.transform(motion_data_2d)
        scale_params = {'method': 'normalization',
                        'mean': scaler.mean_.tolist(),
                        'std': scaler.std_.tolist()}
        return normalized_motion_data_2d, scale_params

    def scale_motion_data_root(self, motion_data_matrix):
        assert len(motion_data_matrix.shape) == 3
        scaled_root_motion_data, scale_root_vector = scale_root_channels(motion_data_matrix)
        scale_params = {'method': 'scaled_root', 'translation_maxima': scale_root_vector}
        scaled_root_motion_data_2d = reshape_data_for_PCA(scaled_root_motion_data)
        return scaled_root_motion_data_2d, scale_params

    def optimize_feature_weights(self, motion_data_matrix, n_low_dims, knots):
        assert len(motion_data_matrix.shape) == 3
        n_samples, n_frames, n_dims = motion_data_matrix.shape
        if self.parameterization == 'quaternion':
            n_joints = (n_dims - LEN_ROOT)/LEN_QUATERNION
        else:
            n_joints = (n_dims - LEN_ROOT)/LEN_EULER
        skeleton_json = os.path.join(os.path.dirname(__file__), r'../../../mgrd/data/skeleton.json')
        if not os.path.isfile(skeleton_json):
            raise IOError('cannot find skeleton.json file')
        sfpca = ScaledFunctionalPCA(self.elementary_action,
                                    self.motion_primitive,
                                    self.data_repo_path,
                                    motion_data_matrix,
                                    n_low_dims,
                                    skeleton_json,
                                    knots,
                                    n_joints)
        sfpca.fit()
        return sfpca

    def reduce_dimension(self, n_low_dims, n_basis=None, save_result=True):
        motion_data = self.load_data(n_basis)
        if self.data_type == 'discrete':
            filelist = list(motion_data.keys())
            motion_data_matrix = np.asarray(list(motion_data.values()))
        else:
            knots = motion_data['knots']
            motion_data_matrix = np.asarray(list(motion_data['functional_data'].values()))
            filelist = list(motion_data['functional_data'].keys())
            n_canonical_frames = motion_data['n_canonical_frames']
        # scale motion data
        if self.scaling == 'normalization':
            motion_data_2d, scale_params = self.normalize_motion_data(motion_data_matrix)
        elif self.scaling == 'scaled_root':
            motion_data_2d, scale_params = self.scale_motion_data_root(motion_data_matrix)
        elif self.scaling == 'normal':
            motion_data_2d = reshape_data_for_PCA(motion_data_matrix)
        elif self.scaling == 'scaled':
            if self.data_type == 'discrete':
                raise NotImplementedError
            else:
                sfpca = self.optimize_feature_weights(motion_data_matrix, n_low_dims, knots)
        else:
            raise KeyError(self.scaling + ' is not supported!')

        # apply dimension reduction methods
        if self.method == 'pca':
            if self.scaling != 'scaled':
                pca = PCA(n_components=n_low_dims)
                projections = pca.fit_transform(motion_data_2d)
            else:
                projections = sfpca.transform()
        elif self.method == 'lle':
            raise NotImplementedError
        elif self.method == 'iosmap':
            raise NotImplementedError
        elif self.method == 'GPLVM':
            raise NotImplementedError
        else:
            raise KeyError(self.method + ' is not supported.')

        if save_result:
            data_analysis_folder = get_data_analysis_folder(self.elementary_action,
                                                            self.motion_primitive,
                                                            self.data_repo_path)
            outfile_name = os.path.join(data_analysis_folder, '_'.join([self.scaling,
                                                                        self.method,
                                                                        'latent',
                                                                        'vector',
                                                                        self.data_type,
                                                                        'data',
                                                                        str(n_basis),
                                                                        'dims.json']))
            low_dims_motion_dic = {}
            for i in range(len(filelist)):
                low_dims_motion_dic[filelist[i]] = list(projections[i])

            if self.method == 'pca':
                if self.scaling != 'scaled':
                    variance_vector = list(pca.explained_variance_ratio_)
                else:
                    variance_vector = list(sfpca.pca.explained_variance_ratio_)
                output_data = {'motion_data': low_dims_motion_dic,
                               'variance_vector': variance_vector}
                write_to_json_file(outfile_name, output_data)
            elif self.method == 'lle':
                raise NotImplementedError
            elif self.method == 'iosmap':
                raise NotImplementedError
            elif self.method == 'GPLVM':
                raise NotImplementedError
            else:
                raise KeyError(self.method + ' is not supported.')


    def backproject_motion_data(self, low_dimension_vectors):
        pass
