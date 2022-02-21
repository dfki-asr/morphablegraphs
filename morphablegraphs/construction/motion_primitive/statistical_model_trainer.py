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
Created on Fri Jan 23 09:34:02 2015

@author: Han Du
"""
import json
import numpy as np
import os
from anim_utils.animation_data.bvh import BVHReader, BVHWriter
from .gmm_trainer import GMMTrainer
from ..utils import get_cubic_b_spline_knots


class StatisticalModelTrainer(GMMTrainer):

    def __init__(self, fdata=None, save_path=None, temporal_data=True):
        super(StatisticalModelTrainer, self).__init__()
        self.use_semantic_annotation = False
        self.use_temporal_data = temporal_data
        self.semantic_eigenvectors = None
        self.semantic_npc = None
        self.semantic_mean = None
        self.semantic_weight = None
        if fdata is not None:
            if self.use_temporal_data:
                self._load_data(fdata)
                self._combine_spatial_temporal_parameters()
            else:
                self._load_spatial_data(fdata)
        self.save_path = save_path

    def gen_motion_primitive_model(self):
        self.fit(self._motion_parameters)
        self._save_model(self.save_path)
        return self.data

    def gen_motion_primitive_model_without_semantic(self, score='AIC'):
        self.fit(self._motion_parameters, score)
        if self.use_temporal_data:
            self._save_model_without_semantic(self.save_path)
        else:
            self._save_spatial_model(self.save_path)
        print(("Export motion primitive to " + self.save_path))

    def export_model_data(self, save_path):
        # save the intermediate data
        save_data = {}
        motion_data = {}
        assert len(self._file_order) == len(self._motion_parameters)
        harms = self._temporal_pca[self._temporal_pca.names.index('harmonics')]
        for i in range(len(self._file_order)):
            motion_data[self._file_order[i]] = self._motion_parameters[i].tolist()
        save_data['motion_data'] = motion_data
        save_data['motion_type'] = self._motion_primitive_name
        save_data['eigen_vector_spatial'] = self._spatial_eigenvectors.tolist()
        save_data['eigen_vector_temporal'] = np.array(harms[harms.names.index('coefs')]).tolist()
        save_data['translation_maxima'] = self._scale_vec
        save_data['n_basis_spatial'] = self._n_basis
        save_data['n_canonical_frames'] = self._n_frames
        save_data['n_basis_temporal'] = 8  # remove magic value
        with open(save_path, 'wb') as outfile:
            json.dump(save_data, outfile)

    def _load_data(self, fdata):
        '''
        Load dimensional representation for motion segements from a json file

        Parameters
        ----------
        * data: json file
        \tThe data is stored in a dictionary
        '''

        self._motion_primitive_name = fdata['motion_type']
        self._spatial_parameters = fdata['spatial_parameters']
        self._spatial_eigenvectors = fdata['spatial_eigenvectors']
        self._n_frames = int(fdata['n_frames'])
        self._scale_vec = fdata['scale_vector']
        self._n_basis = fdata['n_basis']
        self._file_order = fdata['file_order']
        self._mean_motion = fdata['mean_motion']
        self._n_dim_spatial = int(fdata['n_dim_spatial'])
        self._temporal_pca = fdata['temporal_pcaobj']
        self._temporal_parameters = np.asarray(
            self._temporal_pca[self._temporal_pca.names.index('scores')])

    def _load_spatial_data(self, fdata):
        self._motion_primitive_name = fdata['motion_type']
        self._motion_parameters = fdata['spatial_parameters']
        self._spatial_eigenvectors = fdata['spatial_eigenvectors']
        self._n_frames = int(fdata['n_frames'])
        self._scale_vec = fdata['scale_vector']
        self._n_basis = fdata['n_basis']
        self._file_order = fdata['file_order']
        self._mean_motion = fdata['mean_motion']
        self._n_dim_spatial = int(fdata['n_dim_spatial'])

    def _weight_temporal_parameters(self):
        '''
        Weight low dimensional temporal parameters
        '''
        weight_matrix = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        self._temporal_parameters = np.dot(self._temporal_parameters,
                                           weight_matrix)

    def _combine_spatial_temporal_parameters(self):
        '''
        Concatenate temporal and spatial paramters of same motion sample as a 
        long vector
        '''
        assert self._spatial_parameters.shape[0] == \
            self._temporal_parameters.shape[
                0], ('Number of samples are not the same for spatial parameters and temporal parameters')
        self._weight_temporal_parameters()
        self._motion_parameters = np.concatenate((self._spatial_parameters,
                                                  self._temporal_parameters,),
                                                 axis=1)

    def load_data_from_file(self, spatial_temporal_data, temporal_semantic_file):
        # with open(spatial_temporal_file, 'r') as infile:
        #     spatial_temporal_data = json.load(infile)
        with open(temporal_semantic_file, 'r') as infile:
            temporal_semantic_data = json.load(infile)
        # self._motion_parameters = np.array(spatial_temporal_data['motion_data'])
        self.use_semantic_annotation = True
        motion_data = spatial_temporal_data['motion_data']
        self._n_frames = spatial_temporal_data['n_canonical_frames']
        self._scale_vec = spatial_temporal_data['translation_maxima']
        self._n_basis = spatial_temporal_data['n_basis_spatial']
        self._n_dim_spatial = spatial_temporal_data['n_dim_spatial']
        # self._mean_time_vector = np.array(spatial_temporal_data['mean_time_vector'])
        self._motion_primitive_name = spatial_temporal_data['motion_type']
        self._spatial_eigenvectors = np.array(spatial_temporal_data['eigen_vector_spatial'])
        # self._eigen_vectors_time = np.array(spatial_temporal_data['eigen_vector_temporal'])
        self._mean_motion = np.array(spatial_temporal_data['mean_spatial_vector'])

        # loading temporal and semantic data
        self.temporal_semantic_npc = temporal_semantic_data['n_pc']
        self.n_dim_temporal_semantic = temporal_semantic_data['n_dim']
        self._n_basis_temporal_semantic = temporal_semantic_data['n_basis']
        self.temporal_semantic_lowVs = np.array(temporal_semantic_data['low_dimensional_data'])
        self.temporal_semantic_fileorder = temporal_semantic_data['fileorder']
        self.temporal_semantic_eigenvectors = np.array(temporal_semantic_data["eigen_vector"])
        self.temporal_semantic_mean = temporal_semantic_data['mean_vec']
        self.combine_motion_and_temporal_semantic_data(motion_data, self.temporal_semantic_lowVs,
                                                       self.temporal_semantic_fileorder)
        self.semantic_annotation = temporal_semantic_data["semantic_annotation"]

    def combine_motion_and_temporal_semantic_data(self, motion_data, temproal_semantic_data, fileorder):
        self._motion_parameters = []
        for i in range(len(fileorder)):
            if fileorder[i] not in list(motion_data.keys()):
                print((fileorder[i] + ' is not in motion data!'))
                continue
            tmp_spatial_data = motion_data[fileorder[i]][:-3]
            self._motion_parameters.append(np.concatenate((tmp_spatial_data, temproal_semantic_data[i])))
        self._motion_parameters = np.asarray(self._motion_parameters)

    def weight_temporal_semantic_data(self, factor):
        self.temporal_semantic_eigenvectors *= factor

    def _sample_spatial_parameters(self, n, save_path=None):
        '''Generate ranmdon sample from mrophable model based on spatial p
           parameters
        '''
        self.new_ld_vectors = self.gmm.sample(n)
        for i in range(n):
            filename = 'generated_motions' + os.sep + str(i) + '.bvh'
            self._backprojection(self.new_ld_vectors[i], filename=filename)

    def _sample_fd_spatial_parameters(self, n, save_path=None):
        self.new_fd_ld_vectors = self.gmm.sample(n)

    def _backprojection(self, ld_vec, filename=None):
        """Back project a low dimensional spatial parameter to motion
        """
        eigenVectors = np.array(self._spatial_eigenvectors)
        backprojected_vector = np.dot(np.transpose(eigenVectors), ld_vec.T)
        backprojected_vector = np.ravel(backprojected_vector)
        backprojected_vector += self._mean_motion
        # reshape motion vector as a 2d array n_frames * n_dim
        assert len(backprojected_vector) == self._n_dim_spatial * \
            self._n_frames, (
                'the length of back projected motion vector is not correct!')
        if filename is None:
            filename = 'sample.bvh'
        else:
            filename = filename
        frames = np.reshape(backprojected_vector, (self._n_frames,
                                                   self._n_dim_spatial))
        # rescale root position for each frame
        for i in range(self._n_frames):
            frames[i, 0] = frames[i, 0] * self._scale_vec[0]
            frames[i, 1] = frames[i, 1] * self._scale_vec[1]
            frames[i, 2] = frames[i, 2] * self._scale_vec[2]
        skeleton = os.sep.join(('lib', 'skeleton.bvh'))
        reader = BVHReader(skeleton)
        BVHWriter(filename, reader, frames, frame_time=0.013889,
                  is_quaternion=True)

    def save_low_level_semantic_model(self):
        pass

    def _save_model(self, save_path=None):
        '''
        Save model as a json file

        Parameters
        ----------
        *filename: string
        \tGive the file name to json file
        '''
        elementary_action_name = self._motion_primitive_name.split('_')[0]
        if save_path is None:
            filename = self._motion_primitive_name + '_quaternion_mm.json'
        else:
            if not save_path.endswith(os.sep):
                save_path += os.sep
            folder_path = save_path + 'elementary_action_%s' % (elementary_action_name)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            filename = folder_path + os.sep + self._motion_primitive_name + '_quaternion_mm_with_semantic.json'
        weights = self.gmm.weights_.tolist()
        means = self.gmm.means_.tolist()
        covars = self.gmm.covariances_.tolist()
        if not self.use_semantic_annotation:
            mean_fd = self._temporal_pca[self._temporal_pca.names.index('meanfd')]
            self._mean_time_vector = np.array(
                mean_fd[mean_fd.names.index('coefs')])
            self._mean_time_vector = np.ravel(self._mean_time_vector)
            harms = self._temporal_pca[self._temporal_pca.names.index('harmonics')]
            # self._eigen_vectors_time = np.array(harms[harms.names.index('coefs')])

        self.data = {'name': self._motion_primitive_name,
                'gmm_weights': weights,
                'gmm_means': means,
                'gmm_covars': covars,
                'eigen_vectors_spatial': self._spatial_eigenvectors.tolist(),
                'mean_spatial_vector': self._mean_motion.tolist(),
                'n_canonical_frames': self._n_frames,
                'translation_maxima': self._scale_vec,
                'n_basis_spatial': self._n_basis,
                'npc_spatial': len(self._spatial_eigenvectors),
                'eigen_vectors_temporal_semantic': self.temporal_semantic_eigenvectors.tolist(),
                'mean_temporal_semantic_vector': self.temporal_semantic_mean,
                'n_dim_spatial': self._n_dim_spatial,
                'n_basis_temporal_semantic': self._n_basis_temporal_semantic,
                'b_spline_knots_spatial': get_cubic_b_spline_knots(self._n_basis, self._n_frames).tolist(),
                'b_spline_knots_temporal_semantic': get_cubic_b_spline_knots(self._n_basis_temporal_semantic,
                                                                            self._n_frames).tolist(),
                'npc_temporal_semantic': self.temporal_semantic_npc,
                'semantic_annotation': self.semantic_annotation,
                'n_dim_temporal_semantic': self.n_dim_temporal_semantic}
        with open(filename, 'wb') as outfile:
            json.dump(self.data, outfile)
        outfile.close()

    def _save_spatial_model(self, save_path=None):
        '''
        Save model as a json file

        Parameters
        ----------
        *filename: string
        \tGive the file name to json file
        '''
        elementary_action_name = self._motion_primitive_name.split('_')[0]
        if save_path is None:
            filename = self._motion_primitive_name + '_quaternion_spatial_mm.json'
        else:
            if not save_path.endswith(os.sep):
                save_path += os.sep
            folder_path = save_path + 'elementary_action_%s' % (elementary_action_name)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            filename = folder_path + os.sep + self._motion_primitive_name + '_quaternion_spatial_mm.json'
        weights = self.gmm.weights_.tolist()
        means = self.gmm.means_.tolist()
        covars = self.gmm.covariances_.tolist()

        data = {'name': self._motion_primitive_name,
                'gmm_weights': weights,
                'gmm_means': means,
                'gmm_covars': covars,
                'eigen_vectors_spatial': self._spatial_eigenvectors.tolist(),
                'mean_spatial_vector': self._mean_motion.tolist(),
                'n_canonical_frames': self._n_frames,
                'translation_maxima': self._scale_vec,
                'n_basis_spatial': self._n_basis,
                'n_dim_spatial': self._n_dim_spatial,
                'b_spline_knots_spatial': get_cubic_b_spline_knots(self._n_basis, self._n_frames).tolist()}
        with open(filename, 'wb') as outfile:
            json.dump(data, outfile)
        outfile.close()

    def _save_model_without_semantic(self, save_path=None):
        '''
        Save model as a json file

        Parameters
        ----------
        *filename: string
        \tGive the file name to json file
        '''
        elementary_action_name = self._motion_primitive_name.split('_')[0]
        if save_path is None:
            filename = self._motion_primitive_name + '_quaternion_mm.json'
        else:
            if not save_path.endswith(os.sep):
                save_path += os.sep
            folder_path = save_path + 'elementary_action_%s' % (elementary_action_name)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            filename = folder_path + os.sep + self._motion_primitive_name + '_quaternion_mm.json'
        weights = self.gmm.weights_.tolist()
        means = self.gmm.means_.tolist()
        covars = self.gmm.covariances_.tolist()
        mean_fd = self._temporal_pca[self._temporal_pca.names.index('meanfd')]
        self._mean_time_vector = np.array(
            mean_fd[mean_fd.names.index('coefs')])
        self._mean_time_vector = np.ravel(self._mean_time_vector)
        n_basis_time = len(self._mean_time_vector)
        harms = self._temporal_pca[self._temporal_pca.names.index('harmonics')]
        self._eigen_vectors_time = np.array(harms[harms.names.index('coefs')])

        data = {'name': self._motion_primitive_name,
                'gmm_weights': weights,
                'gmm_means': means,
                'gmm_covars': covars,
                'eigen_vectors_spatial': self._spatial_eigenvectors.tolist(),
                'mean_spatial_vector': self._mean_motion.tolist(),
                'n_canonical_frames': self._n_frames,
                'translation_maxima': self._scale_vec,
                'n_basis_spatial': self._n_basis,
                'eigen_vectors_time': self._eigen_vectors_time.tolist(),
                'mean_time_vector': self._mean_time_vector.tolist(),
                'n_dim_spatial': self._n_dim_spatial,
                'n_basis_time': n_basis_time,
                'b_spline_knots_spatial': get_cubic_b_spline_knots(self._n_basis, self._n_frames).tolist(),
                'b_spline_knots_time': get_cubic_b_spline_knots(n_basis_time, self._n_frames).tolist()}
        with open(filename, 'wb') as outfile:
            json.dump(data, outfile)
        outfile.close()
