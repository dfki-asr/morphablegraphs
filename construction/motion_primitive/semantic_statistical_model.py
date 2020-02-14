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
import os
import json
from .statistical_model_trainer import StatisticalModelTrainer
from .gmm_trainer import GMMTrainer
from sklearn import mixture
from ..utils import get_cubic_b_spline_knots


class SemanticStatisticalModel(StatisticalModelTrainer):

    def __init__(self, fdata, semantic_label):
        """

        :param fdata:
        :param semantic_label: dictionary contains semantic label matching the filename, and numeric label for data
        :return:
        """
        super(SemanticStatisticalModel, self).__init__(fdata)
        self.semantic_label = semantic_label
        self.classified_data = {}
        self.semantic_classification()

    def semantic_classification(self):

        for key in list(self.semantic_label.keys()):
            indices = [idx for idx, item in enumerate(self._file_order) if key in item]
            self.classified_data[key] = self._motion_parameters[indices]
            if len(self.classified_data[key]) == 0:
                raise KeyError(key + ' is not found in data')

    def create_gaussian_mixture_model(self):
        gmm_models = []
        semantic_labels = []
        class_weights = []
        n_gaussians = 0
        for key in list(self.classified_data.keys()):
            gmm_trainer = GMMTrainer()
            gmm_trainer.fit(self.classified_data[key])
            gmm_models.append(gmm_trainer.gmm)
            semantic_labels.append(self.semantic_label[key])
            class_weights.append(float(len(self.classified_data[key]))/float(len(self._motion_parameters)))
            n_gaussians += gmm_trainer.numberOfGaussian
        self.gmm = mixture.GMM(n_components=n_gaussians, covariance_type='full')
        new_weights = []
        new_means = []
        new_covars = []
        for i in range(len(gmm_models)):
            new_weights.append(class_weights[i] * gmm_models[i].weights_)
            new_means.append(np.concatenate((gmm_models[i].means_,
                                             np.zeros((len(gmm_models[i].means_), 1)) + semantic_labels[i]),
                                            axis=1))
            covars_shape = gmm_models[i].covars_.shape
            new_covar_mat = np.zeros((covars_shape[0], covars_shape[1] + 1, covars_shape[2] + 1))
            for j in range(covars_shape[0]):
                new_covar_mat[j][:-1, :-1] = gmm_models[i].covars_[j]
                new_covar_mat[j][-1, -1] = 1e-5
            new_covars.append(new_covar_mat)
        self.gmm.weights_ = np.concatenate(new_weights)
        self.gmm.means_ = np.concatenate(new_means)
        self.gmm.covars_ = np.concatenate(new_covars)

    def gen_semantic_motion_primitive_model(self, savepath=None):
        self.create_gaussian_mixture_model()
        self._save_model(savepath)

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
            filename = folder_path + os.sep + self._motion_primitive_name + '_quaternion_mm.json'
        weights = self.gmm.weights_.tolist()
        means = self.gmm.means_.tolist()
        covars = self.gmm.covars_.tolist()
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
                'b_spline_knots_time': get_cubic_b_spline_knots(n_basis_time, self._n_frames).tolist(),
                'semantic_label': self.semantic_label}
        with open(filename, 'wb') as outfile:
            json.dump(data, outfile)
        outfile.close()