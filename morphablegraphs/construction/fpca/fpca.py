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
# encoding: UTF-8
import json
import numpy as np
import scipy.interpolate as si
from ..utils import get_cubic_b_spline_knots
import copy
import matplotlib.pyplot as plt
from .utils import center_data, run_pca


class FPCATimeSemantic(object):
    def __init__(self):
        self.temporal_semantic_data = None
        self.temporal_data = None
        self.semantic_data = None
        self.semantic_annotation_list = None

    def load_time_warping_data(self, time_warping_file):
        with open(time_warping_file, 'r') as infile:
            self.temporal_data = json.load(infile)

    def load_semantic_annotation(self, semantic_annotation_file):
        with open(semantic_annotation_file, 'r') as infile:
            semantic_annotation = json.load(infile)
        self.semantic_data = semantic_annotation['data']
        self.semantic_annotation_list = semantic_annotation['annotation_list']

    def merge_temporal_semantic_data(self):
        if self.temporal_data is None or self.semantic_data is None:
            raise ValueError('Load semantic annotation or time warping data first!')
        temporal_semantic_data_dic = {}
        for key, value in list(self.semantic_data.items()):
            temporal_semantic_data_dic[key] = [self.temporal_data[key]]
            for feature, anno in list(value.items()):
                temporal_semantic_data_dic[key].append(anno)
        self.temporal_semantic_data = list(temporal_semantic_data_dic.values())
        self.file_order = list(temporal_semantic_data_dic.keys())

    def z_t_transform(self):
        for i in range(len(self.temporal_semantic_data)):
            monotonic_indices = FPCATimeSemantic._get_monotonic_indices(self.temporal_semantic_data[i][0])
            assert FPCATimeSemantic._is_strict_increasing(monotonic_indices), \
                ("convert %s to monotonic indices failed" % self.file_order[i])
            w_tmp = np.array(monotonic_indices)
            # add one to each entry, because we start with 0
            w_tmp = w_tmp + 1
            w_tmp = np.insert(w_tmp, 0, 0)  # set w(0) to zero

            w_diff = np.diff(w_tmp)
            z_transform = np.log(w_diff)
            self.temporal_semantic_data[i][0] = z_transform

    @classmethod
    def _get_monotonic_indices(cls, indices, epsilon=0.01, delta=0):
        """Return an ajusted set of Frameindices which is strictly monotonic

        Parameters
        ----------
        indices : list
        The Frameindices

        Returns
        -------
        A numpy-Float Array with indices similar to the provided list,
        but enforcing strict monotony
        """
        shifted_indices = np.array(indices, dtype=np.float)
        if shifted_indices[0] == shifted_indices[-1]:
            raise ValueError("First and Last element are equal")

        for i in range(1, len(shifted_indices) - 1):
            if shifted_indices[i] > shifted_indices[i - 1] + delta:
                continue

            while np.allclose(shifted_indices[i], shifted_indices[i - 1]) or \
                    shifted_indices[i] <= shifted_indices[i - 1] + delta:
                shifted_indices[i] = shifted_indices[i] + epsilon

        for i in range(len(indices) - 2, 0, -1):
            if shifted_indices[i] + delta < shifted_indices[i + 1]:
                break

            while np.allclose(shifted_indices[i], shifted_indices[i + 1]) or \
                    shifted_indices[i] + delta >= shifted_indices[i + 1]:
                shifted_indices[i] = shifted_indices[i] - epsilon

        return shifted_indices

    @classmethod
    def _is_strict_increasing(cls, indices):
        """ Check wether the indices are strictly increasing ore not

        Parameters
        ----------
        indices : list
        The Frameindices

        Returns
        -------
        boolean
        """
        for i in range(1, len(indices)):
            if np.allclose(indices[i], indices[i - 1]) or indices[i] < indices[i - 1]:
                return False
        return True

    def functional_data_representation(self):
        # convert temporal_semantic_data to functional data
        self.fpca_data = []
        n_canonical_frame = len(self.temporal_semantic_data[0][0])
        n_basis_funcs = 8
        knots = get_cubic_b_spline_knots(n_basis_funcs, n_canonical_frame)

        time_vec = list(range(n_canonical_frame))
        for value in self.temporal_semantic_data:
            coeff_vec = []
            for item in value:
                tck = si.splrep(time_vec, item, t=knots[4:-4])
                coeff_vec.append(tck[1][:-4])
            self.fpca_data.append(np.ravel(coeff_vec))
        self.fpca_data = np.asarray(self.fpca_data)

    def functional_pca(self):
        self.z_t_transform()
        self.functional_data_representation()
        self.fpca_data, self.mean_vec, std = center_data(self.fpca_data)
        Vt, npc = run_pca(self.fpca_data, fraction=0.95)
        self.eigenvectors = Vt[:npc]
        print('number of eigenvectors: ' + str(npc))
        self.lowVs = self.project_data(self.fpca_data)
        self.npc = npc

    def project_data(self, data):
        low_vecs = []
        for i in range(len(data)):
            low_vec = np.dot(self.eigenvectors, data[i])
            low_vecs.append(low_vec)
        low_vecs = np.asarray(low_vecs)
        return low_vecs

    def save_data(self, save_file):
        semantic_data = {'fileorder': self.file_order,
                         'low_dimensional_data': self.lowVs.tolist(),
                         'eigen_vector': self.eigenvectors.tolist(),
                         'mean_vec': self.mean_vec.tolist(),
                         'n_basis': 8,
                         'n_pc': int(self.npc),
                         'n_dim': len(self.semantic_annotation_list)+1,
                         'semantic_annotation': self.semantic_annotation_list}
        with open(save_file, 'w') as outfile:
            json.dump(semantic_data, outfile)


class NormalPCA(object):
    def __init__(self, data, scale=False, copy=True):
        self.data = np.asarray(data)
        assert len(self.data.shape) == 2
        self.n_samples, self.n_frames = self.data.shape

    def apply_pca(self, npc):
        tmp = copy.deepcopy(self.data)
        tmp, self.mean, std = center_data(tmp)
        Vt, npc = run_pca(tmp)
        self.eigenvectors = Vt[:npc]
        self.n_pcs = npc
        self.lowVs = np.transpose(np.dot(self.eigenvectors, np.transpose(tmp)))

    def backproect_data(self):
        self.backprojection = np.dot(self.lowVs, self.eigenvectors)
        self.backprojection += self.mean

    def evaluate_backprojection(self):
        error = 0
        for i in range(self.n_samples):
            error += np.linalg.norm(self.backprojection[i] - self.data[i])
        print(('backprojection error is: ', error/self.n_samples))

    def evaluate_backprojection_per_frame(self):
        errors = np.zeros(self.n_frames)
        for i in range(self.n_frames):
            errors[i] = np.linalg.norm(self.backprojection[:, i] - self.data[:, i])/self.n_samples
        return errors


class FunctionalPCA(object):
    def __init__(self, data, n_knots):
        self.data = np.asarray(data)
        assert len(data.shape) == 2
        self.functional_coeffs = None
        self.n_samples, self.n_frames = self.data.shape
        self.knots = get_cubic_b_spline_knots(n_knots, self.n_frames)

    def set_knots(self, knots):
        self.knots = knots

    def convert_functional_data(self):
        self.functional_coeffs = []
        for i in range(self.n_samples):
            tck = si.splrep(list(range(self.n_frames)), self.data[i], t=self.knots[4: -4])
            self.functional_coeffs.append(tck[1][:-4])
        self.functional_coeffs = np.asarray(self.functional_coeffs)

    def plot_coeffs(self):
        fig = plt.figure()
        for i in range(self.n_samples):
            plt.plot(self.functional_coeffs[i])
        plt.show()

    def evaluate_fucntional_data(self):
        self.evalutation_datamat = np.zeros((self.n_samples, self.n_frames))
        for i in range(self.n_samples):
            self.evalutation_datamat[i] = si.splev(list(range(self.n_frames)),
                                                   (self.knots, self.functional_coeffs[i], 3))
        error = 0
        for i in range(self.n_samples):
            error += np.linalg.norm(self.data[i] - self.evalutation_datamat[i])
        print(('functional data representation error is: ', error/self.n_samples))

    def apply_pca_on_smoothed_data(self, npc):
        print('pca on smoothed data')
        tmp = copy.deepcopy(self.evalutation_datamat)
        tmp, mean, std = center_data(tmp)
        Vt, npc = run_pca(tmp)
        eigenvectors = Vt[:npc]
        #print(('fraction on variance kept: ', pcaobj.sumvariance[npc]))
        lowVs = np.transpose(np.dot(eigenvectors, np.transpose(tmp)))

        backprojection = np.dot(lowVs, eigenvectors)
        backprojection += mean
        error = 0
        for i in range(len(backprojection)):
            error += np.linalg.norm(self.evalutation_datamat[i] - backprojection[i])
        print(("reconstruction error of pca on smoothed data is: ", error/len(backprojection)))

    def apply_pca(self, npc):
        print('pca on coeffs')
        self.functional_coeffs, self.mean, std = center_data(self.functional_coeffs)
        Vt, npc = run_pca(self.functional_coeffs)
        self.eigenvectors = Vt[:npc]
        self.n_pcs = npc

        #print(('fraction of variance kept: ', self.pcaobj.sumvariance[npc]))
        self.lowVs = np.transpose(np.dot(self.eigenvectors, np.transpose(self.functional_coeffs)))

    def plot_original_data(self):
        fig = plt.figure()
        x = list(range(self.n_frames))
        for i in x:
            plt.plot(x, self.data[i])
        plt.show()

    def plot_functional_data(self):
        fig = plt.figure()
        x = list(range(self.n_frames))
        for i in x:
            plt.plot(x, self.evalutation_datamat[i])
        plt.show()

    def get_backprojected_functional_data(self):
        backprojection = np.dot(self.lowVs, self.eigenvectors)
        backprojection += self.mean
        return backprojection

    def evaluate_coeffs_error(self):
        backprojection = self.get_backprojected_functional_data()
        coeff_error = 0
        for i in range(self.n_samples):
            coeff_error += np.linalg.norm(self.functional_coeffs[i] + self.mean - backprojection[i])
        print(("coeffs error is: ", coeff_error/self.n_samples))

    def evaluate_back_projection(self):
        backprojection = self.get_backprojected_functional_data()
        evaluation_backproject_mat = np.zeros((self.n_samples, self.n_frames))
        for i in range(self.n_samples):
            evaluation_backproject_mat[i] = si.splev(list(range(self.n_frames)),
                                                     (self.knots, backprojection[i], 3))
        error = 0
        for i in range(self.n_samples):
            error += np.linalg.norm(self.data[i] - evaluation_backproject_mat[i])
        print(('fpca reconstruction error is: ', error/self.n_samples))

    def evaluate_reconstruction_error_per_frame(self):
        backprojection = self.get_backprojected_functional_data()
        evaluation_backproject_mat = np.zeros((self.n_samples, self.n_frames))
        for i in range(self.n_samples):
            evaluation_backproject_mat[i] = si.splev(list(range(self.n_frames)),
                                                     (self.knots, backprojection[i], 3))
        errors = np.zeros(self.n_frames)
        for i in range(self.n_frames):
            errors[i] = np.linalg.norm(self.evalutation_datamat[:, i] - evaluation_backproject_mat[:, i])/self.n_samples
        return errors

    def evalute_functional_backprojection(self):
        backprojection = self.get_backprojected_functional_data()
        evaluation_backproject_mat = np.zeros((self.n_samples, self.n_frames))
        for i in range(self.n_samples):
            evaluation_backproject_mat[i] = si.splev(list(range(self.n_frames)),
                                                     (self.knots, backprojection[i], 3))
        error = 0
        for i in range(self.n_samples):
            error += np.linalg.norm(self.evalutation_datamat[i] - evaluation_backproject_mat[i])
        print(('functional reconstruction error is: ', error/self.n_samples))