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
from .utils import center_data, run_pca


class FPCATimeSemantic(object):
    def __init__(self, n_basis=8, n_components_temporal=None, precision_temporal=0.99):
        self.temporal_semantic_data = None
        self.temporal_data = None
        self.semantic_data = None
        self.semantic_annotation_list = None
        self.n_basis = n_basis
        self.n_components_temporal = n_components_temporal
        self.precision_temporal = precision_temporal

    def load_time_warping_data(self, time_warping_file):
        with open(time_warping_file, 'r') as infile:
            self.temporal_data = json.load(infile)

    def load_semantic_annotation(self, semantic_annotation_file):
        with open(semantic_annotation_file, 'r') as infile:
            semantic_annotation = json.load(infile)
        self.semantic_data = semantic_annotation['data']
        self.semantic_annotation_list = semantic_annotation['annotation_list']

    def load_semantic_annotation_from_dic(self, semantic_annotation_dic):
        self.semantic_data = semantic_annotation_dic['data']
        self.semantic_annotation_list = semantic_annotation_dic['annotation_list']

    def load_time_warping_data_from_dic(self, timewarping_dic):
        self.temporal_data = timewarping_dic

    def merge_temporal_semantic_data(self):
        if self.temporal_data is None or self.semantic_data is None:
            raise ValueError('Load semantic annotation or time warping data first!')
        temporal_semantic_data_dic = {}
        for key, value in list(self.semantic_data.items()):
            if key in list(self.temporal_data.keys()):
                temporal_semantic_data_dic[key] = [self.temporal_data[key]]
                for feature in self.semantic_annotation_list:
                    temporal_semantic_data_dic[key].append(value[feature])
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

    def z_t_transform_vector(self, vec):
        # shift control points to start from 0
        w_tmp = np.array(vec)
        w_tmp -= w_tmp[0]
        w_tmp = self._get_monotonic_indices(w_tmp)
        assert FPCATimeSemantic._is_strict_increasing(w_tmp)

        # add one to each entry, because we start with 0
        w_tmp = w_tmp + 1
        w_tmp = np.insert(w_tmp, 0, 0)  # set w(0) to zero

        w_diff = np.diff(w_tmp)
        z_transform = np.log(w_diff)
        return z_transform

    def functional_data_representation(self):
        # convert temporal_semantic_data to functional data
        self.fpca_data = []
        n_canonical_frame = len(self.temporal_semantic_data[0])
        n_basis_funcs = self.n_basis
        knots = self.get_b_spline_knots(n_basis_funcs, n_canonical_frame)

        time_vec = list(range(n_canonical_frame))
        coeff_vec = []
        warping_function_list = self.temporal_semantic_data
        for warping_function in warping_function_list:
            tck = si.splrep(time_vec, warping_function, t=knots[4:-4])
            control_points = tck[1][:-4]
            control_points[0] = warping_function[0]
            control_points[-1] = warping_function[-1]
            #print "control points", control_points, warping_function
            control_points = self.z_t_transform_vector(control_points)
            coeff_vec.append(control_points)
        self.fpca_data = np.asarray(coeff_vec)

    def functional_pca(self):
        # self.z_t_transform()
        self.functional_data_representation()
        self.fpca_data, self.mean_vec, std = center_data(self.fpca_data)
        Vt, npc = run_pca(self.fpca_data, fraction=self.precision_temporal)
        if self.n_components_temporal is not None:
            self.eigenvectors = Vt[:self.n_components_temporal]
            print('number of eigenvectors for temporal semantic data: ' + str(self.n_components_temporal))
        else:
            self.eigenvectors = Vt[:npc]
            print('number of eigenvectors for temporal semantic data: ' + str(npc))
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
                         'n_basis': self.n_basis,
                         'n_pc': int(self.npc),
                         'n_dim': len(self.semantic_annotation_list)+1,
                         'semantic_annotation': self.semantic_annotation_list}
        with open(save_file, 'w') as outfile:
            json.dump(semantic_data, outfile)

    def get_b_spline_knots_rpy2(self, n_basis, n_canonical_frames):
        import rpy2.robjects as robjects
        rcode = """
            library(fda)
            n_basis = %d
            n_frames = %d
            basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
        """% ( n_basis, n_canonical_frames)
        robjects.r(rcode)
        basis_function = robjects.globalenv['basisobj']
        return np.asarray(robjects.r['knots'](basis_function, False))

    def get_b_spline_knots(self, n_basis, n_canonical_frames):
        """ create cubic bspline knot list, the order of the spline is 4
        :param n_basis: number of knots
        :param n_canonical_frames: length of discrete samples
        :return:
        """
        n_orders = 4
        knots = np.zeros(n_orders + n_basis)
        # there are two padding at the beginning and at the end
        knots[3: -3] = np.linspace(0, n_canonical_frames-1, n_basis-2)
        knots[-3:] = n_canonical_frames - 1
        return knots