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
import json
import os
import scipy.interpolate as si
import numpy as np
from .utils import center_data, run_pca

class FPCASemanticData():
    def __init__(self):
        pass

    def load_data_from_file(self, annotation_file):
        with open(annotation_file, 'r') as infile:
            self.annotation_data = json.load(infile)

    def set_knots(self, knots):
        self.knots = knots

    def spline_representation(self):
        self.semantic_spline = {}
        for item in list(self.annotation_data.keys()):
            filename = os.path.split(item)[-1]
            self.semantic_spline[filename] = []
            for feature, anno_vec in self.annotation_data[item].items():
                # print(range(len(anno_vec)))
                # print(anno_vec)
                # print(self.knots)
                knots, coeffs, degree = si.splrep(list(range(len(anno_vec))), anno_vec, t=self.knots[4: -4])
                self.semantic_spline[filename] += coeffs[:-4].tolist()

    def dimension_reduction(self):
        self.filename_order = list(self.semantic_spline.keys())
        self.semantic_vecs = np.asarray(list(self.semantic_spline.values()))
        self.semantic_vecs, self.mean_vec, std = center_data(self.semantic_vecs)
        Vt, npc = run_pca(self.semantic_vecs, fraction=0.95)
        self.eigenvectors = Vt[:npc]
        print('number of eigenvectors: ' + str(npc))
        self.lowVs = self.project_data(self.semantic_vecs)
        self.npc = npc

    def project_data(self, data):
        low_vecs = []
        for i in range(len(data)):
            low_vec = np.dot(self.eigenvectors, data[i])
            low_vecs.append(low_vec)
        low_vecs = np.asarray(low_vecs)
        return low_vecs

    def save_data(self, save_file):
        semantic_data = {'fileorder': self.filename_order,
                         'semantic_low_dimensional_data': self.lowVs.tolist(),
                         'eigen_vector_semantic': self.eigenvectors.tolist(),
                         'semantic_mean_vec': self.mean_vec.tolist(),
                         'n_basis_semantic': 8,
                         'n_pc': int(self.npc)}
        with open(save_file, 'w') as outfile:
            json.dump(semantic_data, outfile)