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
Created on Mon Aug 03 10:48:53 2015

@author: Han Du
"""
import numpy as np
from .pca_functional_data import PCAFunctionalData
from .functional_data import FunctionalData


class FPCASpatialData(object):

    def __init__(self, n_basis, n_components=None, fraction=0.95):
        """
        * motion_data: dictionary
        \tContains quaternion frames of each file
        """
        self.n_basis = n_basis
        self.fraction = fraction
        self.reshaped_data = None
        self.fpcaobj = None
        self.fileorder = None
        self.n_components = n_components

    def fit_motion_dictionary(self, motion_dic):
        self.fileorder = list(motion_dic.keys())
        self.fit(np.asarray(list(motion_dic.values())))

    def fit(self, motion_data):
        """
        Reduce the dimension of motion data using Functional Principal Component Analysis
        :param motion_data (numpy.array<3d>): can be either spatial data or temporal data, n_samples * n_frames * n_dims
        :return lowVs (numpy.array<2d>): low dimensional representation of motion data
        """
        assert len(motion_data.shape) == 3
        self.fpcaobj = PCAFunctionalData(motion_data,
                                         n_basis=self.n_basis,
                                         fraction=self.fraction,
                                         n_pc=self.n_components)

