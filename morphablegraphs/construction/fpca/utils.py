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
utility functions to perform Principal Component Analysis based on the following resource:
@src: http://stackoverflow.com/questions/1730600/principal-component-analysis-in-python

"""

import numpy as np
from scipy.sparse.linalg import svds


def run_pca(A, fraction=0.90, use_lapack=False):
    """ Returns tuple (Vt, npc) where Vt is the matrix with eigen vectors of A in its rows and npc are row indices ordered based on the corresponding eigenvalues
    """
    assert 0 <= fraction <= 1
    if use_lapack:
        # A = U . diag(d) . Vt, O( m n^2 ), lapack_lite --
        U, D, Vt = np.linalg.svd(A, full_matrices=True)
    else:
        k=max(1, min(A.shape)-1)
        U, D, Vt = svds(A, k)
        indices = sorted(range(len(D)), key = D.__getitem__, reverse=True)
        U = U[indices]
        D = D[indices]
        Vt = Vt[indices]
    assert np.all(D[:-1] >= D[1:])  # sorted
    eigen = D**2
    sumvariance = np.cumsum(eigen)
    sumvariance /= sumvariance[-1]
    orderd_indices = np.searchsorted(sumvariance, fraction) + 1
    return Vt, orderd_indices


def center_data(A, axis=0, scale=False):
    """ Returns Tuple(centered_data, mean, std)
    """
    mean = A.mean(axis=axis)
    centered_data = A-mean
    if scale:
        std = centered_data.std(axis=axis)
        std = np.where(std, std, 1.)
        centered_data /= std
    else:
        std = np.ones(A.shape[-1])
    return centered_data, mean, std
