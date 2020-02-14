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
import matplotlib.pyplot as plt
from sklearn import mixture
from anim_utils.utilities.io_helper_functions import write_to_json_file


class GMMTrainer(object):
    def __init__(self):
        self.averageScore = 0

    def fit(self, data, score='AIC'):
        assert len(data.shape) == 2, ('the data should be a 2d matrix')
        self._train_gmm(data, score=score)
        self._create_gmm(data)

    def _train_gmm(self, data, n_K=40, score='BIC', DEBUG=0):
        obs = np.random.permutation(data)
        n_samples = len(data)
        if n_samples < n_K:
            n_K = n_samples -1
        model_scores = []
        K = list(range(1, n_K + 1))
        for i in K:
            gmm = mixture.GaussianMixture(n_components=i, covariance_type='full')
            gmm.fit(obs)
            if score == 'BIC':
                model_scores.append(gmm.bic(obs))
            elif score == 'AIC':
                model_scores.append(gmm.aic(obs))
            else:
                raise NotImplementedError
        min_idx = min(range(n_K), key=model_scores.__getitem__)
        print(('number of Gaussian: ' + str(min_idx + 1)))
        self.numberOfGaussian = min_idx + 1
        if DEBUG:
            fig = plt.figure()
            plt.plot(model_scores)
            plt.show()

    def _create_gmm(self, data):
        '''
        Using GMM to model the data with optimized number of Gaussian
        '''
        self.gmm = mixture.GaussianMixture(n_components=self.numberOfGaussian,
                               covariance_type='full')
        self.gmm.fit(data)
        scores = self.gmm.score(data)
        self.averageScore = np.mean(scores)
        print('average score is:' + str(self.averageScore))

    def convert_model_to_json(self):
        model_data = {'gmm_weights': self.gmm.weights_.tolist(),
                      'gmm_means': self.gmm.means_.tolist(),
                      'gmm_covars': self.gmm.covariances_.tolist()}
        return model_data

    def save_model(self, filename):
        write_to_json_file(filename, self.convert_model_to_json())
