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
from .motion_primitive import MotionPrimitive as MGMotionPrimitive
from .static_motion_primitive import StaticMotionPrimitive
try:
    from .extended_mgrd_mixture_model import ExtendedMGRDMixtureModel
    from mgrd import MotionPrimitiveModel as MGRDMotionPrimitiveModel
    from mgrd import QuaternionSplineModel as MGRDQuaternionSplineModel
    from mgrd import TemporalSplineModel as MGRDTemporalSplineModel
    from .legacy_temporal_spline_model import LegacyTemporalSplineModel
    has_mgrd = True
except ImportError:
    pass
    has_mgrd = False

from anim_utils.utilities.io_helper_functions import load_json_file
from anim_utils.utilities.log import write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR
from sklearn.mixture.gaussian_mixture import GaussianMixture, _compute_precision_cholesky


class MotionPrimitiveModelWrapper(object):
    """ Class that wraps the MGRD MotionPrimitiveModel

    """
    SPLINE_DEGREE = 3

    def __init__(self):
        self.motion_primitive = None
        self.use_mgrd_mixture_model = False
        self.keyframes = dict()
        self.mgrd = True

    def _load_from_file(self, mgrd_skeleton, file_name, animated_joints=None, use_mgrd_mixture_model=False, scale=None):
        self.use_mgrd_mixture_model = use_mgrd_mixture_model
        data = load_json_file(file_name)
        if data is not None:
            self._initialize_from_json(mgrd_skeleton, data, animated_joints, use_mgrd_mixture_model, scale)

    def _initialize_from_json(self, mgrd_skeleton, data, animated_joints=None, use_mgrd_mixture_model=False, scale=None):
        self.mgrd = False
        if "keyframes" in data:
            self.keyframes = data["keyframes"]

        if "spatial_coeffs" in data:
            self.motion_primitive = StaticMotionPrimitive()
            self.motion_primitive._initialize_from_json(data)
            return

        if not has_mgrd:
            if "tspm" in list(data.keys()):
                self.motion_primitive = self._load_legacy_model_from_mgrd_json(data)
            else:
                self.motion_primitive = self._load_legacy_model_from_legacy_json(data)
        else:
            use_mgrd_mixture_model = True
            if "tspm" in list(data.keys()):
                self.motion_primitive = self._load_mgrd_model_from_mgrd_json(mgrd_skeleton, data, use_mgrd_mixture_model, scale)
                self.mgrd = True
            elif animated_joints is not None:
                self.motion_primitive = self._load_mgrd_model_from_legacy_json(mgrd_skeleton, data, use_mgrd_mixture_model, animated_joints)
                self.mgrd = True
            else:
                raise Exception("Motion Primitive format is not supported")

    def _load_legacy_model_from_mgrd_json(self, data):
        motion_primitive = MGMotionPrimitive(None)
        sspm = data["sspm"]
        tspm = data["tspm"]
        gmm = data["gmm"]
        n_canonical_frames = int(max(tspm["knots"])+1)

        legacy_data = {"eigen_vectors_spatial": sspm["eigen"],
                       "mean_spatial_vector": sspm["mean"],
                       "n_basis_spatial": sspm["n_coeffs"],
                       "n_dim_spatial": sspm["n_dims"],
                       "b_spline_knots_spatial": sspm["knots"],
                       "animated_joints": sspm["animated_joints"],

                       "gmm_covars": gmm["covars"],
                       "gmm_means": gmm["means"],
                       "gmm_weights": gmm["weights"],
                       # time fpca is done differently in mgrd and the legacy motion primitive
                       #"eigen_vectors_time":time_eigen_vectors,
                       #"mean_time_vector":time_mean,
                       #"n_basis_time":tspm["n_coeffs"],
                       #"b_spline_knots_time":tspm["knots"],
                       #"n_dim_time": 1,

                       "n_canonical_frames": n_canonical_frames,
                       "translation_maxima": np.array([1,1,1])
                       }
        motion_primitive._initialize_from_json(legacy_data)
        return motion_primitive

    def _load_mgrd_model_from_legacy_json(self, mgrd_skeleton, data, use_mgrd_mixture_model, animated_joints):
        write_message_to_log("Init motion primitive model without semantic annotation", LOG_MODE_DEBUG)
        mm = MotionPrimitiveModelWrapper.load_mixture_model(data, use_mgrd_mixture_model)
        tspm = LegacyTemporalSplineModel(data)
        sspm = MGRDQuaternionSplineModel.load_from_json(mgrd_skeleton, {
            'eigen': np.asarray(data['eigen_vectors_spatial']),
            'mean': np.asarray(data['mean_spatial_vector']),
            'n_coeffs': data['n_basis_spatial'],
            'n_dims': data['n_dim_spatial'],
            'knots': np.asarray(data['b_spline_knots_spatial']),
            'degree': self.SPLINE_DEGREE,
            'translation_maxima': np.asarray(data['translation_maxima']),
            'animated_joints': animated_joints
        })
        self._pre_scale_root_translation(sspm, data['translation_maxima'])
        return MGRDMotionPrimitiveModel(mgrd_skeleton, sspm, tspm, mm)

    def _load_mgrd_model_from_mgrd_json(self, skeleton, mm_data, use_mgrd_mixture_model=True, scale=None):
        write_message_to_log("Init motion primitive model with semantic annotation", LOG_MODE_INFO)
        sspm = MGRDQuaternionSplineModel.load_from_json(skeleton, mm_data['sspm'])
        tspm = MGRDTemporalSplineModel.load_from_json(mm_data['tspm'])
        mixture_model = MotionPrimitiveModelWrapper.load_mixture_model({
            'gmm_covars': mm_data['gmm']['covars'],
            'gmm_means': mm_data['gmm']['means'],
            'gmm_weights': mm_data['gmm']['weights']
        }, use_mgrd_mixture_model)
        if scale is not None:
            self._pre_scale_root_translation(sspm, scale)
        return MGRDMotionPrimitiveModel(skeleton, sspm, tspm, mixture_model)

    def _load_legacy_model_from_legacy_json(self, data):
        write_message_to_log("Init legacy motion primitive model", LOG_MODE_DEBUG)
        motion_primitive = MGMotionPrimitive(None)
        motion_primitive._initialize_from_json(data)
        return motion_primitive

    def _pre_scale_root_translation(self, sspm, scale):
        """ undo the scaling of the root translation parameters of the principal
        components that was done during offline training

        """
        root_columns = []
        for coeff_idx in range(sspm.n_coeffs):
            coeff_start = coeff_idx * sspm.n_dims
            root_columns += np.arange(coeff_start,coeff_start+3).tolist()

        indices_range = list(range(len(root_columns)))
        x_indices = [root_columns[i] for i in indices_range if i % 3 == 0]
        y_indices = [root_columns[i] for i in indices_range if i % 3 == 1]
        z_indices = [root_columns[i] for i in indices_range if i % 3 == 2]
        sspm.fpca.eigen[:, x_indices] *= scale[0]
        sspm.fpca.eigen[:, y_indices] *= scale[1]
        sspm.fpca.eigen[:, z_indices] *= scale[2]
        sspm.fpca.mean[x_indices] *= scale[0]
        sspm.fpca.mean[y_indices] *= scale[1]
        sspm.fpca.mean[z_indices] *= scale[2]
        print("Prescale root translation")

    def sample_legacy(self, use_time=True):
        return self.motion_primitive.sample(use_time)

    def sample_mgrd(self, use_time=True):
        s_vec = self.sample_low_dimensional_vector()
        quat_spline = self.back_project(s_vec, use_time)
        return quat_spline

    def sample(self, use_time=True):
        if self.mgrd:
            return self.sample_mgrd(use_time)
        else:
            return self.sample_legacy(use_time)

    def sample_vector_legacy(self):
        return self.motion_primitive.sample_low_dimensional_vector(1)

    def sample_vector_mgrd(self):
        return self.motion_primitive.mixture.sample(1, False)[0]# we assume the sklearn Gaussian Mixture model is used
        #return self.motion_primitive.get_random_samples(1)[0]

    def sample_low_dimensional_vector(self):
        if self.mgrd:
            return self.sample_vector_mgrd()
        else:
            return self.sample_vector_legacy()

    def sample_vectors_legacy(self, n_samples=1):
        return self.motion_primitive.sample_low_dimensional_vector(n_samples)

    def sample_vectors_mgrd(self, n_samples=1):
        return self.motion_primitive.mixture.sample(n_samples, True)  # we assume the sklearn Gaussian Mixture model is used
        #return self.motion_primitive.get_random_samples(n_samples)[0]

    def sample_low_dimensional_vectors(self, n_samples=1):
        if self.mgrd:
            return self.sample_vectors_mgrd(n_samples)
        else:
            return self.sample_vectors_legacy(n_samples)

    def back_project_legacy(self, s_vec, use_time_parameters=True, speed=1.0):
        return self.motion_primitive.back_project(s_vec, use_time_parameters, speed)

    def back_project_mgrd(self, s_vec, use_time_parameters=True, speed=1.0):
        if len(np.asarray(s_vec.shape)) == 2:
            s_vec = np.ravel(s_vec)
        quat_spline = self.motion_primitive.create_spatial_spline(s_vec)
        if use_time_parameters:
            time_spline = self.motion_primitive.create_time_spline(s_vec)
            quat_spline = time_spline.warp(quat_spline)
        return quat_spline

    def back_project(self, s_vec, use_time_parameters=True, speed=1.0):
        if self.mgrd:
            return self.back_project_mgrd(s_vec, use_time_parameters, speed)
        else:
            return self.back_project_legacy(s_vec, use_time_parameters, speed)

    def back_project_time_function_legacy(self, s_vec):
        if self.motion_primitive.has_time_parameters:
            return self.motion_primitive._back_transform_gamma_to_canonical_time_function(s_vec[self.get_n_spatial_components():])
        else:
            return list(range(0,self.motion_primitive.n_canonical_frames))

    def back_project_time_function_mgrd(self, s_vec):
        if len(np.asarray(s_vec.shape)) == 2:
            s_vec = np.ravel(s_vec)
        time_spline = self.motion_primitive.create_time_spline(s_vec, labels=[])
        return np.asarray(time_spline.evaluate_domain(step_size=1.0))[:,0]

    def back_project_time_function(self, s_vec):
        if self.mgrd:
            return self.back_project_time_function_mgrd(s_vec)
        else:
            return self.back_project_time_function_legacy(s_vec)

    def get_n_canonical_frames_legacy(self):
        return self.motion_primitive.n_canonical_frames

    def get_n_canonical_frames_mgrd(self):
        #print max(self.motion_primitive.time.knots)+1, self.motion_primitive.time.n_canonical_frames
        return max(self.motion_primitive.time.knots)+1#.n_canonical_frames

    def get_n_canonical_frames(self):
        if self.mgrd:
            return self.get_n_canonical_frames_mgrd()
        else:
            return self.get_n_canonical_frames_legacy()

    def get_n_spatial_components_legacy(self):
        return self.motion_primitive.get_n_spatial_components()

    def get_n_spatial_components_mgrd(self):
        return self.motion_primitive.spatial.get_n_components()

    def get_n_spatial_components(self):
        if self.mgrd:
            return self.get_n_spatial_components_mgrd()
        else:
            return self.get_n_spatial_components_legacy()

    def get_n_time_components_legacy(self):
        return self.motion_primitive.get_n_time_components()

    def get_n_time_components_mgrd(self):
        return self.motion_primitive.time.get_n_components()

    def get_n_time_components(self):
        if self.mgrd:
            return self.get_n_time_components_mgrd()
        else:
            return self.get_n_time_components_legacy()

    def get_gaussian_mixture_model_legacy(self):
        return self.motion_primitive.gaussian_mixture_model

    def get_gaussian_mixture_model_mgrd(self):
        return self.motion_primitive.mixture

    def get_gaussian_mixture_model(self):
        if self.mgrd:
            return self.get_gaussian_mixture_model_mgrd()
        else:
            return self.get_gaussian_mixture_model_legacy()

    def get_time_eigen_vector_matrix_mgrd(self):
        return self.motion_primitive.time.fpca.eigen

    def get_time_eigen_vector_matrix_legacy(self):
        return self.motion_primitive.t_pca["eigen_vectors"]

    def get_time_eigen_vector_matrix(self):
        if self.mgrd:
            return self.get_time_eigen_vector_matrix_mgrd()
        else:
            return self.get_time_eigen_vector_matrix_legacy()

    def get_spatial_eigen_vector_matrix_legacy(self, joints=None, frame_idx=-1):
        return self.motion_primitive.s_pca["eigen_vectors"].T

    def get_spatial_eigen_vector_matrix_mgrd(self, joints=None, frame_idx=-1):
        LEN_TRANSLATION = 3
        LEN_QUATERNION = 4
        n_frames = self.get_n_canonical_frames()
        # spatial = self.motion_primitive.spatial.clone(None)
        eigen = self.motion_primitive.spatial.fpca.eigen.T
        if joints is None:
            return eigen
        else:
            n_params = 3 + len(self.motion_primitive.spatial.animated_joints) * LEN_QUATERNION
            frame_idx = (len(eigen) / n_params) - 1

            modeled_joints = [j.name for j in self.motion_primitive.spatial.animated_joints]
            joint_indices = []
            if frame_idx > -1:
                frame_range = [frame_idx]
            else:
                frame_range = range(n_frames)
            for frame in frame_range:
                frame_offset = frame * n_params
                joint_indices += [frame_offset + i * LEN_QUATERNION + LEN_TRANSLATION for i, j in
                                  enumerate(modeled_joints) if j in joints]

            coeff_indices = []
            for i in joint_indices:
                coeff_indices += list(range(i, i + LEN_QUATERNION))
            return eigen[coeff_indices]

    def get_spatial_eigen_vectors(self, joints=None, frame_idx=-1):
        if self.mgrd:
            return self.get_spatial_eigen_vector_matrix_mgrd()
        else:
            return self.get_spatial_eigen_vector_matrix_legacy()


    @staticmethod
    def load_mixture_model(data, use_mgrd=True):
         if use_mgrd:
             mm = ExtendedMGRDMixtureModel.load_from_json({'covars': data['gmm_covars'],
                                                           'means': data['gmm_means'],
                                                           'weights': data['gmm_weights']})
         else:
             n_components =len(np.array(data['gmm_weights']))
             mm = GaussianMixture(n_components=n_components, covariance_type='full')#weights_init=np.array(data['gmm_weights']),
            #reg_covar=np.array(data['gmm_covars']),
            #means_init=np.array(data['gmm_means']), covariance_type='full')
             mm.weights_ = np.array(data['gmm_weights'])
             mm.means_ = np.array(data['gmm_means'])
             mm.converged_ = True
             mm.covariances_ = np.array(data['gmm_covars'])
             mm.precisions_cholesky_ = _compute_precision_cholesky(mm.covariances_, covariance_type='full')
             mm.n_dims =len(mm.means_[0])
             # if 'gmm_precisions_cholesky' not in data.keys():

             write_message_to_log("Initialize scipy GMM", LOG_MODE_DEBUG)
         return mm

    def get_animated_joints(self):
        if self.mgrd:
            return [j.name for j in self.motion_primitive.spatial.animated_joints]
        else:
            return self.motion_primitive.get_animated_joints()