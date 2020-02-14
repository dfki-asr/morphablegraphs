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
@author: Han Du
"""

import numpy as np
from ..motion_model.motion_primitive import MotionPrimitive
import json
from anim_utils.animation_data.utils import pose_orientation_quat, \
                                            get_cartesian_coordinates_from_quaternion, \
                                            get_rotation_angle
import matplotlib.pyplot as plt
from .motion_primitive.gmm_trainer import GMMTrainer
from sklearn import mixture
from anim_utils.animation_data.bvh import BVHReader
from anim_utils.animation_data.skeleton import Skeleton


class FeaturePointModel(object):
    def __init__(self, motion_primitive_file, skeleton_file):
        self.motion_primitive_model = MotionPrimitive(motion_primitive_file)
        bvhreader = BVHReader(skeleton_file)
        self.skeleton = Skeleton(bvhreader)
        self.low_dimension_vectors = []
        self.feature_points = []
        self.orientations = []
        self.feature_point = None
        self.threshold = None
        self.root_pos = []
        self.root_orientation = []
        self.root_rot_angles = []
        self.feature_point_dist = None
        self.root_feature_dist = None
        self.root_feature_dist_type = None

    def create_feature_points(self, joint_name_list, n, frame_idx):
        """
        Create a set of samples, calculate target point position and orientation
        The motion sample is assumed already to be well aligned.
        They all start from [0,0] and face [0, -1] direction in 2D ground
        :param joint_name:
        :param n:
        :return:
        """
        assert type(joint_name_list) == list, 'joint names should be a list'
        self.feature_point_list = joint_name_list
        for i in range(n):
            low_dimension_vector = self.motion_primitive_model.sample_low_dimensional_vector()
            motion_spline = self.motion_primitive_model.back_project(low_dimension_vector)
            self.low_dimension_vectors.append(low_dimension_vector.tolist())
            quat_frames = motion_spline.get_motion_vector()
            start_root_point = np.array(quat_frames[0][:3])
            tmp = []
            for joint in self.feature_point_list:
                target_point = get_cartesian_coordinates_from_quaternion(self.skeleton,
                                                                         joint,
                                                                         quat_frames[frame_idx])
                relative_target_point = target_point - start_root_point
                tmp.append(relative_target_point)
            self.feature_points.append(np.ravel(tmp))
            ori_vector = pose_orientation_quat(quat_frames[-1]).tolist()
            self.orientations.append(ori_vector)

    def create_root_pos_ori(self, n):
        assert 'walk' or 'carry' in self.motion_primitive_model.name, 'root distribution only works for trajectory motion.'
        for i in range(n):
            motion_spline = self.motion_primitive_model.sample()
            quat_frames = motion_spline.get_motion_vector()
            rot_pos = np.array([quat_frames[-1][0], quat_frames[-1][2]])
            rot_ori = pose_orientation_quat(quat_frames[-1])
            self.root_pos.append(rot_pos)
            self.root_orientation.append(rot_ori)
        self.root_feature_dist_type = 'vector'

    def convert_ori_to_angle_deg(self, ref_ori=[0, -1]):
        assert self.root_orientation != [], ("please create root orientation list first!")
        for ori_vec in self.root_orientation:
            angle = -get_rotation_angle(ori_vec, ref_ori)
            angle_rad = np.deg2rad(angle)
            self.root_rot_angles.append(angle_rad)
        self.root_feature_dist_type = 'angle'

    def model_root_dist(self):
        if self.root_feature_dist_type == 'vector':
            training_samples = np.concatenate((np.asarray(self.root_pos),
                                               np.asarray(self.root_orientation)),
                                              axis=1)
        elif self.root_feature_dist_type == 'angle':
            training_samples = np.concatenate((np.asarray(self.root_pos),
                                               np.reshape(self.root_rot_angles, (len(self.root_rot_angles), 1))),
                                              axis=1)
        else:
            raise ValueError('The type of root feature points is not specified!')
        gmm_trainer = GMMTrainer(training_samples)
        self.root_feature_dist = gmm_trainer.gmm
        self.root_threshold = gmm_trainer.averageScore

    def sample_new_root_feature(self):
        new_sample = np.ravel(self.root_feature_dist.sample())
        # normalize orientation
        if self.root_feature_dist_type == 'vector':
            new_sample[2:] = new_sample[2:]/np.linalg.norm(new_sample[2:])
        return new_sample

    def save_root_feature_dist(self, save_filename):
        data = {'name': self.motion_primitive_model.name,
                'feature_point': 'Hips',
                'gmm_weights': self.root_feature_dist.weights_.tolist(),
                'gmm_means': self.root_feature_dist.means_.tolist(),
                'gmm_covars': self.root_feature_dist.covars_.tolist(),
                'threshold': self.root_threshold,
                'feature_type': self.root_feature_dist_type}
        with open(save_filename, 'wb') as outfile:
            json.dump(data, outfile)

    def load_root_feature_dist(self, model_file):
        with open(model_file, 'rb') as infile:
            data = json.load(infile)
        n_components = len(data['gmm_weights'])
        self.root_feature_dist = mixture.GMM(n_components, covariance_type='full')
        self.root_feature_dist.weights_ = np.array(data['gmm_weights'])
        self.root_feature_dist.means_ = np.array(data['gmm_means'])
        self.root_feature_dist.converged_ = True
        self.root_feature_dist.covars_ = np.array(data['gmm_covars'])
        self.root_feature_dist_type = data['feature_type']
        self.root_feature_threshold = data['threshold']

    def score_trajectory_target(self, target):
        assert self.root_feature_dist is not None, 'Please model or load root feature distribution.'
        if self.root_feature_dist_type == 'vector':
            assert len(target) == 4, 'The trajectory target should be a vector of length 4.'
        else:
            assert len(target) == 3, 'The trajectory target should be a vector of length 3.'
        return self.root_feature_dist.score([target,])[0]

    def save_data(self, save_filename):
        output_data = {}
        output_data['motion_vectors'] = self.low_dimension_vectors
        output_data['feature_points'] = self.feature_points
        output_data['orientations'] = self.orientations
        with open(save_filename, 'wb') as outfile:
            json.dump(output_data, outfile)

    def plot_orientation(self):
        fig = plt.figure()
        for i in range(len(self.orientations)):
            plt.plot([0, self.orientations[i][0]], [0, self.orientations[i][1]], 'r')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.show()

    def load_data_from_json(self, data_file):
        with open(data_file, 'rb') as infile:
            training_data = json.load(infile)
        self.low_dimension_vectors = training_data['motion_vectors']
        self.feature_points = training_data["feature_points"]
        self.orientations = training_data["orientations"]

    def model_feature_points(self):
        # training_samples = np.concatenate((self.feature_points, self.orientations), axis=1)
        training_samples = np.asarray(self.feature_points)
        gmm_trainer = GMMTrainer(training_samples)
        self.feature_point_dist = gmm_trainer.gmm
        self.threshold = gmm_trainer.averageScore - 5

    def save_feature_distribution(self, save_filename):
        data = {'name': self.motion_primitive_model.name,
                'feature_point': self.feature_point,
                'gmm_weights': self.feature_point_dist.weights_.tolist(),
                'gmm_means': self.feature_point_dist.means_.tolist(),
                'gmm_covars': self.feature_point_dist.covars_.tolist(),
                'threshold': self.threshold}
        with open(save_filename, 'wb') as outfile:
            json.dump(data, outfile)

    def sample_new_feature(self):
        return np.ravel(self.feature_point_dist.sample())

    def load_feature_dist(self, dist_file):
        """
        Load feature point distribution from json file
        :param dist_file:
        :return:
        """
        with open(dist_file, 'rb') as infile:
            data = json.load(infile)
        n_components = len(data['gmm_weights'])
        self.feature_point_dist = mixture.GMM(n_components, covariance_type='full')
        self.feature_point_dist.weights_ = np.array(data['gmm_weights'])
        self.feature_point_dist.means_ = np.array(data['gmm_means'])
        self.feature_point_dist.converged_ = True
        self.feature_point_dist.covars_ = np.array(data['gmm_covars'])
        self.threshold = data['threshold']

    def evaluate_target_point(self, target_point):
        assert len(target_point) == len(self.feature_points[0]), 'the length of feature is not correct'
        return self.feature_point_dist.score([target_point,])[0]

    def check_reachability(self, target_point):
        score = self.evaluate_target_point(target_point)
        if score < self.threshold:
            return False
        else:
            return True