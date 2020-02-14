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
Created on Mon Jun 15 14:58:08 2015

@author: Erik Herrmann
"""
import numpy as np
from sklearn.decomposition import PCA

END_EFFECTORS = ["Hips", "RightHand","LeftHand", "RightFoot","LeftFoot"]#4 5
END_EFFECTORS2 = ["LeftUpLeg", "RightUpLeg", "RightHand","Bip01_R_Finger2","LeftHand","Bip01_L_Finger2", "RightFoot", "Bip01_R_Toe0","LeftFoot","Bip01_L_Toe0"]
LEN_END_EFFECTOR = 3


def _back_project_data_flat(data, model):
    motions = []
    for s in data:
        m = model.back_project_example(s)
        shape = m.shape#motions
        motions.append(m.reshape(shape[0]*shape[1]))
    motions = np.array(motions)
    return motions


def _back_project_data(data, model):
    motions = []
    for s in data:
        m = model.back_project_example(s)
        motions.append(m)
    motions = np.array(motions)
    return motions


def map_to_3d_positions(data, model, joint_name):
    motions = _back_project_data_flat(data, model)
    features = []
    for m in motions:
        p = model.sample_generator.skeleton.nodes[joint_name].get_global_position(m[-1])
        features.append(p)
    return np.array(features), None


def map_to_pca_on_3d_spline(data, model, joint_name):
    motions = _back_project_data(data, model)
    features = []
    for m in motions:
        sample = np.zeros(len(m) * LEN_END_EFFECTOR)
        offset = 0
        for idx, f in enumerate(m):
            # print offset, idx, sample.shape
            p = model.sample_generator.skeleton.nodes[joint_name].get_global_position(f)
            sample[offset:offset + LEN_END_EFFECTOR] = p
            offset += LEN_END_EFFECTOR
        features.append(sample)
    mean = np.average(features)
    features = features - mean
    np.array(features)
    pca = PCA(n_components=model.n_components)
    features = pca.fit_transform(features)
    return features, (pca, mean)


def map_to_pca_on_multi_joint_3d_spline(data, model, joint_names=END_EFFECTORS):
    motions = _back_project_data(data, model)
    print("motions shape",motions.shape)
    n_frames = len(motions[0])#16
    n_end_effectors = len(joint_names)#
    features = []
    for m in motions:
        sample = np.zeros(n_frames * LEN_END_EFFECTOR * n_end_effectors)
        offset = 0
        #print "motion", m.shape

        for idx, f in enumerate(m):
            for joint_name in joint_names:#for for
                #print joint_name, f
                p = model.sample_generator.skeleton.nodes[joint_name].get_global_position(f)
                sample[offset:offset + LEN_END_EFFECTOR] = p
                offset += LEN_END_EFFECTOR
        features.append(sample)
    mean = np.average(features)
    pca = PCA(n_components=model.n_components)
    features = pca.fit_transform(features)
    return features, (pca, mean)


def map_to_pca_on_multi_joint_3d_spline2(data, model, joint_names=END_EFFECTORS,n_components=16, step=4):
    print("motions shape",data.shape,n_components)
    motions = _back_project_data(data, model)
    return map_motions_to_pca_on_multi_joint_3d_spline2(motions, model.sample_generator.skeleton, joint_names, n_components, step)


def map_motions_to_pca_on_multi_joint_3d_spline2(motions, skeleton, joint_names=END_EFFECTORS,n_components=16, step=4):
    point_clouds = map_motions_to_euclidean_space(motions, skeleton, joint_names, step)
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(point_clouds)
    return features, pca

def map_to_euclidean_pca(data, model, joint_names=END_EFFECTORS, step=4):
    motions = _back_project_data(data, model)
    return map_motions_to_euclidean_pca(motions, model.sample_generator.skeleton, joint_names,  step)


def map_motions_to_euclidean_pca(motions, skeleton, joint_names=END_EFFECTORS,step=4):
    point_clouds = map_motions_to_euclidean_space(motions, skeleton, joint_names, step)
    pca = PCA(n_components=0.95)#svd_solver="full"
    features = pca.fit_transform(point_clouds)
    print("projected features with shape",features.shape)
    return features, pca


def map_motions_to_euclidean_space(motions, skeleton, joint_names, step=4):
    print("project frames to euclidean space")
    n_end_effectors = len(joint_names)
    point_clouds = []
    count = 0
    n_samples = len(motions)
    for m_idx, m in enumerate(motions):
        sample = []#np.zeros(n_frames * LEN_END_EFFECTOR * n_end_effectors)
        for idx, f in enumerate(m):
            if idx % step == 0:
                offset = 0
                frame = np.zeros(LEN_END_EFFECTOR * n_end_effectors)
                for joint_name in joint_names:
                    p = skeleton.nodes[joint_name].get_global_position(f)
                    frame[offset:offset + LEN_END_EFFECTOR] = p
                    offset += LEN_END_EFFECTOR
            sample.append(frame)
        point_clouds.append(np.array(sample).flatten())
        count += 1
        print("back projected", count, "of", n_samples)
    return point_clouds


def map_to_pca_on_multi_joint_3d_spline_relative(data, model, joint_names=END_EFFECTORS2, step=4):
    motions = _back_project_data(data, model)
    return map_motions_to_pca_on_multi_joint_3d_spline_relative(motions,model.sample_generator.skeleton, joint_names, model.n_components, step)


def map_motions_to_pca_on_multi_joint_3d_spline_relative(motions,skeleton, joint_names, n_components, step=4):
    print("motions shape",motions.shape)
    n_frames = len(motions[0])
    n_end_effectors = len(joint_names)
    features = []
    for m_idx, m in enumerate(motions):
        sample = []#np.zeros(n_frames * LEN_END_EFFECTOR * n_end_effectors)
        initial_pose = np.zeros(LEN_END_EFFECTOR * n_end_effectors)
        offset = 0
        for joint_name in joint_names:
            p = skeleton.nodes[joint_name].get_global_position(m[0])
            initial_pose[offset:offset + LEN_END_EFFECTOR] = p
            offset += LEN_END_EFFECTOR
        for idx in range(1, n_frames):
            if idx % step == 0:
                offset = 0
                frame = np.zeros(LEN_END_EFFECTOR * n_end_effectors)
                for joint_name in joint_names:
                    p = skeleton.nodes[joint_name].get_global_position(m[idx])
                    frame[offset:offset + LEN_END_EFFECTOR] = p
                    offset += LEN_END_EFFECTOR
                if idx == 1:
                    prev_frame = initial_pose
                else:
                    prev_frame = sample[-1]
                sample.append(frame-prev_frame)
        features.append(np.array(sample).flatten())
    mean = np.average(features)
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(features)
    return features, (pca, mean)
