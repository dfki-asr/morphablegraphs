#!/usr/bin/env python
#
# Copyright 2019 Daimler AG.
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
# -*- coding: utf-8 -*-
"""
Created on Wed May 07 13:44:05 2014

@author: Markus Mauer
"""
import os
from copy import deepcopy
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from anim_utils.animation_data import BVHReader, BVHWriter, SkeletonBuilder
from anim_utils.animation_data.utils import get_cartesian_coordinates_from_euler_full_skeleton as get_cartesian_coords


def dict_and(a, b):
    """ FOR INTERNAL USE ONLY!
    Compare one Dictionaries with another and returns a
    Dictionaries containing the logical piecewise AND (only for Keys in both
    Dictionaries)

    Parameters
    ----------
    a : dict
        The first dict
    b : dict
        The second dict

    Returns
    -------
    A Dictionarie, containing the logical, piecewise AND of a and b.
    True is 1 and False is 0
    """
    keys = list(set.intersection(set(a.keys()), set(b.keys())))

    return {key: (1 if a[key] and b[key] else 0) for key in keys}


def dict_or(a, b):
    """ FOR INTERNAL USE ONLY!
    Compare one Dictionaries with another and returns a
    Dictionaries containing the logical piecewise OR (only for Keys in both
    Dictionaries)

    Parameters
    ----------
    a : dict
        The first dict
    b : dict
        The second dict

    Returns
    -------
    A Dictionarie, containing the logical, piecewise OR of a and b.
    True is 1 and False is 0
    """
    keys = list(set.intersection(set(a.keys()), set(b.keys())))

    return {key: (1 if a[key] or b[key] else 0) for key in keys}


def smooth_bitvectors(bitvectors, threshold=4):
    """ Smooth the bitvector by flating out every peak which is smaller than
    the given threshold

    Parameters
    ----------
    bitvectors: list of dicts
        The bitvector
    threshod: int
        The minimum number for a peak
    """
    features = list(bitvectors.keys())
    vectors = deepcopy(bitvectors)

    counter = 0
    at_start = True

    for feature in features:
        for i in range(1, len(bitvectors[feature])):
            if vectors[feature][i] != vectors[feature][i-1]:
                if at_start:
                    at_start = False
                    counter = 0
                elif counter < threshold:
                    for j in range(1, counter+2):
                        vectors[feature][i-j] = vectors[feature][i]
                else:
                    counter = 0
            else:
                counter += 1
    return vectors


def calc_bitvector_walking(frames, features, skeleton=None, verbose=False,
                           threshold=0.2):
    """ Detect a bit vector for each frame in the motion

    Parameters
    ----------
    frames : numpy.ndarray
        The frames of the walking motion
    feature : list of str
        The Food features
    skeleton: animation_data.skeleton.Skeleton
        The skeleton to be used. If None, the default interact skeleton is
        used.
    verbose: bool
        Wether to print / plot debug information or not
    threshold: int
        T.B.D.

    Returns
    -------
    A list containing a bit vector for each frame. Each bit vector has one
    element for each feature, indicating wether this feature is on the ground
    or not.
    """
    if skeleton is None:
        reader = BVHReader('skeleton.bvh')
        skeleton = Skeleton(reader)

    if isinstance(features, str):
        features = [features]

    jointpositions = {}

    # Get cartesian position frame wise. Change this for performance?
    for feature in features:
        jointpositions[feature] = []
        for frame in frames:
            jointpositions[feature].append(get_cartesian_coords(reader,
                                                                skeleton,
                                                                feature,
                                                                frame))
        jointpositions[feature] = np.array(jointpositions[feature])

    bitvector = {}
    xz_threshold = threshold
    relativ_velo = {}

    for feature in features:
        dif = np.diff(jointpositions[feature], axis=0)
        relativ_velo[feature] = dif[:, 0]**2 + dif[:, 1]**2 + dif[:, 2]**2
        bitvector[feature] = relativ_velo[feature] < threshold
        bitvector[feature] = np.concatenate((bitvector[feature],
                                             [bitvector[feature][-1]],
                                             [bitvector[feature][-1]]))

    height_bitvectors = [{feature: 0 for feature in features}
                         for i in range(frames.shape[0])]

    height_threshold = threshold

    jointpositions_y = {}
    for feature in features:
        jointpositions_y[feature] = [pos[1] for pos in jointpositions[feature]]
        for i in range(len(jointpositions_y[feature])):
            if jointpositions_y[feature][i] < height_threshold:
                height_bitvectors[i][feature] = 1

    bitvectors_smoothed = smooth_bitvectors(bitvector, threshold=8)
    bitvectors_smoothed = smooth_bitvectors(bitvectors_smoothed, threshold=4)
    bitvectors_smoothed = smooth_bitvectors(bitvectors_smoothed, threshold=2)
    bitvectors_smoothed = smooth_bitvectors(bitvectors_smoothed, threshold=1)

    if verbose:
        # Plots:

        plt.figure()
        for feature in ['Bip01_L_Toe0', 'Bip01_R_Toe0']:
            plt.plot(bitvectors_smoothed[feature], label=feature)
        plt.legend()
        plt.ylim([0, 2])
        plt.title('walk')
        plt.xlabel('frameindex')
        plt.ylabel('bitvalue')
        plt.figure()
        for feature in ['Bip01_L_Toe0', 'Bip01_R_Toe0']:
            plt.plot(bitvector[feature], label=feature)
        plt.legend()
        plt.ylim([0, 2])
        plt.title('walk')
        plt.xlabel('frameindex')
        plt.ylabel('bitvalue')

#        plt.figure()
#        for feature in ['LeftFoot', 'RightFoot']:
#            plt.plot(bitvectors_smoothed[feature], label=feature)
#        plt.legend()
#        plt.ylim([0, 2])
#        plt.title('walk')
#        plt.xlabel('frameindex')
#        plt.ylabel('bitvalue')

        plt.figure()
        line_x = list(range(len(relativ_velo[features[0]])))
        line_y = [xz_threshold] * len(line_x)
        plt.plot(line_x, line_y)
        for feature in features:
            plt.plot(relativ_velo[feature], label=feature)
        plt.legend()
        plt.xlabel('frameindex')
        plt.ylabel('relativ velocity in xz')

#        plt.figure()
#        for feature in features:
#            tmp = [vector[feature] for vector in height_bitvectors]
#            plt.plot(tmp, label=feature)
#        plt.legend()
#        plt.ylim([0, 2])
#        plt.title('walk')
#        plt.xlabel('frameindex')
#        plt.ylabel('bitvalue (using height)')
#
#        plt.figure()
#        line_x = range(len(relativ_velo_xz[features[0]]))
#        line_y = [xz_threshold] * len(line_x)
#        plt.plot(line_x, line_y)
#        for feature in features:
#            plt.plot(jointpositions_y[feature], label=feature)
#        plt.legend()
#        plt.xlabel('frameindex')
#        plt.ylabel('relativ velocity in xz')

        plt.ioff()
        plt.show()

    return bitvectors_smoothed


def detect_walking_keyframes(frames, features, skeleton, verbose=False):
    """ FOR INTERNAL USE ONLY! Use detect_keyframes with motion_type='walking'
    Detect all Keyframes for the given Feature(s) in the given Walking Motion

    Parameters
    ----------
    frames : numpy.ndarray
        The frames of the walking motion
    feature : list of str
        The Food features
    skeleton: animation_data.skeleton.Skeleton
        The skeleton to be used.
    verbose: bool
        Wether to print / plot debug information or not

    Returns
    -------
    A dictionary containing a list for each feature.
    Each list contains all [Startframe, Endframe] Pairs for this feature.
    """
    bitvectors = calc_bitvector_walking(frames, features, skeleton, verbose)

    keyframes = {feature: [] for feature in features}

    print(features)

    def next_keyframe(bitvector):
        for i in range(1, len(bitvector)):
            if bitvector[i] == 0 and bitvector[i-1] == 1:
                yield i

    last = 0
    highest = 0
    highest_feature = None

    feature_order = [(f, next(next_keyframe(bitvectors[f]))) for f in features]
    feature_order = sorted(feature_order, key=itemgetter(1))

    gens = {feature: next_keyframe(bitvectors[feature])
            for feature in features}

    while len(list(gens.values())) > 0:
        pop = []
        for feature, _ in feature_order:
            try:
                i = next(gens[feature])
            except StopIteration:
                pop.append((feature, _))
                continue
            keyframes[feature].append([last, i])
            last = i
            if highest < i:
                highest = i
                highest_feature = feature
        for f, _ in pop:
            print("pop", f)
            gens.pop(f)
            feature_order.remove((f, _))

    f = None
    for feature in features:
        if feature != highest_feature:
            f = feature
            break
    keyframes[f].append([highest, len(bitvectors[f])-1])

    for feature in features:
        keyframes[feature].sort()

    print("Keyframes:", keyframes)
    return keyframes


def detect_keyframes(frames, features, skeleton=None,
                     motion_type='walking', verbose=False):
    """ Detect all Keyframes for the given Feature(s) in the given Motion

    Parameters
    ----------
    frames : numpy.ndarray
        The frames of the walking motion
    feature : list of str
        The features corresponding to the searched Keyframes.
        if the motion_type is 'walking', than this should be eather the left
        foot, the right foot or both.
    skeleton: animation_data.skeleton.Skeleton
        The skeleton to be used. If None, the default Interact skeleton
        will be loaded
    motion_type: string
        The motion type of the given frames. Currently, only 'walking'
        is supported
    verbose: bool
        Wether to print / plot debug information or not


    Returns
    -------
    A list containing all Keyframes.
    """
    if motion_type == 'walk' or motion_type == 'walking':
        return detect_walking_keyframes(frames, features, skeleton, verbose)

    raise ValueError('The motiontype "%s" is not supported yet' % motion_type)


def splitt_motion(frames, keyframes, mname, skeleton_file='skeleton.bvh',
                  outputpath=''):
    """ Splitt a Motion by the given Keyframes

    Parameters
    ----------
    frames : numpy.ndarray
        The frames of the walking motion
    keyframes : dict of list of int
        A dictionary containing a list for each feature.
        Each list contains all Keyframes for this feature.
    mname: string
        Subfix of the splitted motions (i.e. the original name of the
        motion)
    skeleton_file: string (optional)
        The path to the skeleton file. Default is the 'skeleton.bvh' in the
        current folder
    outputpath: string (optional)
        The path where to save the motions. Default is the current folder

    Returns
    -------
    None
    """

    # Calc number of steps for status update
    n = 0.0
    counter = 0.0
    for feature in keyframes:
        n += len(keyframes[feature])

    tmpmins = []
    tmpmax = []
    for feature in keyframes:
        tmpmins.append(min(keyframes[feature])[0])
        tmpmax.append(max(keyframes[feature])[1])
    firstframe = min(tmpmins)
    lastframe = max(tmpmax)

    reader = BVHReader(skeleton_file)
    skel = SkeletonBuilder().load_from_bvh(reader)
    for feature in keyframes:
        # save first step:
        if firstframe in keyframes[feature][0]:
            keyframe = keyframes[feature][0]
            subframes = frames[keyframe[0]:keyframe[1]]
            name = 'begin_' + str(keyframe[0]) + '_' + str(keyframe[1]) \
                + '_' + feature + '_' + mname
            BVHWriter(outputpath + os.sep + name, skel, subframes, 0.013889)
            keyframes[feature] = keyframes[feature][1:]

        # last step:
        if lastframe in keyframes[feature][-1]:
            keyframe = keyframes[feature][-1]
            subframes = frames[keyframe[0]:keyframe[1]]
            name = 'end_' + str(keyframe[0]) + '_' + str(keyframe[1]) \
                + '_' + feature + '_' + mname
            BVHWriter(outputpath + os.sep + name, skel, subframes, 0.013889)
            keyframes[feature] = keyframes[feature][:-1]

        for keyframe in keyframes[feature]:
            subframes = frames[keyframe[0]:keyframe[1]]
            name = str(keyframe[0]) + '_' + str(keyframe[1]) \
                + '_' + feature + '_' + mname
            BVHWriter(outputpath + os.sep + name, skel, subframes, 0.013889)

            counter += 1.0


def get_joint_speed(bvhreader, feature_joints):
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    left_toe_pos = []
    right_toe_pos = []
    left_toe_speed = [0]
    right_toe_speed = [0]
    for i in range(len(bvhreader.frames)):
        left_toe_pos.append(get_cartesian_coords(bvhreader,
                                                 skeleton,
                                                 feature_joints[0],
                                                 bvhreader.frames[i]))
        right_toe_pos.append(get_cartesian_coords(bvhreader,
                                                  skeleton,
                                                  feature_joints[1],
                                                  bvhreader.frames[i]))
    for i in range(len(bvhreader.frames) - 1):
        left_toe_speed.append((left_toe_pos[i+1][0] - left_toe_pos[i][0])**2 +
                              (left_toe_pos[i+1][2] - left_toe_pos[i][2])**2)
        right_toe_speed.append((right_toe_pos[i+1][0] - right_toe_pos[i][0])**2 +
                               (right_toe_pos[i+1][2] - right_toe_pos[i][2])**2)
    return left_toe_speed, right_toe_speed


def count_blocks(bit_vec):
    extended_bit_vec = np.zeros(len(bit_vec) + 1)
    extended_bit_vec[:-1] = bit_vec[:]
    extended_bit_vec[-1] = bit_vec[-2]
    blocks = []
    block_length = 1
    for i in range(len(bit_vec)):
        if extended_bit_vec[i+1] == extended_bit_vec[i]:
            block_length += 1
        else:
            blocks.append(block_length)
            block_length = 1
    blocks.append(block_length)
    return blocks


def majority_vote(index, vector, w):
    selected_elements = vector[index-w: index+w+1]
    one_counter = sum(selected_elements == 1)
    zero_counter = sum(selected_elements == 0)
    if one_counter > zero_counter:
        vector[index] = 1
    else:
        vector[index] = 0


def majority_vote_smoothing(annotation_vec, n_block):
    """
    automatically adapt window size based on the number of blocks the final result wants
    :param annotation_vec:
    :return:
    """
    block_sizes = count_blocks(annotation_vec)
    if len(block_sizes) > 3:
        sorted_block_sizes = sorted(block_sizes, reverse=True)
        w = sorted_block_sizes[3]
        # mirror the boundary
        extended_vec = np.zeros(len(annotation_vec) + 2*w)
        extended_vec[w: -w] = annotation_vec[:]
        for i in range(w):
            extended_vec[i] = annotation_vec[w-1-i]
            extended_vec[-i + 1] = annotation_vec[-w+i]
        for i in range(len(annotation_vec)):
            majority_vote(i, extended_vec, w)
        smoothed_annotation_vec = extended_vec[w: -w]
        return smoothed_annotation_vec
    else:

        return [int(i) for i in annotation_vec]


def gen_annotation(left_speed, right_speed, label):
    if label == 'left':
        threshold = max(right_speed)
        # threshold = 0.1
        annotation = [i<=threshold for i in left_speed]
        annotation = majority_vote_smoothing(annotation, n_block=3)
    elif label == 'right':
        threshold = max(left_speed)
        # threshold = 0.1
        annotation = [i<=threshold for i in right_speed]
        annotation = majority_vote_smoothing(annotation, n_block=3)
    elif label == 'sideStep':
        threshold = 0.01
        annotation = [i<=threshold for i in right_speed]
        annotation = majority_vote_smoothing(annotation, n_block=3)
    else:
        raise KeyError('Unknown label')
    annotation[0] = 1
    annotation[-1] = 1
    if type(annotation) is not list:
        annotation = annotation.tolist()
    return annotation


def gen_foot_contact_annotation(bvhfile, feature_joints, motion_primitive_model):
    """

    :param bvhfile:
    :param feature_joints: [left joint, right joint]
    :return:
    """
    bvhreader = BVHReader(bvhfile)
    n_frames = len(bvhreader.frames)
    left_joint_speed, right_joint_speed = get_joint_speed(bvhreader, feature_joints)
    start_anno = np.zeros(n_frames)
    start_anno[0] = 1.0
    end_anno = np.zeros(n_frames)
    end_anno[-1] = 1.0
    semantic_annotation = {'LeftFootContact': [],
                           'RightFootContact': [],
                           'start': start_anno.tolist(),
                           'end': end_anno.tolist()}
    if motion_primitive_model == 'leftStance':
        semantic_annotation['RightFootContact'] = np.ones(n_frames).tolist()
        semantic_annotation['LeftFootContact'] = gen_annotation(left_joint_speed, right_joint_speed, 'left')
    elif motion_primitive_model == 'rightStance':
        semantic_annotation['RightFootContact'] = gen_annotation(left_joint_speed, right_joint_speed, 'right')
        semantic_annotation['LeftFootContact'] = np.ones(n_frames).tolist()
    elif motion_primitive_model == 'sideStep':
        semantic_annotation['RightFootContact'] = gen_annotation(left_joint_speed, right_joint_speed, 'sideStep')
        semantic_annotation['LeftFootContact'] = gen_annotation(left_joint_speed, right_joint_speed, 'sideStep')
    else:
        raise NotImplementedError
    return semantic_annotation
