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
import os
import glob
import json
from . import gen_foot_contact_annotation
from anim_utils.animation_data import BVHReader
from anim_utils.utilities.io_helper_functions import write_to_json_file
from ..utils import get_aligned_data_folder
import numpy as np



def create_low_level_semantic_annotation(elementary_action,
                                         motion_primitive,
                                         repo_dir=None):
    if repo_dir is None:
        repo_dir = r'C:\repo'
    if 'pick' in elementary_action.lower() or 'place' in elementary_action.lower():
        print(('create synthetic semantic annotation for ' + elementary_action))
        gen_synthetic_semantic_annotation_pick_and_place(elementary_action,
                                                         motion_primitive)
    elif 'walk' in elementary_action.lower() or 'carry' in elementary_action.lower():
        print(('create synthetic semantic annotation for ' + elementary_action))
        gen_walk_annotation(elementary_action,
                            motion_primitive)
    elif 'screw' in elementary_action.lower():
        print(('create synthetic semnatic annotation for ' + elementary_action))
        gen_synthetic_semantic_annotation_for_screw(elementary_action,
                                                    motion_primitive)
    elif 'transfer' in elementary_action.lower():
        print(('create synthetic semnatic annotation for ' + elementary_action))
        gen_synthetic_semantic_annotation_for_transfer(elementary_action,
                                                       motion_primitive)
    else:
        raise KeyError('Unknow action type')


def gen_walk_annotation(elementary_action, motion_primitive):

    aligned_data_folder = get_aligned_data_folder(elementary_action,
                                                  motion_primitive)
    if 'rightstance' in motion_primitive.lower():

        motion_primitive_model = 'rightStance'
    elif 'leftstance' in motion_primitive.lower():
        motion_primitive_model = 'leftStance'
    elif 'sidestep' in motion_primitive.lower():
        motion_primitive_model = 'sideStep'
    elif 'turnleft' in motion_primitive.lower():
        motion_primitive_model = 'rightStance'
    elif 'turnright' in motion_primitive.lower():
        motion_primitive_model = 'leftStance'
    else:
        raise KeyError('unknown motion type!')
    feature_points = ['Bip01_L_Toe0', 'Bip01_R_Toe0']
    annotation_data = {}
    aligned_bvhfiles = glob.glob(aligned_data_folder + os.sep + '*.bvh')
    for item in aligned_bvhfiles:
        filename = os.path.split(item)[-1]
        anno_vec = gen_foot_contact_annotation(item, feature_points, motion_primitive_model)
        annotation_data[filename] = anno_vec
    output_filename = aligned_data_folder + os.sep + elementary_action + '_' + motion_primitive + '_semantic_annotation.json'
    semantic_annotation = {'annotation_list': ['LeftFootContact', 'RightFootContact', 'start', 'end'],
                           'data': annotation_data}
    # with open(output_filename, 'w') as outfile:
    #     json.dump(semantic_annotation, outfile)
    write_to_json_file(output_filename, semantic_annotation)

def gen_synthetic_semantic_annotation_pick_and_place(elementary_action,
                                                     motion_primitive):
    data_folder = get_aligned_data_folder(elementary_action,
                                          motion_primitive)
    semantic_annotation_data = {'annotation_list': ['leftHandContact', 'rightHandContact', 'leftFootContact',
                                                    'rightFootContact', 'start', 'end'],
                                'data':{}}
    bvhfiles = glob.glob(data_folder + os.sep + '*.bvh')
    bvhreader = BVHReader(bvhfiles[0])
    n_frames = len(bvhreader.frames)
    for item in bvhfiles:
        filename = os.path.split(item)[-1]
        leftHand = np.zeros(n_frames)
        rightHand = np.zeros(n_frames)
        leftFoot = np.ones(n_frames)
        rightFoot = np.ones(n_frames)
        start = np.zeros(n_frames)
        start[0] = 1.0
        end = np.zeros(n_frames)
        end[-1] = 1.0
        if motion_primitive == 'first' and 'left' in elementary_action.lower():
            leftHand[-1] = 1
        elif motion_primitive == 'first' and 'right' in elementary_action.lower():
            rightHand[-1] = 1
        elif motion_primitive == 'first' and 'both' in elementary_action.lower():
            leftHand[-1] = 1
            rightHand[-1] = 1
        elif motion_primitive == 'second' and 'left' in elementary_action.lower():
            leftHand[:] = 1
        elif motion_primitive == 'second' and 'right' in elementary_action.lower():
            rightHand[:] = 1
        elif motion_primitive == 'second' and 'both' in elementary_action.lower():
            leftHand[:] = 1
            rightHand[:] = 1
        else:
            ValueError('motion primitive type is not support!')
        semantic_annotation_data['data'][filename] = {'leftHandContact': leftHand.tolist(),
                                                      'rightHandContact': rightHand.tolist(),
                                                      'leftFootContact': leftFoot.tolist(),
                                                      'rightFootContact': rightFoot.tolist(),
                                                      'start': start.tolist(),
                                                      'end': end.tolist()}
    output_path = data_folder + os.sep + '_'.join([elementary_action,
                                                   motion_primitive,
                                                   'semantic',
                                                   'annotation.json'])
    with open(output_path, 'w') as outfile:
        json.dump(semantic_annotation_data, outfile)


def gen_synthetic_semantic_annotation_for_screw(elementary_action,
                                                motion_primitive):
    data_folder = get_aligned_data_folder(elementary_action,
                                          motion_primitive)
    # if motion_primitive == 'retrieve':
    #     semantic_annotation_data = {'annotation_list': ['retrieveTool', 'start', 'end'],
    #                                 'data': {}}
    #     bvhfiles = glob.glob(os.path.join(data_folder, '*.bvh'))
    #     bvhreader = BVHReader(bvhfiles[0])
    #     n_frames = len(bvhreader.frames)
    #     for filename in bvhfiles:
    #
    # semantic_annotation_data = {'annotation_list': ['fastenerContact', 'toolActivity', 'retrieveTool'],
    #                             'data': {}}
    # if
    semantic_annotation_data = {'annotation_list': ['start', 'end'],
                                'data': {}}
    bvhfiles = glob.glob(os.path.join(data_folder,'*.bvh'))
    bvhreader = BVHReader(bvhfiles[0])
    n_frames = len(bvhreader.frames)
    for item in bvhfiles:
        filename = os.path.split(item)[-1]
        start = np.zeros(n_frames)
        start[0] = 1.0
        end = np.zeros(n_frames)
        end[-1] = 1.0
        semantic_annotation_data['data'][filename] = {'start': start.tolist(),
                                                      'end': end.tolist()}
    output_path = data_folder + os.sep + '_'.join([elementary_action,
                                                   motion_primitive,
                                                   'semantic',
                                                   'annotation.json'])
    with open(output_path, 'w') as outfile:
        json.dump(semantic_annotation_data, outfile)


def gen_synthetic_semantic_annotation_for_transfer(elementary_action,
                                                   motion_primitive):
    data_folder = get_aligned_data_folder(elementary_action,
                                          motion_primitive)


if __name__ == '__main__':
    gen_walk_annotation('walk', 'beginRightStance')