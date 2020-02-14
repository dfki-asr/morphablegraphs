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
Created on Tue Jul 21 13:44:05 2015

@author: Han Du
"""

import os
from anim_utils.animation_data.bvh import BVHReader, BVHWriter
import json


class MotionSegmentation(object):

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.aligned_motions = {}
        self.annotation_label = {}
        self.bvhreader = None

    def segment_motions(self, elementary_action, primitive_type,
                        data_path, annotation_file):
        self.load_annotation(annotation_file)
        self.cut_files(elementary_action, primitive_type, data_path)

    def load_annotation(self, annotation_file):
        self.annotation_label = MotionSegmentation._convert_to_json(annotation_file,
                                                                    export=False)
        if self.verbose:
            print("Load %d files." % len(list(self.annotation_label.keys())))

    @classmethod
    def _check_motion_type(cls, elementary_action,
                           primitive_type,
                           primitive_data):
        if primitive_data['elementary_action'] == elementary_action \
           and primitive_data['motion_primitive'] == primitive_type:
            return True
        else:
            return False

    @classmethod
    def _get_annotation_information(cls, data_path, filename, primitive_data):
        file_path = data_path + filename
        if not os.path.isfile(file_path):
            raise IOError(
                'cannot find ' +
                filename +
                ' in ' +
                data_path)
        start_frame = primitive_data['frames'][0]
        end_frame = primitive_data['frames'][1]
        # filename_segments = filename[:-4].split('_')
        return start_frame, end_frame

    def cut_files(self, elementary_action, primitive_type, data_path):
        if not data_path.endswith(os.sep):
            data_path += os.sep
        if self.verbose:
            print("search files in " + data_path)
        for filename, items in self.annotation_label.items():
            for primitive_data in items:
                if MotionSegmentation._check_motion_type(elementary_action,
                                                         primitive_type,
                                                         primitive_data):
                    print("find motion primitive " + elementary_action + '_' \
                          + primitive_type + ' in ' + filename)
                    start_frame, end_frame = \
                        MotionSegmentation._get_annotation_information(data_path,
                                                                       filename,
                                                                       primitive_data)
                    newfilename = elementary_action + '_' + filename[:-4]
                    cutted_frames = self._cut_one_file(data_path + filename,
                                                       start_frame,
                                                       end_frame)
                    outfilename = newfilename + \
                                  '_%s_%d_%d.bvh' % (primitive_type,
                                                     start_frame,
                                                     end_frame)
                    self.aligned_motions[outfilename] = cutted_frames
                else:
                    print("cannot find motion primitive " + elementary_action + '_' \
                          + primitive_type + ' in ' + filename)

    def save_segments(self, save_path=None):
        if save_path is None:
            raise ValueError('Please give saving path!')
        if not save_path.endswith(os.sep):
            save_path += os.sep
        for outfilename, frames in self.aligned_motions.items():
            save_filename = save_path + outfilename
            BVHWriter(save_filename, self.bvhreader, frames,
                      frame_time=self.bvhreader.frame_time,
                      is_quaternion=False)

    def _cut_one_file(self, input_file, start_frame, end_frame):
        self.bvhreader = BVHReader(input_file)
        new_frames = self.bvhreader.frames[start_frame: end_frame]
        return new_frames

    @classmethod
    def _convert_to_json(cls, annotation_file, export=False):
        with open(annotation_file, 'rb') as input_file:
            annotation_data = {}
            current_motion = None
            for line in input_file:
                line = line.rstrip()
                if '.bvh' in line:
                    current_motion = line
                    annotation_data[current_motion] = []
                elif current_motion is not None and line != '' and line != '\n':
                    try:
                        line_split = line.split(' ')
                        tmp = {'elementary_action': line_split[0], 'motion_primitive': line_split[
                            1], 'frames': [int(line_split[2]), int(line_split[3])]}
                        annotation_data[current_motion].append(tmp)
                    except ValueError:
                        raise ValueError("Couldn't process line: %s" % line)
        if export:
            filename = os.path.split(annotation_file)[-1]
            with open(filename[:-4] + '.json', 'w+') as outfile:
                json.dump(annotation_data, outfile)
                outfile.close()
        return annotation_data
