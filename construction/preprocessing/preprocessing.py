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
Created on Fri Jul 31 14:38:45 2015

@author: Han Du
"""

from .motion_dtw import MotionDynamicTimeWarping


class Preprocessor(MotionDynamicTimeWarping):

    def __init__(self, params, verbose=False):
        super(Preprocessor, self).__init__(verbose)
        self.params = params

    def preprocess(self):
        # self.segment_motions(self.params.elementary_action,
        #                      self.params.motion_primitive,
        #                      self.params.retarget_folder,
        #                      self.params.annotation_file)
        self.normalize_root(self.params.ref_position)
        self.align_motion_by_vector(self.params.align_frame_idx,
                          self.params.ref_orientation)
        # self.correct_up_axis(self.params.align_frame_idx,
        #                      self.params.ref_up_vector)
        self.dtw()

    def save_result(self, save_path):
        self.save_warped_motion(save_path)
