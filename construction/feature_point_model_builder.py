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

from .feature_point_model import FeaturePointModel
import json
import os


class FeaturePointModelBuilder():
    def __init__(self):
        self.morphable_model_directory = None
        self.n_samples = 10000


    def set_config(self, config_file_path):
        config_file = open(config_file_path)
        config = json.load(config_file)
        self.morphable_model_directory = config["model_data_dir"]
        self.skeleton_file = config["skeleton_file_dir"]
        self.n_samples = config["n_samples"]

    def build(self):
        for root, dirs, files in os.walk(self.morphable_model_directory):
            for file in files:
                if "_quaternion_mm.json" in file:
                    segments = file.split('_')
                    elementary_action = segments[0]
                    motion_primitive = segments[1]
                    model = FeaturePointModel(root + os.sep + file,
                                              self.skeleton_file)
                    model.create_root_pos_ori(self.n_samples)
                    model.convert_ori_to_angle_deg()
                    model.model_root_dist()
                    save_filename = root + os.sep + '_'.join([elementary_action,
                                                              motion_primitive,
                                                              "root_dist_angle.json"])
                    model.save_root_feature_dist(save_filename)
