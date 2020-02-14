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
import mgrd as mgrd
from ...utilities import load_json_file
from anim_utils.animation_data import BVHReader, convert_euler_frames_to_quaternion_frames
from ..fpca import FunctionalData
import numpy as np


class QuatSplineConstructor(object):

    def __init__(self, semantic_motion_primitive_file, skeleton_jsonfile):
        self.mm_data = load_json_file(semantic_motion_primitive_file)
        self.skeleton = mgrd.SkeletonJSONLoader(skeleton_jsonfile).load()
        self.motion_primitive = mgrd.MotionPrimitiveModel.load_from_json(self.skeleton,
                                                                         self.mm_data)

    def create_quat_spline_from_bvhfile(self, bvhfile, n_basis, degree=3):
        bvhreader = BVHReader(bvhfile)
        quat_frames = convert_euler_frames_to_quaternion_frames(bvhreader, bvhreader.frames)
        fd = FunctionalData()
        functional_coeffs = fd.convert_motion_to_functional_data(quat_frames, n_basis, degree)
        functional_coeffs = mgrd.asarray(functional_coeffs.tolist())
        knots = fd.knots
        sspm = mgrd.QuaternionSplineModel.load_from_json(self.skeleton, self.mm_data['sspm'])
        sspm.motion_primitive = self.motion_primitive
        coeffs_structure = mgrd.CoeffStructure(len(self.mm_data['sspm']['animated_joints']),
                                               mgrd.CoeffStructure.LEN_QUATERNION,
                                               mgrd.CoeffStructure.LEN_ROOT_POSITION)
        quat_spline = mgrd.QuatSpline(functional_coeffs,
                                      knots,
                                      coeffs_structure,
                                      degree,
                                      sspm)
        return quat_spline

    def create_quat_spline_from_functional_data(self, functional_datamat, knots, degree=3):
        functional_coeffs = mgrd.asarray(functional_datamat)
        knots = mgrd.asarray(knots)
        sspm = mgrd.QuaternionSplineModel.load_from_json(self.skeleton, self.mm_data['sspm'])
        sspm.motion_primitive = self.motion_primitive
        coeffs_structure = mgrd.CoeffStructure(len(self.mm_data['sspm']['animated_joints']),
                                               mgrd.CoeffStructure.LEN_QUATERNION,
                                               mgrd.CoeffStructure.LEN_ROOT_POSITION)
        quat_spline = mgrd.QuatSpline(functional_coeffs,
                                      knots,
                                      coeffs_structure,
                                      degree,
                                      sspm)
        return quat_spline


