#!/usr/bin/env python
#
# Copyright 2019 DFKI GmbH, Daimler AG
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
Created on Thu Jul 23 10:06:37 2015

@author: Han Du, Markus Mauer
"""

import os
import glob
import json
from anim_utils.animation_data import BVHReader, BVHWriter, SkeletonBuilder
from .motion_normalization import MotionNormalization
from anim_utils.animation_data.utils import calculate_frame_distance
import numpy as np
#import rpy2.robjects.numpy2ri as numpy2ri
#import rpy2.robjects as robjects
import matplotlib.pyplot as plt


class MotionDynamicTimeWarping(MotionNormalization):

    def __init__(self, verbose=False):
        super(MotionDynamicTimeWarping, self).__init__()
        self.ref_motion = {}
        self.dic_distgrid = {}
        self.aligned_motions = {}
        self.verbose = verbose
        self.len_aligned_motions = 0
        self.warped_motions = {}
        self.ref_bvhreader = None

    def load_motion_from_files_for_DTW(self, folder_path):
        if not folder_path.endswith(os.sep):
            folder_path += os.sep
        print("search bvh files in " + folder_path)
        motion_files = glob.glob(folder_path + '*.bvh')
        print(str(len(motion_files)) + " are found!")
        for bvh_file_path in motion_files:
            bvhreader = BVHReader(bvh_file_path)
            filename = os.path.split(bvh_file_path)[-1]
            self.aligned_motions[filename] = bvhreader.frames
        self.ref_bvhreader = bvhreader

    def dtw(self):
        if self.ref_motion == {}:
            print("automatically search best reference motion")
            self.find_ref_motion()
        self.warp_all_motions_to_ref_motion()

    def find_ref_motion(self):
        """The reference motion can be found by using the motion which has
           minimal average distance to others
        """
        self.len_aligned_motions = len(self.aligned_motions)
        self.get_all_distgrid()
        # calculate motion distance from distgrid
        average_dists = {}
        counter = 0
        for ref_filename in list(self.dic_distgrid.keys()):
            average_dist = 0
            for test_filename in list(self.dic_distgrid[ref_filename].keys()):
                res = MotionDynamicTimeWarping.calculate_path(
                    self.dic_distgrid[ref_filename][test_filename])
                average_dist += res[2]
            average_dist = average_dist / self.len_aligned_motions
            counter += 1
            average_dists[ref_filename] = average_dist
            print(counter, '/', self.len_aligned_motions)
        self.ref_motion['filename'] = min(average_dists, key=lambda k: average_dists[k])
        self.ref_motion['frames'] = self.aligned_motions[self.ref_motion['filename']]

    def set_ref_motion(self, filepath):
        ref_filename = os.path.split(filepath)[-1]
        ref_bvh = BVHReader(filepath)
        self.ref_motion['filename'] = ref_filename
        self.ref_motion['frames'] = ref_bvh.frames

    def warp_test_motion_to_ref_motion(self, ref_motion, test_motion):
        distgrid = self.get_distgrid(ref_motion, test_motion)
        res = MotionDynamicTimeWarping.calculate_path(distgrid)
        ref_indeces = res[0]
        test_indeces = res[1]
        warping_index = self.get_warping_index(ref_indeces, test_indeces,
                                               (int(ref_indeces[-1]), int(test_indeces[-1])))
        warped_frames = MotionDynamicTimeWarping.get_warped_frames(warping_index,
                                                                   test_motion['frames'])
        return warped_frames, warping_index

    def warp_all_motions_to_ref_motion(self):
        if self.ref_motion == {}:
            raise ValueError('There is no reference motion for DTW')
        if self.aligned_motions == {}:
            raise ValueError('No motion for DTW')
        for filename, frames in self.aligned_motions.items():
            print(filename)
            test_motion = {}
            test_motion['filename'] = filename
            test_motion['frames'] = frames
            warped_frames, warping_index = self.warp_test_motion_to_ref_motion(self.ref_motion,
                                                                               test_motion)
            self.warped_motions[filename] = {}
            self.warped_motions[filename]['warping_index'] = warping_index
            self.warped_motions[filename]['frames'] = warped_frames

    def save_warped_motion(self, save_path):
        if not save_path.endswith(os.sep):
            save_path += os.sep
        skeleton = Skeleton(self.ref_bvhreader)
        warping_index_dic = {}
        for filename, motion_data in self.warped_motions.items():
            BVHWriter(save_path + filename, skeleton,
                      motion_data['frames'],
                      frame_time=self.ref_bvhreader.frame_time,
                      is_quaternion=False)
            warping_index_dic[filename] = np.array(motion_data['warping_index']).tolist()
        with open(save_path + 'timewarping.json', 'wb') as outfile:
            json.dump(warping_index_dic, outfile)

    @classmethod
    def get_warped_frames(cls, warping_index, frames):

        if warping_index[-1] > len(frames):
            raise ValueError('index is larger than length of frames!')
        warped_frames = []
        for idx in warping_index:
            warped_frames.append(frames[idx])
        return warped_frames

    def get_distgrid(self, ref_motion, test_motion):
        skeleton = SkeletonBuilder().load_from_bvh(self.ref_bvhreader)
        n_ref_frames = len(ref_motion['frames'])
        n_test_frames = len(test_motion['frames'])
        distgrid = np.zeros([n_test_frames, n_ref_frames])
        for i in range(n_test_frames):
            for j in range(n_ref_frames):
                distgrid[i, j] = calculate_frame_distance(skeleton,
                                                          ref_motion['frames'][j],
                                                          test_motion['frames'][i])
        if self.verbose:
            res = MotionDynamicTimeWarping.calculate_path(distgrid)
            ref_indices = res[0]
            test_indices = res[1]
            shape = (n_test_frames, n_ref_frames)
            path = self.get_warping_index(test_indices, ref_indices, shape)
            distgrid = distgrid.T
            plt.figure()
            plt.imshow(distgrid)
            plt.plot(list(range(len(path))), path, color='red')
            plt.ylabel(ref_motion['filename'])
            plt.ylim(0, n_ref_frames)
            plt.xlim(0, n_test_frames)
            plt.xlabel(test_motion['filename'])
            plt.title('similarity grid with path')
            plt.show()
        return distgrid

    def get_warping_index(self, ref_indices, test_indices, shape):
        """ @brief Calculate the warping index from a given set of x and y values

        Calculate the warping path from the return values of the dtw R function
        This R functions returns a set of (x, y) pairs saved as x vecotr and
        y vector. These pairs are used to initialize a Bitmatrix representing
        the Path through the Distance grid.
        The indexes for the testmotion is than calculated based on this matrix.

        @param test_indices list of ints - The x-values
        @param ref_indices list of ints - The y-values
        @param shape array or tuple with two elements - The shape of the distgrid,\
            normaly (testmotion_framenumber, refmotion_framenumber)
        @param verbose (Optional) Displays the warping indexes

        @return A list with exactly refmotion_framenumber Elements.
        """
        # create Pairs:
        path_pairs = [(int(ref_indices[i]) - 1, int(test_indices[i]) - 1)
                      for i in range(len(ref_indices))]
        # create Pathmatirx:
        pathmatrix = np.zeros(shape)
        for pair in path_pairs:
            pathmatrix[pair] = 1
        warping_index = []
        for i in range(shape[1]):
            index = np.nonzero(pathmatrix[:, i])[0][-1]
            warping_index.append(index)

        if self.verbose:
            print("warping index from R is: ")
            print(warping_index)
        return warping_index

    @classmethod
    def calculate_path(cls, distgrid, steppattern="typeIb", window="itakura"):
        """ @brief Calculates the optimal path through the given Distance grid

        Calculates an optimal path through the given Distance grid with
        the R Package "dtw". The path restrictions can be varried with the
        steppattern and window parameter
        !!! NOTE: This package is distributed under the GPL(v2) Version !!!

        @param distgrid arraylike object with shape
            (testmotion_framenumber, refmotion_framenumber) -
            The calculated distance grid
        @param steppattern string - The steppattern to be used.
            The steppattern is normaly used to define local constraints
            See "http://cran.r-project.org/web/packages/dtw/dtw.pdf" for a detailed
            list of available options
        @param window string - The window to be used.
            The window is normaly used to define global constraints
            Available options are: "none", "itakura", "sakoechiba", "slantedband"
            See "http://cran.r-project.org/web/packages/dtw/dtw.pdf" for a detailed
            description

        @return numpy array - matched elements: indices in x
        @return numpy array - corresponding mapped indices in y
        @return float - the minimum global distance computed. normalized for path
            length, if normalization is
            known for chosen step pattern
        """

        robjects.conversion.py2ri = numpy2ri.numpy2ri
        distgrid = np.asarray(distgrid)
        max_len = float(max(distgrid.shape))
        min_len = float(min(distgrid.shape))
        ratio = max_len / min_len
        if ratio > 1.5:
            steppattern = 'symmetric2'
            window = 'none'
        rdistgrid = robjects.Matrix(np.asarray(distgrid))
        rcode = '''
                library("dtw")

                path = dtw(x = as.matrix(%s), y = NULL,
                           dist.method="Euclidean",
                           step.pattern = %s,
                           window.type = "%s",
                           keep.internals = FALSE,
                           distance.only = FALSE,
                           open.end = FALSE,
                           open.begin = FALSE)
                xindex = path$index1
                yindex = path$index2
                dist = path$distance

                ''' % (rdistgrid.r_repr(), steppattern, window)

        robjects.r(rcode)

        return np.array(robjects.globalenv['xindex']), \
            np.array(robjects.globalenv['yindex']), \
            np.array(robjects.globalenv['dist'])[0]

    def get_all_distgrid(self):
        """calculate the distance matrix for each pair of motions in
           aligned_motions
        """
        print("start to compute distance grid for all pairs pf motions")
        total_calculation = self.len_aligned_motions * \
            (self.len_aligned_motions - 1) / 2
        print("There are %d pairs in total" % total_calculation)
        counter = 0
        keys = list(self.aligned_motions.keys())
        for i in range(self.len_aligned_motions):
            for j in range(i + 1, self.len_aligned_motions):
                counter += 1
                print(counter, '/', total_calculation)
                ref_motion = {'filename': keys[i],
                              'frames': self.aligned_motions[keys[i]]}
                test_motion = {'filename': keys[j],
                               'frames': self.aligned_motions[keys[j]]}
                distgrid = self.get_distgrid(ref_motion,
                                             test_motion)
                try:
                    self.dic_distgrid[keys[i]][keys[j]] = distgrid
                except KeyError:
                    self.dic_distgrid[keys[i]] = {}
                    self.dic_distgrid[keys[i]][keys[j]] = distgrid
                try:
                    self.dic_distgrid[keys[j]][keys[i]] = distgrid
                except KeyError:
                    self.dic_distgrid[keys[j]] = {}
                    self.dic_distgrid[keys[j]][keys[i]] = distgrid
