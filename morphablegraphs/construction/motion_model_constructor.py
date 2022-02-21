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
import numpy as np
from copy import copy
import collections
from datetime import datetime
from transformations import quaternion_from_euler
from anim_utils.animation_data import BVHWriter
from .fpca.fpca_spatial_data import FPCASpatialData
from .fpca.fpca_time_semantic import FPCATimeSemantic
from .motion_primitive.statistical_model_trainer import GMMTrainer
from anim_utils.animation_data.utils import pose_orientation_quat, get_rotation_angle
from .dtw import get_warping_function, find_optimal_dtw, find_optimal_dtw_async, warp_motion
from ..motion_model.motion_spline import MotionSpline
from .utils import _convert_pose_to_point_cloud, rotate_frames, align_quaternion_frames, get_cubic_b_spline_knots,\
                  normalize_root_translation, scale_root_translation_in_fpca_data, gen_gaussian_eigen, BSPLINE_DEGREE
from os import cpu_count
import asyncio
from concurrent.futures import ProcessPoolExecutor

def convert_poses_to_point_clouds(params):
    skeleton, keys, motions, normalize = params
    point_clouds = dict()
    for k in keys:
        point_cloud = []
        for f in motions[k]:
            p = _convert_pose_to_point_cloud(skeleton, f, normalize)
            point_cloud.append(p)
        point_clouds[k] = point_cloud
    return point_clouds


@asyncio.coroutine
def run_conversion_coroutine(pool, params, results):
    print("start task")
    fut = pool.submit(convert_poses_to_point_clouds, params)
    while not fut.done() and not fut.cancelled():
        yield from asyncio.sleep(0.1)
    print("done")
    results.update(fut.result())

def chunks(l, n):
    """Yield successive n-sized chunks from l.
    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks"""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def convert_motions_to_point_clouds_parallel(skeleton, motions, normalize=False):
    n_workers = max(cpu_count()-1, 1)
    pool = ProcessPoolExecutor(max_workers=n_workers)
    n_motions = len(motions)
    if n_workers > n_motions:
        n_workers = n_motions
    n_batches = int(len(motions) / n_workers)
    tasks = []
    results = dict()
    for batch_keys in chunks(list(motions.keys()), n_batches):
        print("start batch", len(batch_keys))
        t = run_conversion_coroutine(pool, (skeleton, batch_keys, motions,  normalize), results)
        tasks.append(t)
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))
    return results


def export_frames_to_bvh_file(output_dir, skeleton, frames, prefix="", time_stamp=True, is_quaternion=True):
    """ Exports a list of frames to a bvh file

    Parameters
    ---------
    * output_dir : string
        directory without trailing os.sep
    * skeleton : Skeleton
        contains joint hiearchy information
    * frames : np.ndarray
        Represents the motion

    """
    if len(frames) > 0 and len(frames[0]) < skeleton.reference_frame_length:
        frames = skeleton.add_fixed_joint_parameters_to_motion(frames)
    bvh_writer = BVHWriter(None, skeleton, frames, skeleton.frame_time, is_quaternion)
    if time_stamp:
        filepath = output_dir + os.sep + prefix + "_" + \
            str(datetime.now().strftime("%d%m%y_%H%M%S")) + ".bvh"
    elif prefix != "":
        filepath = output_dir + os.sep + prefix + ".bvh"
    else:
        filepath = output_dir + os.sep + "output" + ".bvh"
    bvh_writer.write(filepath)


class MotionModel(object):
    def __init__(self, skeleton):
        self._skeleton = skeleton
        self._input_motions = dict()
        self._aligned_frames = dict()
        self._temporal_data = dict()
        self._spatial_fpca_data = None
        self._temporal_fpca_data = None
        self._gmm_data = None

    def back_project_sample(self, alpha):
        coeffs = np.dot(self._spatial_fpca_data["eigenvectors"].T, alpha)
        coeffs += self._spatial_fpca_data["mean"]
        coeffs = coeffs.reshape((self._spatial_fpca_data["n_basis"], self._spatial_fpca_data["n_dim"]))
        # undo the scaling on the translation
        translation_maxima = self._spatial_fpca_data["scale_vec"]
        coeffs[:, 0] *= translation_maxima[0]
        coeffs[:, 1] *= translation_maxima[1]
        coeffs[:, 2] *= translation_maxima[2]
        return coeffs

    def export_sample(self, name, alpha):
        spatial_coeffs = self.back_project_sample(alpha)
        self.export_coeffs(name, spatial_coeffs)

    def export_coeffs(self, name, spatial_coeffs):
        n_frames = len(self._aligned_frames[0])
        time_function = np.arange(0, n_frames)

        spatial_knots = get_cubic_b_spline_knots(self._spatial_fpca_data["n_basis"], n_frames).tolist()
        spline = MotionSpline(0, spatial_coeffs, time_function, spatial_knots, None)
        frames = spline.get_motion_vector()
        self._export_frames(name, frames)

    def _export_frames(self, name, frames):
        frames = np.array([self._skeleton.add_fixed_joint_parameters_to_frame(f) for f in frames])
        export_frames_to_bvh_file(".", self._skeleton, frames, name, time_stamp=False)


class MotionModelConstructor(MotionModel):
    def __init__(self, skeleton, config):
        super(MotionModelConstructor, self).__init__(skeleton)
        self.skeleton = skeleton
        self.config = config
        self.use_multi_processing = True
        if "use_multi_processing" in config:
            self.use_multi_processing = config["use_multi_processing"]
        self.temp_data_dir = None
        if "temp_data_dir" in config:
            self.temp_data_dir = config["temp_data_dir"]
        self.ref_orientation = [0,-1]  # look into -z direction in 2d
        self._dtw_sections = None
        self._keyframes = dict()
        self._temporal_data = None
        self._aligned_frames = None

    def set_motions(self, motions):
        """ Set the input data.
        Args:
        -----
             motions (List): input motion data in quaternion format.
        """
        self._input_motions = motions

    def set_dtw_sections(self, dtw_sections):
        """ Set sections input data.
        Args:
        -----
             dtw_sections (List): list of dictionaries with "start_idx" and "end_idx".
        """
        self._dtw_sections = dtw_sections
        self._keyframes = dict()

    def set_aligned_frames(self, motions, keyframes=None):
        """ Set pre aligned input data.
        Args:
        -----
             motions (List): input motion data in quaternion format.
             keyframes (dict): map from label to frame index.
        """
        self._aligned_frames = motions
        if keyframes is not None:
            self._keyframes = keyframes

    def set_timewarping(self, temporal_data):
        self._temporal_data = temporal_data
    
    def construct_model(self, name, version=1, save_skeleton=False, mean_key=None, align_frames=True):
        """ Runs the construction pipeline
        Args:
        -----
             name (string): name of the motion primitive
             version (int): format supported values are 1, 2 and 3
             save_skeleton (bool): add skeleton to model json string
             mean_key (string): optional mean motion for temporal alignhment to avoid seach
             align_frames (bool): optionally disable alignment if frames are already aligned
        """
        if align_frames or self._temporal_data is None or self._aligned_frames is None:
            self._align_frames(mean_key)
        self.run_dimension_reduction()
        self.learn_statistical_model()
        model_data = self.convert_motion_model_to_json(name, version, save_skeleton)
        return model_data

    def _align_frames(self, mean_key=None):
        aligned_frames = self._align_frames_spatially(self._input_motions)
        print("aligned", len(aligned_frames), len(self._input_motions))
        if self._temporal_data is not None:
            self._aligned_frames = aligned_frames
            temp = collections.OrderedDict()
            for key in self._aligned_frames.keys():
                if key in self._temporal_data:
                    temp[key] = self._temporal_data[key]
            self._temporal_data = temp
        elif self._dtw_sections is not None:
            self._aligned_frames, self._temporal_data = self._align_frames_temporally_split(aligned_frames,
                                                                                            self._dtw_sections,
                                                                                            mean_key=mean_key)
        else:
            self._aligned_frames, self._temporal_data = self._align_frames_temporally(aligned_frames, mean_key)
        print("finished processing",len(self._aligned_frames), len(self._temporal_data))
        if self.temp_data_dir is not None:
            self._export_aligned_frames()
            np.save(self.temp_data_dir+os.sep+"temporal_data.npy", self._temporal_data)

    def _export_aligned_frames(self):
        for key, frames in self._aligned_frames.items():
            print(np.array(frames).shape)
            name = self.temp_data_dir+os.sep+"aligned"+str(key)
            self._export_frames(name, frames)

    def _align_frames_spatially(self, input_motions):
        print("run spatial alignment", self.ref_orientation)
        aligned_frames = collections.OrderedDict()
        frame_idx = 0
        for key, input_m in input_motions.items():
            ma = input_m[:]

            # align orientation to reference orientation
            m_orientation = pose_orientation_quat(ma[frame_idx])
            rot_angle = get_rotation_angle(self.ref_orientation, m_orientation)
            e = np.deg2rad([0, rot_angle, 0])
            q = quaternion_from_euler(*e)
            ma = rotate_frames(ma, q)

            # normalize position
            delta = copy(ma[0, :3])
            for f in ma:
                f[:3] -= delta# + self._skeleton.nodes[self._skeleton.root].offset
            aligned_frames[key] = ma
        return aligned_frames

    def get_average_time_line(self, input_motions):
        n_frames = [len(m) for m in input_motions.values()]
        mean = np.mean(n_frames)
        best_key = None
        least_distance = np.inf
        for k, m in input_motions.items():
            d = abs(len(m)-mean)
            if d < least_distance:
                best_key = k
                least_distance = d
        return best_key

    def _align_frames_temporally(self, input_motions, mean_key=None):
        print("run temporal alignment", self.use_multi_processing)
        print("convert motions to point clouds")
        if self.use_multi_processing:
            point_clouds = convert_motions_to_point_clouds_parallel(self._skeleton, input_motions, normalize=False)
        else:
            point_clouds = convert_poses_to_point_clouds((self._skeleton, input_motions.keys(), input_motions, False))
        print("find reference motion", len(point_clouds))
        if mean_key is None:
            mean_key = self.get_average_time_line(input_motions)
            print("set reference to index", mean_key, "of", len(input_motions), "motions")
        if self.use_multi_processing:
            print("run dtw using multiple processes")
            dtw_results = find_optimal_dtw_async(point_clouds, mean_key)
        else:
            print("run dtw in single process")
            dtw_results = find_optimal_dtw(point_clouds)
        warped_frames = collections.OrderedDict()
        warping_functions = collections.OrderedDict()
        for k, m in input_motions.items():
            path = dtw_results[k]
            warping_function = get_warping_function(path)
            warped_motion = warp_motion(m, warping_function)
            warped_frames[k] = np.array(warped_motion)
            warping_functions[k] = warping_function
        return warped_frames, warping_functions

    def _align_frames_temporally_split(self, input_motions, sections=None, mean_key=None):
        if mean_key is None:
            mean_key = self.get_average_time_line(input_motions)
        print("set mean key to ", mean_key, len(input_motions))
        
        if sections is not None:
            #print("set reference to index", mean_key, "of", len(input_motions), "motions", sections[mean_key])
            # use segment end as keyframe
            for i, s in enumerate(sections[mean_key]):
                self._keyframes["contact"+str(i)] = s["end_idx"]
        # split_motions into sections
        if sections is not None:
            key = list(input_motions.keys())[0]
            n_sections = len(sections[key])
            splitted_motions = [collections.OrderedDict() for section_idx in range(n_sections)]
            for key, input_motion in input_motions.items():
                splitted_motion = []
                for section_idx, section in enumerate(sections[key]):
                    start_idx = section["start_idx"]
                    end_idx = section["end_idx"]
                    split = input_motion[start_idx:end_idx]
                    print("split motion", key, len(splitted_motion))
                    splitted_motions[section_idx][key] = split
        else:
            splitted_motions = [input_motions]

        # run dtw for each section
        splitted_dtw_results = []
        for section_samples in splitted_motions:
            result = self._align_frames_temporally(section_samples, mean_key)
            splitted_dtw_results.append(result)

        # combine sections
        warped_frames = collections.OrderedDict()
        warping_functions = collections.OrderedDict()
        for key in input_motions.keys():
            combined_frames = []
            combined_warping_function = []
            for section_idx, result in enumerate(splitted_dtw_results):
                print(key, section_idx, len(result),len(result[0]))
                combined_frames += list(result[0][key])
                combined_warping_function += list(result[1][key])
            warped_frames[key] = combined_frames
            warping_functions[key] = combined_warping_function

        return warped_frames, warping_functions

    def run_dimension_reduction(self):
        self.run_spatial_dimension_reduction()
        self.run_temporal_dimension_reduction()

    def run_spatial_dimension_reduction(self):
        key = list(self._aligned_frames.keys())[0]
        n_basis = len(self._aligned_frames[key]) * self.config["n_spatial_basis_factor"]
        print("use", n_basis, "basis functions for fpca")
        scaled_quat_frames, scale_vec = normalize_root_translation(self._aligned_frames)
        smoothed_quat_frames = align_quaternion_frames(self._skeleton, scaled_quat_frames)

        fpca_spatial = FPCASpatialData(int(n_basis),
                                       self.config["n_components"],
                                       self.config["fraction"])
        fpca_spatial.fileorder = smoothed_quat_frames.keys()#list(range(len(smoothed_quat_frames)))
        #print(smoothed_quat_frames.shape)
        input_data = np.array(list(smoothed_quat_frames.values()))

        n_samples = len(input_data)
        n_frames = len(input_data[0])
        n_dims = len(input_data[0][0])
        input_data = input_data.reshape((n_samples, n_frames, n_dims))
        print(fpca_spatial.fileorder, input_data.shape)
        fpca_spatial.fit(input_data)

        result = dict()
        result['parameters'] = fpca_spatial.fpcaobj.low_vecs
        result['file_order'] = fpca_spatial.fileorder
        result['n_basis'] = fpca_spatial.fpcaobj.n_basis
        eigenvectors = fpca_spatial.fpcaobj.eigenvectors
        mean = fpca_spatial.fpcaobj.mean
        data = fpca_spatial.fpcaobj.functional_data
        result["n_coeffs"] = len(data[0])
        result['n_dim'] = len(data[0][0])
        result["scale_vec"] = [1,1,1]
        mean, eigenvectors = scale_root_translation_in_fpca_data(mean,
                                                                 eigenvectors,
                                                                 scale_vec,
                                                                 result['n_coeffs'],
                                                                 result['n_dim'])
        result['mean'] = mean
        result['eigenvectors'] = eigenvectors
        self._spatial_fpca_data = result

    def run_temporal_dimension_reduction(self):
        fpca_temporal = FPCATimeSemantic(self.config["n_basis_functions_temporal"],
                                          n_components_temporal=self.config["npc_temporal"],
                                          precision_temporal=self.config["precision_temporal"])
        print(np.array(self._temporal_data.values()).shape)
        input_data = []
        print(len(self._temporal_data))
        for k in self._temporal_data.keys():
            input_data.append((self._temporal_data[k]))
        input_data = np.array(input_data)
        #input_data = np.array(self._temporal_data.values())
        fpca_temporal.temporal_semantic_data = input_data
        fpca_temporal.semantic_annotation_list = []
        fpca_temporal.functional_pca()
        result = dict()
        result['eigenvectors'] = fpca_temporal.eigenvectors
        result['mean'] = fpca_temporal.mean_vec
        result['parameters'] = fpca_temporal.lowVs
        result['n_basis'] = fpca_temporal.n_basis
        result['n_dim'] = len(fpca_temporal.semantic_annotation_list)+1
        result['semantic_annotation'] = fpca_temporal.semantic_annotation_list
        self._temporal_fpca_data = result

    def learn_statistical_model(self):
        if self._temporal_fpca_data is not None:

            spatial_parameters = self._spatial_fpca_data["parameters"]
            temporal_parameters = self._temporal_fpca_data["parameters"]
            print(spatial_parameters, temporal_parameters)
            motion_parameters = np.concatenate((spatial_parameters, temporal_parameters,),axis=1)
        else:
            motion_parameters = self._spatial_fpca_data["parameters"]
        trainer = GMMTrainer()
        trainer.fit(motion_parameters)
        self._gmm_data = trainer.convert_model_to_json()

    def convert_motion_model_to_json(self, name="", version=1, save_skeleton=False):
        weights = self._gmm_data['gmm_weights']
        means = self._gmm_data['gmm_means']
        covars = self._gmm_data['gmm_covars']

        key = list(self._aligned_frames.keys())[0]
        n_frames = len(self._aligned_frames[key])

        mean_motion = self._spatial_fpca_data["mean"].tolist()
        spatial_eigenvectors = self._spatial_fpca_data["eigenvectors"].tolist()
        scale_vec = self._spatial_fpca_data["scale_vec"]
        n_dim_spatial = self._spatial_fpca_data["n_dim"]
        n_basis_spatial = self._spatial_fpca_data["n_basis"]
        spatial_knots = get_cubic_b_spline_knots(n_basis_spatial, n_frames).tolist()

        if self._temporal_fpca_data is not None:
            temporal_mean = self._temporal_fpca_data["mean"].tolist()
            temporal_eigenvectors = self._temporal_fpca_data["eigenvectors"].tolist()
            n_basis_temporal = self._temporal_fpca_data["n_basis"]
            temporal_knots = get_cubic_b_spline_knots(n_basis_temporal, n_frames).tolist()
            semantic_label = self._temporal_fpca_data["semantic_annotation"]
        else:
            temporal_mean = []
            temporal_eigenvectors = []
            n_basis_temporal = 0
            temporal_knots = []
            semantic_label = dict()

        if version == 1:
            data = {'name': name,
                    'gmm_weights': weights,
                    'gmm_means': means,
                    'gmm_covars': covars,
                    'eigen_vectors_spatial': spatial_eigenvectors,
                    'mean_spatial_vector': mean_motion,
                    'n_canonical_frames': n_frames,
                    'translation_maxima': scale_vec,
                    'n_basis_spatial': n_basis_spatial,
                    'npc_spatial': len(spatial_eigenvectors),
                    'eigen_vectors_temporal_semantic': temporal_eigenvectors,
                    'mean_temporal_semantic_vector': temporal_mean,
                    'n_dim_spatial': n_dim_spatial,
                    'n_basis_temporal_semantic': n_basis_temporal,
                    'b_spline_knots_spatial': spatial_knots,
                    'b_spline_knots_temporal_semantic':temporal_knots,
                    'npc_temporal_semantic': self.config["npc_temporal"],
                    'semantic_annotation': {},
                    'n_dim_temporal_semantic': 1}
        elif version == 2:
            data = {'name': name,
                    'gmm_weights': weights,
                    'gmm_means': means,
                    'gmm_covars': covars,
                    'eigen_vectors_spatial': spatial_eigenvectors,
                    'mean_spatial_vector': mean_motion,
                    'n_canonical_frames': n_frames,
                    'translation_maxima': scale_vec,
                    'n_basis_spatial': n_basis_spatial,
                    'eigen_vectors_time': temporal_eigenvectors,
                    'mean_time_vector': temporal_mean,
                    'n_dim_spatial': n_dim_spatial,
                    'n_basis_time': n_basis_temporal,
                    'b_spline_knots_spatial': spatial_knots,
                    'b_spline_knots_time': temporal_knots}
                    #'semantic_label': semantic_label
        else:
            data = dict()
            data['sspm'] = dict()
            data['tspm'] = dict()
            data['gmm'] = dict()
            data['sspm']['eigen'] = spatial_eigenvectors
            data['sspm']['mean'] = mean_motion
            data['sspm']['n_coeffs'] = n_basis_spatial
            data['sspm']['n_dims'] = n_dim_spatial
            data['sspm']['knots'] = spatial_knots
            data['sspm']['animated_joints'] = self._skeleton.animated_joints
            data['sspm']['degree'] = BSPLINE_DEGREE
            data['gmm']['covars'] = covars
            data['gmm']['means'] = means
            data['gmm']['weights'] = weights
            data['gmm']['eigen'] = gen_gaussian_eigen(covars).tolist()
            data['tspm']['eigen'] = temporal_eigenvectors
            data['tspm']['mean'] = temporal_mean
            data['tspm']['n_coeffs'] = n_basis_temporal
            data['tspm']['n_dims'] = 1
            data['tspm']['knots'] = temporal_knots
            data['tspm']['degree'] = BSPLINE_DEGREE
            data['tspm']['semantic_labels'] = semantic_label
            data['tspm']['frame_time'] = self._skeleton.frame_time
        if save_skeleton:
            data["skeleton"] = self.skeleton.to_json()

        data["keyframes"] = self._keyframes
        return data
