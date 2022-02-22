import os
import json
import numpy as np
import scipy.interpolate as si
import collections
import argparse
import glob
from anim_utils.animation_data.bvh import BVHReader
from anim_utils.animation_data import SkeletonBuilder, MotionVector
from anim_utils.animation_data.skeleton_models import SKELETON_MODELS
from anim_utils.utilities.io_helper_functions import load_json_file
from morphablegraphs.construction.motion_model_constructor import MotionModelConstructor
from morphablegraphs.motion_model.motion_primitive_wrapper import MotionPrimitiveModelWrapper
from morphablegraphs.construction.utils import get_cubic_b_spline_knots


MM_FILE_ENDING = "_quaternion_mm.json"


def load_skeleton(file_path, joint_filter=None, scale=1.0):
    target_bvh = BVHReader(file_path)
    bvh_joints = list(target_bvh.get_animated_joints())
    if joint_filter is not None:
        animated_joints = [j for j in bvh_joints if j in joint_filter]
    else:
        print("set default joints")
        animated_joints = bvh_joints
    skeleton = SkeletonBuilder().load_from_bvh(target_bvh, animated_joints)
    skeleton.scale(scale)
    return skeleton


def load_motion_vector_from_bvh_file(bvh_file_path, animated_joints):
    bvh_data = BVHReader(bvh_file_path)
    mv = MotionVector(None)
    mv.from_bvh_reader(bvh_data, filter_joints=False, animated_joints=animated_joints)
    return mv


def load_motion_data(motion_folder, max_count=np.inf, animated_joints=None):
    motions = collections.OrderedDict()
    for root, dirs, files in os.walk(motion_folder):
        for file_name in files:
            if file_name.endswith("bvh"):
                mv = load_motion_vector_from_bvh_file(motion_folder + os.sep + file_name, animated_joints)
                motions[file_name[:-4]] = np.array(mv.frames, dtype=np.float)
                if len(motions) > max_count:
                    break
    return motions


def get_standard_config():
    config = dict()
    config["n_spatial_basis_factor"] = 0.2
    config["n_basis_functions_spatial"] = 16
    config["fraction"] = 0.95
    config["n_basis_functions_temporal"] = 8
    config["npc_temporal"] = 3
    config["n_components"] = None
    config["precision_temporal"] = 0.99
    return config


def export_frames_to_bvh(skeleton, frames, filename):
    print("export", len(frames[0]))
    mv = MotionVector()
    mv.frames = np.array([skeleton.add_fixed_joint_parameters_to_frame(f) for f in frames])
    print(mv.frames.shape)
    mv.export(skeleton, filename, add_time_stamp=False)


def export_motions(skeleton, motions):
    for idx, frames in enumerate(motions):
        export_frames_to_bvh(skeleton, frames, "out" + str(idx))


def define_sections_from_keyframes(motion_names, keyframes):
    sections = []
    for key in motion_names:
        if key not in keyframes:
            continue
        m_sections = []
        keyframe = keyframes[key]
        section = dict()
        section["start_idx"] = 0
        section["end_idx"] = keyframe
        m_sections.append(section)
        section = dict()
        section["start_idx"] = keyframe
        section["end_idx"] = -1
        m_sections.append(section)
        sections.append(m_sections)
    return sections


def smooth_quaternion_frames(skeleton, frames, reference_frame):
    print("smooth", len(frames[0]), len(reference_frame))
    for frame in frames:
        for idx, node in enumerate(skeleton.animated_joints):
            o = idx*4 + 3
            ref_q = reference_frame[o:o+4]
            q = frame[o:o+4]
            if np.dot(q, ref_q) < 0:
                frame[o:o + 4] = -q
    return frames


def define_sections_from_annotations(motion_folder, motions):
    filtered_motions = collections.OrderedDict()
    sections = collections.OrderedDict()
    for key in motions.keys():
        annotations_file = motion_folder + os.sep + key + "_sections.json"
        if os.path.isfile(annotations_file):
            data = load_json_file(annotations_file)
            annotations = data["semantic_annotation"]
            motion_sections = dict()
            for label in annotations:
                annotations[label].sort()
                section = dict()
                section["start_idx"] = annotations[label][0]
                section["end_idx"] = annotations[label][-1]
                motion_sections[section["start_idx"]] = section
            motion_sections = collections.OrderedDict(sorted(motion_sections.items()))
            sections[key] = motion_sections.values()
            filtered_motions[key] = motions[key]

    if len(sections) > 0:
        motions = filtered_motions
        return motions, sections
    else:
        return motions, None


def convert_motion_to_static_motion_primitive(name, motion, skeleton, n_basis=7, degree=3):
    """
        Represent motion data as functional data, motion data should be narray<2d> n_frames * n_dims,
        the functional data has the shape n_basis * n_dims
    """

    motion_data = np.asarray(motion)
    n_frames, n_dims = motion_data.shape
    knots = get_cubic_b_spline_knots(n_basis, n_frames)
    x = list(range(n_frames))
    coeffs = [si.splrep(x, motion_data[:, i], k=degree, t=knots[degree + 1: -(degree + 1)])[1][:-4] for i in range(n_dims)]
    coeffs = np.asarray(coeffs).T

    data = dict()
    data["name"] = name
    data["spatial_coeffs"] = coeffs.tolist()
    data["knots"] = knots.tolist()
    data["n_canonical_frames"] = len(motion)
    data["skeleton"] = skeleton.to_json()
    return data


def train_model(name, motion_folder, output_folder, skeleton, max_training_samples=100, animated_joints=None, save_skeleton=False, use_multi_processing=True, temp_data_dir=None, pre_aligned=False):
    print("train model",name, motion_folder, use_multi_processing)
    motions = load_motion_data(motion_folder, max_count=max_training_samples, animated_joints=animated_joints)
    ref_frame = None
    for key, m in motions.items():
        if ref_frame is None:
            ref_frame = m[0]
        motions[key] = smooth_quaternion_frames(skeleton, m, ref_frame)

    keyframes_filename = motion_folder+os.sep+"keyframes.json"
    if os.path.isfile(keyframes_filename):
        keyframes = load_json_file(keyframes_filename)
        sections = define_sections_from_keyframes(motions.keys(), keyframes)
        filtered_motions = collections.OrderedDict()
        for key in motions.keys():
            if key in keyframes:
                filtered_motions[key] = motions[key]
        motions = filtered_motions
    else:
        motions, sections = define_sections_from_annotations(motion_folder, motions)

    out_filename = output_folder + os.sep + name + MM_FILE_ENDING
    if len(motions) > 1:
        config = get_standard_config()
        config["use_multi_processing"] = use_multi_processing
        config["temp_data_dir"] = temp_data_dir
        constructor = MotionModelConstructor(skeleton, config)
        align_frames = True
        if not pre_aligned or not os.path.isfile(motion_folder+ os.sep+ "temporal_data.npy"):
            constructor.set_motions(motions)
            constructor.set_dtw_sections(sections)
        else:
            constructor.set_aligned_frames(motions)
            temporal_data = np.load(motion_folder+ os.sep+ "temporal_data.npy",allow_pickle=True)
            constructor.set_timewarping(temporal_data)
            align_frames = False
        model_data = constructor.construct_model(name, version=3, save_skeleton=save_skeleton, align_frames=align_frames)
        
        with open(out_filename, 'w') as outfile:
            json.dump(model_data, outfile)

    elif len(motions) == 1:
        keys = list(motions.keys())
        model_data = convert_motion_to_static_motion_primitive(name, motions[keys[0]], skeleton)
        with open(out_filename, 'w') as outfile:
            json.dump(model_data, outfile)
    else:
        print("Error: Did not find any BVH files in the directory", motion_folder)
        model_data = dict()
        model_data["n_motions"] = len(motions)
        model_data["n_files"] = len(glob.glob(motion_folder+"*"))
        out_filename = output_folder + os.sep + "MODELING_FAILED"
        with open(out_filename, 'w') as outfile:
            json.dump(model_data, outfile)


def load_model(filename, skeleton):
    with open(filename, 'r') as infile:
        model_data = json.load(infile)
        model = MotionPrimitiveModelWrapper()
        model._initialize_from_json(skeleton.convert_to_mgrd_skeleton(), model_data)
        motion_spline = model.sample(False)
        frames = motion_spline.get_motion_vector()
        print(frames.shape)
        export_frames_to_bvh(skeleton, frames, "sample")



def main():

    parser = argparse.ArgumentParser(description='Creat motion model.')
    parser.add_argument('--name', help='name')
    parser.add_argument('--skel_filename', help='skeleton filename')
    parser.add_argument('--input_folder', help='folder containing BVH files')
    parser.add_argument('--output_folder', help='folder containing the resulting model')
    parser.add_argument('--scale', nargs='?', default=1.0, help='scale')
    parser.add_argument('-n','--n_max_samples', nargs='?', default=1000, help='Maximum number of samples per primitive')
    parser.add_argument('-s','--save_skeleton', nargs='?', default="True", help='stores skeleton in the model file')
    parser.add_argument('--single_process', action="store_true", help='deactivates multiple processes')
    parser.add_argument('--pre_aligned', action="store_true", help='deactivates time warping')
    parser.add_argument('-t', "--temp_data_dir", nargs='?',default=None, help='directory for temp data export')
    parser.add_argument( "--joint_filter", nargs='+',default=None, help='Sequence of joint names to model')
    parser.add_argument( "--user", nargs='+',default=None, help='User for server access')
    parser.add_argument( "--password", nargs='+',default=None, help='Password for server access')
    parser.add_argument( "--token", nargs='+',default=None, help='Encrypted user password for server access')
    args = parser.parse_args()
    if args.name and args.skel_filename and args.input_folder and args.output_folder:
        joint_filter = args.joint_filter
        skeleton = load_skeleton(args.skel_filename, joint_filter, args.scale)
        animated_joints = skeleton.animated_joints
        if not os.path.isdir(args.output_folder):
            os.makedirs(args.output_folder)
        train_model(args.name, args.input_folder, args.output_folder, 
                    skeleton, args.n_max_samples, 
                    animated_joints, save_skeleton=args.save_skeleton, 
                    use_multi_processing=not args.single_process, temp_data_dir=args.temp_data_dir, pre_aligned=args.pre_aligned)
    else:
        print("Not enough arguments")

if __name__ == "__main__":
    main()
 


