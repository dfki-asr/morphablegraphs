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
Created on Mon Jun 20 2017

@author: Erik Herrmann
"""
import numpy as np
import json
import requests
import collections
import scipy.interpolate as si
import bson
import warnings
from anim_utils.animation_data import BVHReader, BVHWriter, MotionVector, SkeletonBuilder
from anim_utils.utilities.db_interface import get_skeleton_from_remote_db, get_skeleton_model_from_remote_db, get_motion_list_from_remote_db,\
                                             get_bvh_str_by_id_from_remote_db, get_annotation_by_id_from_remote_db, \
                                            get_time_function_by_id_from_remote_db, get_motion_by_id_from_remote_db, upload_motion_to_db, delete_motion_by_id_from_remote_db
from ..construction.motion_model_constructor import MotionModelConstructor
from ..construction.utils import get_cubic_b_spline_knots
from ..motion_model.motion_primitive_wrapper import MotionPrimitiveModelWrapper
from ..construction.cluster_tree_builder import FeatureClusterTree
from . import convert_to_mgrd_skeleton

def call_rest_interface(url, method, data):
    method_url = url+method
    r = requests.post(method_url, data=json.dumps(data))
    return r.text

def call_bson_rest_interface(url, method, data):
    method_url = url+method
    r = requests.post(method_url, data=json.dumps(data))
    return r.content

def get_model_list_from_remote_db(url, collection_id, skeleton="", session=None):
    data = {"collection_id": collection_id, "skeleton": skeleton}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "get_model_list", data)
    try:
        result_data = json.loads(result_str)
    except:
        result_data = None
    return result_data


def delete_model_by_id_from_remote_db(url, model_id, session=None):
    data = {"model_id": model_id}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "delete_model", data)
    return result_str

def upload_motion_model_to_remote_db(url, name, collection, skeleton_name, model_data, config, session=None):
    data = {"name":name, "collection": collection, "skeleton_name": skeleton_name,
            "data": model_data, "config": config}
    if session is not None:
        data.update(session)
    call_rest_interface(url, "upload_motion_model", data)


def download_motion_model_from_remote_db(url, model_id, session=None):
    data = {"model_id": model_id}
    if session is not None:
        data.update(session)
    result_str= call_rest_interface(url, "download_motion_model", data)
    return result_str

def upload_cluster_tree_to_remote_db(url, model_id, cluster_tree_data, session=None):
    data = {"model_id": model_id, "cluster_tree_data": cluster_tree_data}
    if session is not None:
        data.update(session)
    result_str= call_rest_interface(url, "upload_cluster_tree", data)
    return

def download_cluster_tree_from_remote_db(url, model_id, session=None):
    data = {"model_id": model_id}
    if session is not None:
        data.update(session)
    result_str= call_rest_interface(url, "download_cluster_tree", data)
    return result_str

def create_cluster_tree_from_model(model_data, n_samples, n_subdivisions_per_level = 4, session=None):
    skeleton = SkeletonBuilder().load_from_json_data(model_data["skeleton"])
    CLUSTERING_METHOD_KMEANS = 0
    mp = MotionPrimitiveModelWrapper()
    mp.cluster_tree = None
    mp._initialize_from_json(convert_to_mgrd_skeleton(skeleton), model_data)
    data = mp.sample_low_dimensional_vectors(n_samples)
    n_spatial = mp.get_n_spatial_components()
    features = data[:, :n_spatial]
    options = {"n_subdivisions": n_subdivisions_per_level,
                "clustering_method": CLUSTERING_METHOD_KMEANS,
                "use_feature_mean": False}
    return FeatureClusterTree(features, data, None, options, [])

def load_cluster_tree_from_json(tree_data):
    return FeatureClusterTree.load_from_json(tree_data)

def load_skeleton_from_db(db_url, skeleton_name, session=None):
    skeleton_data = get_skeleton_from_remote_db(db_url, skeleton_name, session)
    if skeleton_data is not None:
        skeleton = SkeletonBuilder().load_from_custom_unity_format(skeleton_data)
        skeleton_model = get_skeleton_model_from_remote_db(db_url, skeleton_name, session)
        skeleton.skeleton_model = skeleton_model
        return skeleton


def get_bvh_data(db_url, collection, skeleton="", is_aligned=0, session=None):
    motion_list = get_motion_list_from_remote_db(db_url, collection, skeleton, is_aligned)
    return get_bvh_data_from_motion_list(db_url, motion_list)


def get_motion_vectors(db_url, collection, skeleton="", is_aligned=False, session=None):
    motion_list = get_motion_list_from_remote_db(db_url, collection, skeleton, is_aligned)
    return get_motion_vectors_from_motion_list(db_url, motion_list, is_aligned)

def get_motion_vectors_from_motion_list(db_url, motion_list, is_processed=False, session=None):
    count = 1
    n_motions = len(motion_list)
    bvh_data = dict()
    for node_id, name in motion_list:
        print("download motion", str(count)+"/"+str(n_motions), node_id, name, is_processed)
        motion = get_motion_by_id_from_remote_db(db_url, node_id, is_processed)
        annotation_str = get_annotation_by_id_from_remote_db(db_url, node_id, is_processed)
        section_annotation = None
        if annotation_str != "": 
            data = json.loads(annotation_str)
            if "sections" in data:
                section_annotation = data["sections"]
        
        time_function_str = get_time_function_by_id_from_remote_db(db_url, node_id)
        time_function = None
        if time_function_str != "": 
            time_function = json.loads(time_function_str)
            #print("found str", node_id, name, type(time_function),  time_function_str)
            print(time_function)
            if isinstance(time_function, str) and time_function != "":
                time_function = json.loads(time_function)
                print("WARINING: time function was deserialized to string instead of list", time_function_str, type(time_function))
                

        bvh_data[node_id] = dict()
        bvh_data[node_id]["data"] = motion
        bvh_data[node_id]["section_annotation"] = section_annotation
        bvh_data[node_id]["time_function"] = time_function
        bvh_data[node_id]["name"] = name
        count+=1
    return bvh_data

def get_bvh_data_from_motion_list(db_url, motion_list, is_processed=False, session=None):
    count = 1
    n_motions = len(motion_list)
    bvh_data = dict()
    for node_id, name in motion_list:
        print("download motion", str(count)+"/"+str(n_motions), node_id, name)
        bvh_str = get_bvh_str_by_id_from_remote_db(db_url, node_id, is_processed, )
        annotation_str = get_annotation_by_id_from_remote_db(db_url, node_id, is_processed, session)
        section_annotation = None
        if annotation_str != "": 
            data = json.loads(annotation_str)
            section_annotation = data["sections"]
        
        time_function_str = get_time_function_by_id_from_remote_db(db_url, node_id)
        time_function = None
        if time_function_str != "": 
            time_function = json.loads(time_function_str)
            #print("found str", node_id, name, type(time_function),  time_function_str)
            print(time_function)
            if isinstance(time_function, str) and time_function != "":
                time_function = json.loads(time_function)
                print("WARINING: time function was deserialized to string instead of list", type(time_function))
                

        bvh_data[node_id] = dict()
        bvh_data[node_id]["bvh_str"] = bvh_str
        bvh_data[node_id]["section_annotation"] = section_annotation
        bvh_data[node_id]["time_function"] = time_function
        bvh_data[node_id]["name"] = name
        count+=1
    return bvh_data

def get_bvh_string(skeleton, frames):
    print("generate bvh string", len(skeleton.animated_joints))
    frames = np.array(frames)
    frames = skeleton.add_fixed_joint_parameters_to_motion(frames)
    frame_time = skeleton.frame_time
    bvh_writer = BVHWriter(None, skeleton, frames, frame_time, True)
    return bvh_writer.generate_bvh_string()

def get_motion_vector(skeleton, frames):
    print("generate motion vector", len(skeleton.animated_joints))
    frames = np.array(frames)
    #frames = skeleton.add_fixed_joint_parameters_to_motion(frames)
    frame_time = skeleton.frame_time
    mv = MotionVector()
    mv.frames = frames
    mv.n_frames = len(frames)
    mv.skeleton = skeleton
    return mv

def create_sections_from_keyframes(keyframes):
    sorted_keyframes = collections.OrderedDict(sorted(keyframes.items(), key=lambda t: t[1]))
    start = 0
    #end = n_canonical_frames
    semantic_annotation = collections.OrderedDict()
    for k, v in sorted_keyframes.items():
        print("set key",start, v)
        semantic_annotation[start] = {"start_idx":start,  "end_idx":v}
        start = v
    #semantic_annotation[start] = {"start_idx":start,  "end_idx":end}
    return list(semantic_annotation.values())


def align_motion_data(skeleton, motion_data, config=None, mean_key=None):
    motions, sections, temporal_data = generate_training_data(motion_data)
    if config is None:
        config = get_standard_config()
    constructor = MotionModelConstructor(skeleton, config)
    if len(sections) == len(motions):
        constructor.set_motions(motions)
        constructor.set_dtw_sections(sections)
    elif len(sections) > 0: # filter motions by sections
        _motions = collections.OrderedDict()
        for key in sections:
            _motions[key] = motions[key]
        constructor.set_motions(_motions)
        constructor.set_dtw_sections(sections) 
    else: # ignore sections
        constructor.set_motions(motions)
        constructor.set_dtw_sections(None) 

    constructor._align_frames(mean_key)

    # generate from keyframes
    if len(constructor._keyframes) > 0:
        #first_key = list(constructor._aligned_frames.keys())[0]
        #n_canonical_frames = len(constructor._aligned_frames[first_key])
        key = list(constructor._aligned_frames.keys())[0]
        n_frames = len(constructor._aligned_frames[key])
        for key in constructor._keyframes:
            if constructor._keyframes[key] == -1:
                constructor._keyframes[key] = n_frames-1
        meta_info = dict()
        meta_info["sections"] = create_sections_from_keyframes(constructor._keyframes)
        meta_info_str = json.dumps(meta_info)
        print("Found", len(constructor._keyframes), "keyframes")
    else:
        print("No keyframes found")
        meta_info_str = ""

    aligned_data = dict()
    for key, frames in constructor._aligned_frames.items():
        aligned_data[key] = dict()

        aligned_data[key]["frames"] = frames # needs to be converted to bvh str before upload
        aligned_data[key]["meta_info"] = meta_info_str
        aligned_data[key]["time_function"] = constructor._temporal_data[key]

    return aligned_data


def get_standard_config():
    config = dict()
    config["n_basis_functions_spatial"] = 16
    config["n_spatial_basis_factor"] = 1.0/5.0
    config["fraction"] = 0.95
    config["n_basis_functions_temporal"] = 8
    config["npc_temporal"] = 3
    config["n_components"] = None
    config["precision_temporal"] = 0.99
    return config


def get_bvh_from_str(bvh_str):
    bvh_reader = BVHReader("")
    lines = bvh_str.split("\n")
    # print(len(lines))
    lines = [l for l in lines if len(l) > 0]
    bvh_reader.process_lines(lines)
    return bvh_reader


def generate_training_data_from_bvh(bvh_data, animated_joints=None):
    motions = collections.OrderedDict()
    sections = collections.OrderedDict()
    temporal_data = collections.OrderedDict()
    skeleton = None
    for name, value in bvh_data.items():
        bvh_str = value["bvh_str"]
        print("process", name)
        mv = create_motion_vector_from_bvh(bvh_str, animated_joints)
        if skeleton is None:
            skeleton = mv.skeleton
        motions[name] = mv.frames
        if value["section_annotation"] is not None:
            sections[name] = value["section_annotation"]#create_sections_from_annotation(annotation)
        if value["time_function"] is not None:
            temporal_data[name] =  value["time_function"]
    return skeleton, motions, sections, temporal_data


def generate_training_data(motion_data, animated_joints=None):
    motions = collections.OrderedDict()
    sections = collections.OrderedDict()
    temporal_data = collections.OrderedDict()
    for name, value in motion_data.items():
        data = value["data"]
        motion_vector = MotionVector()
        motion_vector.from_custom_db_format(data)
        motions[name] = motion_vector.frames
        if value["section_annotation"] is not None:#
            v_type = type(value["section_annotation"])
            if v_type == list:
                sections[name] = value["section_annotation"]
            elif v_type == dict:
                sections[name] = list()#create_sections_from_annotation(annotation)
                print(value["section_annotation"])
                for section_key in value["section_annotation"]:
                    n_sections  = len(value["section_annotation"][section_key])
                    if n_sections == 1: # take only the first segment in the list
                        sections[name].append(value["section_annotation"][section_key][0])
                    else:
                        warnings.warn("number of annotations "+str(section_key)+" "+str(n_sections))
            else:
                warnings.warn("type unknown", name, v_type)
        if value["time_function"] is not None:
            temporal_data[name] = value["time_function"]
    return motions, sections, temporal_data

def create_keyframes_from_sections(sections):
    keyframes = dict()
    for i, s in enumerate(sections):
        keyframes["contact"+str(i)] = s["end_idx"]
    return keyframes


def create_motion_primitive_model(name, skeleton, motion_data, config=None, animated_joints=None, save_skeleton=True, align_frames=True):
    print("create model", animated_joints)
    motions, sections, temporal_data  = generate_training_data(motion_data, animated_joints)
    if config is None:
        config = get_standard_config()

    constructor = MotionModelConstructor(skeleton, config)
    constructor.set_motions(motions)

    if align_frames:
        if len(sections) == len(motions):
            constructor.set_dtw_sections(sections)
        else:
            constructor.set_dtw_sections(None) # ignore
    else:
        keyframes = dict()
        if len(sections) > 0:
            first_key = list(sections.keys())[0]
            keyframes = create_keyframes_from_sections(sections[first_key])
        constructor.set_aligned_frames(motions, keyframes)
        constructor.set_timewarping(temporal_data)
    print("model", len(motions), align_frames, len(constructor._aligned_frames))
    model_data = constructor.construct_model(name, version=3, save_skeleton=save_skeleton, align_frames=align_frames)
    return model_data


def convert_motion_to_static_motion_primitive(name, frames, skeleton, n_basis=7, degree=3):
    """
        Represent motion data as functional data, motion data should be narray<2d> n_frames * n_dims,
        the functional data has the shape n_basis * n_dims
    """    
    frames = np.asarray(frames)
    n_frames, n_dims = frames.shape
    knots = get_cubic_b_spline_knots(n_basis, n_frames)
    x = list(range(n_frames))
    coeffs = [si.splrep(x, frames[:, i], k=degree,
                        t=knots[degree + 1: -(degree + 1)])[1][:-4] for i in range(n_dims)]
    coeffs = np.asarray(coeffs).T

    data = dict()
    data["name"] = name
    data["spatial_coeffs"] = coeffs.tolist()
    data["knots"] = knots.tolist()
    data["n_canonical_frames"] = len(frames)
    data["skeleton"] = skeleton.to_json()
    return data

def create_motion_vector_from_bvh(bvh_str, animated_joints=None):
    bvh_reader = get_bvh_from_str(bvh_str)
    print("loaded motion", bvh_reader.frames.shape)
    if animated_joints is None:
        animated_joints = [key for key in list(bvh_reader.node_names.keys()) if not key.endswith("EndSite")]
    skeleton = SkeletonBuilder().load_from_bvh(bvh_reader, animated_joints)

    motion_vector = MotionVector()
    motion_vector.from_bvh_reader(bvh_reader, False, animated_joints)
    motion_vector.skeleton = skeleton
    return motion_vector


def create_motion_vector_from_json(motion_data):
    motion_vector = MotionVector()
    motion_vector.from_custom_db_format(motion_data)
    return motion_vector


def retarget_motion_in_db(db_url, retargeting, motion_id, motion_name, collection, skeleton_model_name, is_aligned=False, session=None):
    motion_data = get_motion_by_id_from_remote_db(db_url, motion_id, is_processed=is_aligned)
    if motion_data is None:
        print("Error: motion data is empty")
        return
    
    meta_info_str = get_annotation_by_id_from_remote_db(db_url, motion_id, is_processed=is_aligned)
    motion_vector = MotionVector()
    motion_vector.from_custom_db_format(motion_data)
    motion_vector.skeleton = retargeting.src_skeleton
    new_frames = retargeting.run(motion_vector.frames, frame_range=None)
    target_motion = MotionVector()
    target_motion.frames = new_frames
    target_motion.skeleton = retargeting.target_skeleton
    target_motion.frame_time = motion_vector.frame_time
    target_motion.n_frames = len(new_frames)
    m_data = target_motion.to_db_format()
    upload_motion_to_db(db_url, motion_name, m_data, collection, skeleton_model_name, meta_info_str, is_processed=is_aligned, session=session)


def align_motions_in_db(db_url, skeleton_name, c_id, session=None):
    motion_data = get_motion_vectors(db_url, c_id, skeleton_name, is_aligned=0)
    # delete old data
    old_aligned_motions = get_motion_list_from_remote_db(db_url, c_id, skeleton_name, is_processed=True)
    for motion in old_aligned_motions:
        delete_motion_by_id_from_remote_db(db_url, motion[0], is_processed=True, session=session)
    skeleton = load_skeleton_from_db(db_url, skeleton_name)

    n_motions = len(motion_data)
    if n_motions > 1:
        print("start alignment", n_motions)
        aligned_data = align_motion_data(skeleton, motion_data)
        print("finished alignment")
        for key, data in aligned_data.items():
            frames = data["frames"]
            name = motion_data[key]["name"] + "_aligned"
            mv = get_motion_vector(skeleton, frames)
            m_data = mv.to_db_format()
            try:
                meta_data = json.loads(data["meta_info"])
            except:
                meta_data = dict()
            meta_data["time_function"] = data["time_function"]
            meta_data_str = json.dumps(meta_data)
            upload_motion_to_db(db_url, name, m_data, c_id, skeleton_name, meta_data_str, is_processed=True, session=session)
        print("uploaded aligned data")
    elif n_motions == 1:
        
        first_key = list(motion_data.keys())[0]
        name = motion_data[first_key]["name"] + "_aligned"
        mdata = motion_data[first_key]["data"]
        meta_data_str = get_annotation_by_id_from_remote_db(db_url, first_key)
        try:
            meta_data = json.loads(meta_data_str)
        except:
            meta_data = dict()
        print("process",name)
        motion = create_motion_vector_from_json(mdata)
        time_function = list(range(motion.n_frames))
        meta_data["time_function"] = time_function
        meta_data_str = json.dumps(meta_data)
        upload_motion_to_db(db_url, name, mdata, c_id, skeleton_name, meta_data_str, is_processed=True, session=session)
        print("Need more than 1 motion, found", n_motions)
    else:
        print("No motions found")


def create_motion_model_in_db(db_url, skeleton_name, c_id, c_name, spline_basis_factor, animated_joints=None, session=None):
    motion_data = get_motion_vectors(db_url, c_id, skeleton_name, is_aligned=1)
    skeleton = load_skeleton_from_db(db_url, skeleton_name)
    n_motions = len(motion_data)
    if n_motions > 1:
        print("start modeling", n_motions, spline_basis_factor)
        config = get_standard_config()
        config["n_spatial_basis_factor"] = spline_basis_factor
        if animated_joints is None:
            animated_joints = skeleton.animated_joints
        model_data = create_motion_primitive_model(c_name, skeleton, motion_data, config, animated_joints, save_skeleton=True, align_frames=False)
        
        print("finished modelling")
        name = c_name+"_"+skeleton_name+"_"+str(n_motions)
        upload_motion_model_to_remote_db(db_url, name, c_id, skeleton_name, model_data, config, session)
        print("uploaded model")
    elif n_motions == 1:
        print("Create static motion primitive fromn 1 motion")
        first_key = list(motion_data.keys())[0]
        motion_vector = MotionVector()
        motion_vector.from_custom_db_format(motion_data[first_key]["data"])

        config = get_standard_config()
        n_basis = int(config["n_spatial_basis_factor"]*motion_vector.n_frames)
        name = c_name+"_"+skeleton_name+"_"+str(n_motions)
        model_data = convert_motion_to_static_motion_primitive(name, motion_vector.frames, skeleton, n_basis=n_basis, degree=3)
        upload_motion_model_to_remote_db(db_url, name, c_id, skeleton_name, model_data, config, session)
        print("uploaded model")
    else:
        print("No motion, found")


def get_graph_list_from_db(url, skeleton, session=None):
    data = {"skeleton":skeleton}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "get_graph_list", data)
    try:
        result_data = json.loads(result_str)
    except:
        result_data = None
    return result_data


def create_new_graph_in_db(url, name, skeleton, graph_data, session=None):
    data = {"name": name, "skeleton": skeleton, "data": graph_data}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "upload_graph", data)


def replace_graph_in_remote_db(url, graph_id, name, skeleton, graph_data, session=None):
    data = {"id":graph_id,"name": name, "skeleton": skeleton, "data": graph_data}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "replace_graph", data)


def delete_graph_from_remote_db(url, graph_id, session=None):
    data = {"id": graph_id}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "remove_graph", data)


def download_graph_from_remote_db(url, graph_id, session=None):
    data = {"id": graph_id}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "download_graph", data)
    try:
        result_data = json.loads(result_str)
    except:
        result_data = None
    return result_data
