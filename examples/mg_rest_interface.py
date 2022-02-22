# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:05:40 2015
REST interface for the MorphableGraphs algorithm based on the Tornado library.
Implemented according to the following tutorial:
http://www.drdobbs.com/open-source/building-restful-apis-with-tornado/240160382?pgno=1
@author: erhe01
"""
import os
# change working directory to the script file directory
file_dir_name, file_name = os.path.split(os.path.abspath(__file__))
os.chdir(file_dir_name)
import urllib.request, urllib.error, urllib.parse
import numpy as np
import socket
import tornado.escape
import tornado.ioloop
import tornado.web
import json
import time
from datetime import datetime
import argparse
from anim_utils.utilities.io_helper_functions import load_json_file, write_to_json_file
from morphablegraphs import MotionGenerator, DEFAULT_ALGORITHM_CONFIG
from morphablegraphs.motion_model import MotionStateGraphLoader
from anim_utils.utilities.log import write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR, set_log_mode
from anim_utils.retargeting import retarget_from_src_to_target, GAME_ENGINE_TO_ROCKETBOX_MAP, ROCKETBOX_ROOT_OFFSET
from anim_utils.animation_data import SkeletonBuilder, MotionVector, BVHReader, BVHWriter
from morphablegraphs.motion_generator import AnnotatedMotionVector
from jsonpath_wrapper import update_data_using_jsonpath

SERVICE_CONFIG_FILE = "config" + os.sep + "service.config"
TARGET_SKELETON = "game_engine_target.bvh"

ROCKETBOX_TO_GAME_ENGINE_MAP = dict()
ROCKETBOX_TO_GAME_ENGINE_MAP["Hips"] = "pelvis"
ROCKETBOX_TO_GAME_ENGINE_MAP["Spine"] = "spine_01"
ROCKETBOX_TO_GAME_ENGINE_MAP["Spine_1"] = "spine_02"
ROCKETBOX_TO_GAME_ENGINE_MAP["Neck"] = "neck_01"
ROCKETBOX_TO_GAME_ENGINE_MAP["Head"] = "head"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftShoulder"] = "clavicle_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightShoulder"] = "clavicle_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftArm"] = "upperarm_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightArm"] = "upperarm_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftForeArm"] = "lowerarm_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightForeArm"] = "lowerarm_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftHand"] = "hand_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightHand"] = "hand_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftUpLeg"] = "thigh_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightUpLeg"] = "thigh_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftLeg"] = "calf_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightLeg"] = "calf_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftFoot"] = "foot_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightFoot"] = "foot_r"


def get_bvh_writer(skeleton, quat_frames, is_quaternion=True):
    """
    Returns
    -------
    * bvh_writer: BVHWriter
        An instance of the BVHWriter class filled with Euler frames.
    """
    if len(quat_frames) > 0 and len(quat_frames[0]) < skeleton.reference_frame_length:
        quat_frames = skeleton.add_fixed_joint_parameters_to_motion(quat_frames)
    bvh_writer = BVHWriter(None, skeleton, quat_frames, skeleton.frame_time, is_quaternion)
    return bvh_writer


def retarget_motion_vector(src_motion_vector, target_skeleton, scale_factor=1):
    write_message_to_log("Start retargeting...", LOG_MODE_INFO)
    target_frames = retarget_from_src_to_target(src_motion_vector.skeleton,
                                                target_skeleton,
                                                src_motion_vector.frames,
                                                GAME_ENGINE_TO_ROCKETBOX_MAP, None, scale_factor)
    target_skeleton.frame_time = src_motion_vector.skeleton.frame_time
    target_motion_vector = AnnotatedMotionVector(target_skeleton)
    target_motion_vector.frame_time = src_motion_vector.frame_time
    target_motion_vector.frames = target_frames
    return target_motion_vector


def load_target_skeleton(file_path, scale_factor=1.0):
    skeleton = None
    target_bvh = BVHReader(file_path)
    animated_joints = list(target_bvh.get_animated_joints())
    skeleton = SkeletonBuilder().load_from_bvh(target_bvh, animated_joints, add_tool_joints=False)
    for node in list(skeleton.nodes.values()):
        node.offset[0] *= scale_factor
        node.offset[1] *= scale_factor
        node.offset[2] *= scale_factor
    return skeleton


class GenerateMotionHandler(tornado.web.RequestHandler):
    """Handles HTTP POST Requests to a registered server url.
        Starts the morphable graphs algorithm if an input file
        is detected in the request body.
    """

    def __init__(self, application, request, **kwargs):
        tornado.web.RequestHandler.__init__(
            self, application, request, **kwargs)
        self.application = application

    def get(self):
        error_string = "GET request not implemented. Use POST instead."
        write_message_to_log(error_string, LOG_MODE_ERROR)
        self.write(error_string)

    def post(self):
        try:
            mg_input = json.loads(self.request.body.decode("utf-8"))
        except Exception as e:
            error_string = "Error: Could not decode request body as JSON." + str(e.args)
            write_message_to_log(error_string, LOG_MODE_ERROR)
            self.write(error_string)
            return
        motion_vector = self.application.generate_motion(mg_input, False)

        if motion_vector is not None:
            self._handle_result(mg_input, motion_vector)
        else:
            error_string = "Error: Could not generate motion."
            self.write(error_string)

    def _handle_result(self, mg_input, motion_vector):
        """Sends the result back as an answer to a post request.
        """
        if motion_vector.has_frames():
            if mg_input["outputMode"] == "Unity":

                target_skeleton = self.application.get_target_skeleton()

                if target_skeleton is not None:
                    motion_vector = retarget_motion_vector(motion_vector, target_skeleton)

                result_object = motion_vector.to_unity_format()

            else:
                result_object = self.convert_to_interact_format(motion_vector)

            self.write(json.dumps(result_object))

        else:
            error_string = "Error: Failed to generate motion data."
            write_message_to_log(error_string, LOG_MODE_ERROR)
            self.write(error_string)

    def convert_to_interact_format(self, motion_vector):
        write_message_to_log("Converting the motion into the BVH format...", LOG_MODE_DEBUG)
        start = time.time()
        bvh_writer = get_bvh_writer(motion_vector.skeleton, motion_vector.frames)
        bvh_string = bvh_writer.generate_bvh_string()
        result_object = {
            "bvh": bvh_string,
            "annotation": motion_vector.keyframe_event_list.frame_annotation,
            "event_list": motion_vector.keyframe_event_list.keyframe_events_dict}
        message = "Finished converting the motion to a BVH string in " + str(time.time() - start) + " seconds"
        write_message_to_log(message, LOG_MODE_INFO)
        if self.application.export_motion_to_file:
            self._export_motion_to_file(bvh_string, motion_vector)
        return result_object

    def _export_motion_to_file(self, bvh_string, motion_vector):
        bvh_filename = self.application.service_config["output_dir"] + os.sep + self.application.service_config["output_filename"]
        if self.application.add_timestamp_to_filename:
            bvh_filename += "_"+str(datetime.now().strftime("%d%m%y_%H%M%S"))
        write_message_to_log("export motion to file " + bvh_filename, LOG_MODE_DEBUG)
        with open(bvh_filename+".bvh", "wb") as out_file:
            out_file.write(bvh_string)
        if motion_vector.mg_input is not None:
            write_to_json_file(bvh_filename+ "_input.json", motion_vector.mg_input.mg_input_file)
        if motion_vector.keyframe_event_list is not None:
            motion_vector.keyframe_event_list.export_to_file(bvh_filename)


class GetSkeletonHandler(tornado.web.RequestHandler):
    """Handles HTTP POST Requests to a registered server url.
        Starts the morphable graphs algorithm if an input file
        is detected in the request body.
    """

    def __init__(self, application, request, **kwargs):
        tornado.web.RequestHandler.__init__(
            self, application, request, **kwargs)
        self.application = application

    def get(self):
        error_string = "GET request not implemented. Use POST instead."
        write_message_to_log(error_string, LOG_MODE_ERROR)
        self.write(error_string)

    def post(self):
        target_skeleton = self.application.get_target_skeleton()
        if target_skeleton is None:
            target_skeleton = self.application.get_skeleton()

        result_object = target_skeleton.to_unity_format(joint_name_map=ROCKETBOX_TO_GAME_ENGINE_MAP)
        self.write(json.dumps(result_object))


class SetConfigurationHandler(tornado.web.RequestHandler):
    """Handles HTTP POST Requests to a registered server url.
        Sets the configuration of the morphable graphs algorithm
        if an input file is detected in the request body.
    """

    def __init__(self, application, request, **kwargs):
        tornado.web.RequestHandler.__init__(
            self, application, request, **kwargs)
        self.application = application

    def get(self):
        error_string = "GET request is not implemented. Use POST instead."
        write_message_to_log(error_string, LOG_MODE_ERROR)
        self.write(error_string)

    def post(self):
        #  try to decode message body
        try:
            algorithm_config = json.loads(self.request.body.decode("utf-8"))
        except:
            error_string = "Error: Could not decode request body as JSON."
            self.write(error_string)
            return
        if "use_constraints" in list(algorithm_config.keys()):
            self.application.set_algorithm_config(algorithm_config)
            print("Set algorithm config to", algorithm_config)
        else:
            error_string = "Error: Did not find expected keys in the input data.", algorithm_config
            self.write(error_string)


class MGRestApplication(tornado.web.Application):
    """ Extends the Application class with a MotionGenerator instance and algorithm options.
        This allows access to the data in the MGInputHandler class
    """
    def __init__(self, service_config, algorithm_config, handlers=None, default_host="", transforms=None, **settings):
        tornado.web.Application.__init__(self, handlers, default_host, transforms)

        self.algorithm_config = algorithm_config
        self.service_config = service_config
        self.export_motion_to_file = False
        self.activate_joint_map = False
        self.apply_coordinate_transform = True
        self.add_timestamp_to_filename = True

        if "export_motion_to_file" in list(self.service_config.keys()):
            self.export_motion_to_file = service_config["export_motion_to_file"]
        if "add_time_stamp" in list(self.service_config.keys()):
            self.add_timestamp_to_filename = service_config["add_time_stamp"]
        if "activate_joint_map" in list(self.service_config.keys()):
            self.activate_joint_map = service_config["activate_joint_map"]
        if "activate_coordinate_transform" in list(self.service_config.keys()):
            self.activate_coordinate_transform = service_config["activate_coordinate_transform"]

        if self.export_motion_to_file:
            write_message_to_log("Motions are written as BVH file to the directory" + self.service_config["output_dir"], LOG_MODE_INFO)
        else:
            write_message_to_log("Motions are returned as answer to the HTTP POST request", LOG_MODE_INFO)

        if not service_config["activate_collision_avoidance"] or not self._test_ca_interface(service_config):
            service_config["collision_avoidance_service_url"] = None

        start = time.clock()

        graph_loader = MotionStateGraphLoader()
        graph_loader.set_data_source(service_config["model_data"], algorithm_config["use_transition_model"])
        motion_state_graph = graph_loader.build()
        self.motion_generator = MotionGenerator(motion_state_graph, self.service_config, self.algorithm_config)
        self.target_skeleton = None
        message = "Finished construction from file in " + str(time.clock() - start) + " seconds"
        write_message_to_log(message, LOG_MODE_INFO)

    def generate_motion(self, mg_input, complete_motion_vector=True):
        return self.motion_generator.generate_motion(mg_input, activate_joint_map=self.activate_joint_map,
                                                     activate_coordinate_transform=self.activate_coordinate_transform,
                                                     complete_motion_vector=complete_motion_vector)

    def is_initiated(self):
        return self.motion_generator._motion_state_graph.skeleton is not None \
                and len(self.motion_generator._motion_state_graph.nodes) > 0

    def get_skeleton(self):
        return self.motion_generator.get_skeleton()

    def get_target_skeleton(self):
        return self.target_skeleton

    def set_target_skeleton(self, skeleton_file,scale_factor=1.0):
        self.target_skeleton = load_target_skeleton(skeleton_file, scale_factor)

    def set_algorithm_config(self, algorithm_config):
        self.motion_generator.set_algorithm_config(algorithm_config)

    def _test_ca_interface(self, service_config):
        if "collision_avoidance_service_url" in list(service_config.keys()) and "collision_avoidance_service_port" in list(service_config.keys()):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            address = (service_config["collision_avoidance_service_url"], service_config["collision_avoidance_service_port"])
            try:
                write_message_to_log("Try to connect to CA interface using address " + str(address), LOG_MODE_DEBUG)
                s.connect(address)
                s.close()
                write_message_to_log("Collision avoidance will be activated", LOG_MODE_INFO)
                return True
            except Exception as e:
                write_message_to_log("Warning: Could not create connection to collision avoidance interface" + str(e.args), LOG_MODE_ERROR)
        write_message_to_log("Collision avoidance will be disabled", LOG_MODE_INFO)
        service_config["collision_avoidance_service_url"] = None
        return False




class MGRESTInterface(object):
    """Implements a REST interface for MorphableGraphs.

    Parameters:
    ----------
    * service_config_file : String
        Path to service configuration
    * json_path_expressions : List
        List of JSONPath expressions to change the default values in the
        configuration file. Example: "$.port=8889"

    How to use from client side:
    ----------------------------
    send POST request to 'http://localhost:port/generate_motion' with JSON
    formatted input as body.
    Example with urllib2 when output_mode is answer_request:
    request = urllib2.Request(mg_server_url, mg_input_data)
    handler = urllib2.urlopen(request)
    bvh_string, annotations, actions = json.loads(handler.read())

    configuration can be changed by sending the data to the URL
    'http://localhost:port/config_morphablegraphs'
    """

    def __init__(self, service_config_file, json_path_expressions):

        #  Load configuration files
        service_config = load_json_file(service_config_file)
        update_data_using_jsonpath(service_config, json_path_expressions)
        algorithm_config_file = "config" + os.sep + service_config["algorithm_settings"] + "_algorithm.config"
        if os.path.isfile(algorithm_config_file):
            write_message_to_log("Load algorithm configuration from " + algorithm_config_file, LOG_MODE_INFO)
            algorithm_config = load_json_file(algorithm_config_file)
        else:
            write_message_to_log("Did not find algorithm configuration file " + algorithm_config_file, LOG_MODE_INFO)
            algorithm_config = DEFAULT_ALGORITHM_CONFIG

        #  Construct morphable graph from files
        self.application = MGRestApplication(service_config, algorithm_config,
                                             [(r"/run_morphablegraphs", GenerateMotionHandler),#legacy
                                              (r"/config_morphablegraphs", SetConfigurationHandler),
                                              (r"/generate_motion", GenerateMotionHandler),
                                               (r"/get_skeleton", GetSkeletonHandler)
                                              ])

        self.port = service_config["port"]

    def set_target_skeleton(self, skeleton_file, scale_factor=1.0):
        self.application.set_target_skeleton(skeleton_file, scale_factor)

    def start(self):
        """ Start the web server loop
        """
        if self.application.is_initiated():
            write_message_to_log("Start listening to port " + str(self.port), LOG_MODE_INFO)
            self.application.listen(self.port)
            tornado.ioloop.IOLoop.instance().start()
        else:
            write_message_to_log("Error: Could not initiate MG REST service", LOG_MODE_ERROR)

    def stop(self):
        tornado.ioloop.IOLoop.instance().stop()



def main():

    parser = argparse.ArgumentParser(description="Start the MorphableGraphs REST-interface")
    parser.add_argument("-set", nargs='+', default=[], help="JSONPath expression, e.g. -set $.model_data=path/to/data")
    parser.add_argument("-config_file", nargs='?', default=SERVICE_CONFIG_FILE, help="Path to default config file")
    parser.add_argument("-target_skeleton", nargs='?', default=None, help="Path to target skeleton file")
    parser.add_argument("-skeleton_scale", nargs='?', default=1.0, help="Scale applied to the target skeleton offsets")
    args = parser.parse_args()
    if os.path.isfile(args.config_file):
        mg_service = MGRESTInterface(args.config_file, args.set)
        if args.target_skeleton is not None:
            mg_service.set_target_skeleton(args.target_skeleton, scale_factor=args.skeleton_scale)
        mg_service.start()
    else:
        write_message_to_log("Error: could not open service or algorithm configuration file", LOG_MODE_ERROR)


if __name__ == "__main__":
    main()

