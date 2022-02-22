"""run tornado with asynchio loop to use standard coroutines that give tasks to the ProcessPoolExecutor
   https://gist.github.com/arvidfm/11067131
   https://stackoverflow.com/questions/15375336/how-to-best-perform-multiprocessing-within-requests-with-the-python-tornado-serv
"""
import json
import os
import tornado.web
import time
from tornado import gen
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import argparse
import asyncio
import tornado.platform.asyncio
from anim_utils.utilities.io_helper_functions import load_json_file, write_to_json_file
from morphablegraphs import MotionGenerator, DEFAULT_ALGORITHM_CONFIG
from morphablegraphs.motion_model import MotionStateGraphLoader
from anim_utils.utilities.log import write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR, set_log_mode
from anim_utils.animation_data.skeleton_models import SKELETON_MODELS
from anim_utils.animation_data import BVHReader, SkeletonBuilder
from anim_utils.retargeting import retarget_from_src_to_target
from morphablegraphs.utilities.height_map_interface import HeightMapInterface


def eval_func( n):
    count = 0
    for i in range(n.count + 1):
        count += i
    return str(count)


class Context(object):
    def __init__(self, service_config, algorithm_config, target_skeleton=None):
        self.service_config = service_config
        self.algorithm_config = algorithm_config
        graph_loader = MotionStateGraphLoader()
        graph_loader.set_data_source(service_config["model_data"], algorithm_config["use_transition_model"])
        motion_state_graph = graph_loader.build()
        self.generator = MotionGenerator(motion_state_graph, self.service_config, self.algorithm_config)
        self.request_count = 0

        if target_skeleton is not None:
            self.target_skeleton = target_skeleton
            joint_map = dict()
            src_skeleton = self.generator.get_skeleton()
            for j in src_skeleton.skeleton_model["joints"]:
                src = src_skeleton.skeleton_model["joints"][j]
                if j in self.target_skeleton.skeleton_model["joints"]:
                    joint_map[src] = self.target_skeleton.skeleton_model["joints"][j]
                else:
                    joint_map[src] = None
            self.joint_map = joint_map
        else:
            self.target_skeleton = None
            self.joint_map = None

    def get_target_skeleton(self):
        if self.target_skeleton is None:
            return self.generator.get_skeleton()
        else:
            return self.target_skeleton

def export_motion_to_json(filename, skeleton_data, motion_data):
    export_data = copy(skeleton_data)
    export_data.update(motion_data)
    #export_data = motion_data
    export_data["animated_joints"] = motion_data["jointSequence"]
    with open(filename, "w") as outfile:
        json.dump(export_data, outfile)

def generate_motion(context, mg_input):
    motion_vector = context.generator.generate_motion(mg_input, False, False, complete_motion_vector=False)

    if context.target_skeleton is not None:
        scale_factor = 1.0
        frame_range = None
        src_skeleton = context.generator.get_skeleton()
        target_skeleton = context.get_target_skeleton()
        new_frames = retarget_from_src_to_target(src_skeleton, target_skeleton,
                                                 motion_vector.frames,
                                                 additional_rotation_map=context.additional_rotation_map,
                                                 scale_factor=scale_factor,
                                                 frame_range=frame_range,
                                                 place_on_ground=False)
        motion_vector.skeleton = target_skeleton
        motion_vector.frames = new_frames
        skeleton_data = target_skeleton.to_json()

    else:
        skeleton_data = context.generator.get_skeleton().to_json()

    motion_data = motion_vector.to_unity_format()
    if context.service_config["export_motion_to_file"]:
        filename = context.service_config["output_dir"] + os.sep + "motion_m.json"
        export_motion_to_json(filename, skeleton_data, motion_data)

    return json.dumps(motion_data)


class GenerateMotionHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def post(self):
        global context
        try:
            mg_input = json.loads(self.request.body.decode("utf-8"))
        except Exception as e:
            error_string = "Error: Could not decode request body as JSON." + str(e.args)
            write_message_to_log(error_string, LOG_MODE_ERROR)
            self.write(error_string)
            return


        try:
            id = context.request_count
            print("start task", id)
            context.request_count += 1
            fut = pool.submit(generate_motion, context, mg_input)
            while not fut.done():
                 yield gen.sleep(0.2)  # start process and wait until it is done
            result_str = fut.result()
            print("end task", id)
            self.write(result_str)
        except Exception as e:
            print("caught exception in get")
            self.write("Caught an exception: %s" % e)
            raise
        finally:
            self.finish()

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
        global context
        target_skeleton = context.get_target_skeleton()

        result_object = target_skeleton.to_unity_format(joint_name_map=None)
        self.write(json.dumps(result_object))


def set_height_map(context, data):
    """ https://stackoverflow.com/questions/12511408/accepting-json-image-file-in-python
    https://stackoverflow.com/questions/32908639/open-pil-image-from-byte-file/32908899
    """
    from PIL import Image
    success = False
    if "image_path" in data:
        image_path = data["image_path"]
        width = data["width"]
        depth = data["depth"]
        scale = [1.0, 1.0]
        if "scale" in data:
            scale = data["scale"]
        height_scale = data["height_scale"]
        if os.path.isfile(image_path):
            with open(image_path, "rb") as input_file:
                img = Image.open(input_file)
                img_copy = img.copy()  # work with a copy of the image to close the file
                img.close()
                pixel_is_tuple = not image_path.endswith("bmp")
                print("set height map from file", image_path, "size:",img.size,"mode:", img.mode)
                scene = HeightMapInterface(img_copy, width, depth, scale, height_scale, pixel_is_tuple)
                context.generator.scene_interface.set_scene(scene)
                success = True
    elif "image" in data:
        import base64
        size = data["size"]
        mode = data["mode"]
        width = data["width"]
        depth = data["depth"]
        height_scale = data["height_scale"]
        img = Image.frombytes(mode, size, base64.decodebytes(data["image"]))
        print("set height map from string", "size:", img.size, "mode:", img.mode)
        scene = HeightMapInterface(img, width, depth, height_scale)
        context.generator.scene_interface.set_scene(scene)
        success = True
    return success


class SetHeightMapHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def post(self):
        global context
        try:
            data = json.loads(self.request.body.decode("utf-8"))
        except Exception as e:
            error_string = "Error: Could not decode request body as JSON." + str(e.args)
            write_message_to_log(error_string, LOG_MODE_ERROR)
            self.write(error_string)
            return
        try:
            success = set_height_map(context, data)
            self.write("{'success': "+str(success)+"}")

        except Exception as e:
            print("caught exception in get")
            self.write("Caught an exception: %s" % e)
            raise
        finally:
            self.finish()

app = tornado.web.Application([
    (r"/generate_motion", GenerateMotionHandler),
    (r"/get_skeleton", GetSkeletonHandler),
    (r"/set_height_map", SetHeightMapHandler)
])

SERVICE_CONFIG_FILE = "config" + os.sep + "service.config"
MODEL_DATA_DIR = r"E:\projects\model_data"
context = None
pool = None
def main():
    global context, pool
    port = 8888
    target_skeleton_file = MODEL_DATA_DIR + os.sep + "iclone_female4.bvh"
    skeleton_model = "iclone"
    target_skeleton_file = None
    parser = argparse.ArgumentParser(description="Start the MorphableGraphs REST-interface")
    parser.add_argument("-set", nargs='+', default=[], help="JSONPath expression, e.g. -set $.model_data=path/to/data")
    parser.add_argument("-config_file", nargs='?', default=SERVICE_CONFIG_FILE, help="Path to default config file")
    parser.add_argument("-target_skeleton", nargs='?', default=target_skeleton_file, help="Path to target skeleton file")
    parser.add_argument("-skeleton_scale", nargs='?', default=1.0, help="Scale applied to the target skeleton offsets")
    args = parser.parse_args()
    if os.path.isfile(args.config_file):
            service_config = load_json_file(args.config_file)
            algorithm_config_file = "config" + os.sep + service_config["algorithm_settings"] + "_algorithm.config"
            algorithm_config = load_json_file(algorithm_config_file)
            port = service_config["port"]
            if args.target_skeleton is not None:
                # TODO use custom json file instead
                bvh_reader = BVHReader(args.target_skeleton)
                animated_joints = list(bvh_reader.get_animated_joints())
                target_skeleton = SkeletonBuilder().load_from_bvh(bvh_reader, animated_joints=animated_joints)
                target_skeleton.skeleton_model = SKELETON_MODELS[skeleton_model]
            else:
                target_skeleton = None

            context = Context(service_config, algorithm_config, target_skeleton)
    count = cpu_count()
    print("run {} processes on port {}".format(count, port))
    pool = ProcessPoolExecutor(max_workers=count)

    # configure tornado to work with the asynchio loop
    tornado.platform.asyncio.AsyncIOMainLoop().install()
    app.listen(port)
    asyncio.get_event_loop().run_forever()
    pool.shutdown()


if __name__ == "__main__":
    main()

