import os
import time
from anim_utils.utilities.io_helper_functions import load_json_file
from morphablegraphs.construction.cluster_tree_builder import ClusterTreeBuilder, TREE_TYPE_CLUSTER_TREE, TREE_TYPE_FEATURE_CLUSTER_TREE, FEATURE_TYPE_S_VECTOR

dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname)

CONIFG_FILE_PATH = "config" + os.sep + "space_partitioning.json"

def main():
    skeleton_path = "skeleton.bvh"
    skeleton_path = "raw_skeleton.bvh"
    settings = dict()
    settings["tree_type"] = TREE_TYPE_FEATURE_CLUSTER_TREE
    settings["feature_type"] = FEATURE_TYPE_S_VECTOR
    settings["output_mode"] = "json"
    cluster_tree_builder = ClusterTreeBuilder(settings)
    config = load_json_file(CONIFG_FILE_PATH)
    cluster_tree_builder.set_config(config)
    cluster_tree_builder.load_skeleton(skeleton_path)
    start = time.clock()
    success = cluster_tree_builder.build()

    time_in_seconds = time.clock()-start
    if success:
        print("Finished construction in", int(time_in_seconds/60), "minutes and", time_in_seconds % 60, "seconds")
    else:
        print("Failed to read data from directory")

if __name__ == "__main__":
    main()