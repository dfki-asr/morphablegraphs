NODE_TYPE_START = "start"
NODE_TYPE_STANDARD = "standard"
NODE_TYPE_END = "end"
NODE_TYPE_IDLE = "idle"
NODE_TYPE_SINGLE = "single_primitive"
NODE_TYPE_CYCLE_END = "cycle_end"
B_SPLINE_DEGREE = 3
ELEMENTARY_ACTION_DIRECTORY_NAME = "elementary_action_models"
TRANSITION_MODEL_DIRECTORY_NAME = "transition_models"
META_INFORMATION_FILE_NAME = "meta_information.json"
TRANSITION_DEFINITION_FILE_NAME = "graph_definition.json"
TRANSITION_MODEL_FILE_ENDING = ".GPM"

from .motion_state_graph_loader import MotionStateGraphLoader
