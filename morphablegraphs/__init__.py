import os
import sys
# change working directory to the script file directory
file_dir_name, file_name = os.path.split(os.path.abspath(__file__))
sys.path.append(os.sep.join([file_dir_name, 'mgrd']))  # add mgrd package to import path
from .motion_generator import MotionGenerator, GraphWalkOptimizer, DEFAULT_ALGORITHM_CONFIG, AnnotatedMotionVector
