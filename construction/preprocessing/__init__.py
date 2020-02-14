LEN_ROOT = 3
LEN_QUATERNION = 4
LEN_EULER = 3
from .bitvector import gen_foot_contact_annotation
from .semantic_annotation import gen_walk_annotation, gen_synthetic_semantic_annotation_pick_and_place, \
                                 gen_synthetic_semantic_annotation_for_screw, \
                                 gen_synthetic_semantic_annotation_for_transfer, \
                                 create_low_level_semantic_annotation
from .motion_normalization import MotionNormalization
from .motion_dtw import MotionDynamicTimeWarping