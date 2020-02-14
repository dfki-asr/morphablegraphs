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
import numpy as np
from transformations import quaternion_matrix
from .spatial_constraints.keyframe_constraints import GlobalTransformConstraint
REF_VECTOR = [0,0,-1]

FOOT_OFFSETS = dict()
FOOT_OFFSETS["left"] = np.array([-20,0,0])
FOOT_OFFSETS["right"] = np.array([20,0,0])

def quaternion_from_vector_to_vector(a, b):
    "src: http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another"
    v = np.cross(a, b)
    w = np.sqrt((np.linalg.norm(a) ** 2) * (np.linalg.norm(b) ** 2)) + np.dot(a, b)
    q = np.array([w, v[0], v[1], v[2]])
    return q/ np.linalg.norm(q)


class FootStepConstraintsBuilder(object):
    def __init__(self, skeleton, step_model, precision, settings, foot_offsets=FOOT_OFFSETS):
        self.skeleton = skeleton
        self.step_model = step_model
        self.precision = precision
        self.settings = settings
        self.foot_offsets = foot_offsets

    def generate_step_constraints(self, trajectory, mp_type, start_arc_length, step_arc_length, start_frame, n_canonical_frames):
        """Returns a constraint on the initial stance and final stance foot
        """
        if mp_type not in self.step_model:
            return list()
        init_side = self.step_model[mp_type]["stance_foot"]
        final_side = self.step_model[mp_type]["swing_foot"]
        constraints = []
        if init_side == "both":
            c1 = self._create_foot_constraint(trajectory, start_arc_length, "left", "start", start_frame, n_canonical_frames)
            c2 = self._create_foot_constraint(trajectory, start_arc_length, "right", "start", start_frame, n_canonical_frames)
            constraints += [c1, c2]
        else:
            c = self._create_foot_constraint(trajectory, start_arc_length, init_side, "start", start_frame, n_canonical_frames)
            constraints.append(c)

        last_frame = int(start_frame + n_canonical_frames)
        final_arc_length = start_arc_length + step_arc_length
        if final_side == "both":
            c1 = self._create_foot_constraint(trajectory, final_arc_length, "left", "end", last_frame, n_canonical_frames)
            c2 = self._create_foot_constraint(trajectory, final_arc_length, "right", "end", last_frame, n_canonical_frames)
            constraints += [c1, c2]
        else:
            c = self._create_foot_constraint(trajectory, final_arc_length, final_side, "end", last_frame, n_canonical_frames)
            constraints.append(c)
        return constraints

    def _create_foot_constraint(self, trajectory, arc_length, side, key_frame_label, frame, n_canonical_frames):
        offset = self.foot_offsets[side]
        joint = self.skeleton.skeleton_model["joints"][side+"_heel"]
        pos, dir_vec = trajectory.get_tangent_at_arc_length(arc_length)
        q = quaternion_from_vector_to_vector(REF_VECTOR, dir_vec)
        m = quaternion_matrix(q)[:3, :3]
        foot_position = pos + np.dot(m, offset)
        print(side, arc_length, foot_position)
        return self._create_position_constraint(key_frame_label, frame, joint, foot_position, n_canonical_frames)

    def _create_position_constraint(self, keyframe_label, keyframe, joint_name, position, n_canonical_frames):
        desc = {"joint": joint_name,"canonical_keyframe": keyframe, "position": position, "n_canonical_frames": n_canonical_frames,
                "semanticAnnotation": {"keyframeLabel": keyframe_label, "generated": True}}
        return GlobalTransformConstraint(self.skeleton, desc, self.precision["pos"], self.settings["position_constraint_factor"])
