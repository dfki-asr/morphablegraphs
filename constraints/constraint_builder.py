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
import collections
from transformations import quaternion_multiply, quaternion_inverse, quaternion_slerp, quaternion_matrix, quaternion_from_matrix
from morphablegraphs.constraints.spatial_constraints.keyframe_constraints import GlobalTransformConstraint, Direction2DConstraint, RelativeTransformConstraint
from morphablegraphs.constraints.spatial_constraints.keyframe_constraints.pose_constraint import PoseConstraint
from morphablegraphs.motion_model import MotionStateGraphLoader, NODE_TYPE_STANDARD, NODE_TYPE_END, NODE_TYPE_START, NODE_TYPE_IDLE, NODE_TYPE_SINGLE
from anim_utils.animation_data.skeleton_models import STANDARD_MIRROR_MAP
from morphablegraphs.constraints.motion_primitive_constraints import MotionPrimitiveConstraints


def unity_frame_to_mg_frame(skeleton, unity_frame, animated_joints, scale):
    #unity_frame = {"rotations": [], "rootTranslation": None, "action": action, "events": events, "isIdle": is_idle, "success": success}
    n_dims = len(animated_joints) * 4 + 3
    frame = np.zeros(n_dims)
    frame[0] = -unity_frame["rootTranslation"]["x"]
    frame[1] = unity_frame["rootTranslation"]["y"]
    frame[2] = unity_frame["rootTranslation"]["z"]
    idx = 0
    o = 3
    for node_name in skeleton.nodes.keys():
        if node_name in animated_joints:
            q = unity_frame["rotations"][idx]
            q =np.array([-q["w"], -q["x"],q["y"],q["z"]])
            frame[o:o+4] =  q
            o+=4
            idx +=1
    return frame


class MockActionConstraints(object):
    def __init__(self, action_name, mg):
        self.motion_state_graph = mg
        self.action_name = action_name
        self.prev_action_name = None


class UnityFrameConstraint(object):
    def __init__(self, node, keyframe_label, joint, position, orientation, hold_frame=False, offset=None, end_keyframe_label=None):
        self.node = node #  the graph node on which the constraint should be applied
        self.joint = joint
        self.position = position
        self.orientation = orientation
        self.hold_frame = hold_frame
        self.keyframe_events = list()
        self.offset = offset # optional tool offset
        self.keyframe_label = keyframe_label
        self.keyframe = None # gets looked up later based on keyframe_label
        self.end_keyframe_label = end_keyframe_label
        self.end_keyframe = None  # gets looked up later based on end_keyframe_label
        self.relative_joint_name = None
        self.mirror_joint_name = None  # create a dynamic constraint to keep the mirror joint at its original position
        
        # create a constraint on the parent position
        self.constrained_parent = None
        self.vector_to_parent= None  

        # optional tool alignment
        self.src_tool_cos = None
        self.dest_tool_cos = None

        # needs to be activated to enact the constraint in the region between keyframe and end_keyframe
        self.constrain_position_in_region = False
        self.constrain_orientation_in_region = False

        self.cycle = 0 # needed for assignment to nodes in cyclic actions that repeat nodes multiple times
        self.look_at = False
        



class ConstraintBuilder(object):
    def __init__(self, skeleton, graph, planner_settings, algorithm_config):
        self.skeleton = skeleton
        self._graph = graph
        self.settings = planner_settings
        self.algorithm_config = algorithm_config
        self.constrained_joints = []
        self.joint_weights_map = collections.OrderedDict()
        joint_map = self.skeleton.skeleton_model["joints"]
        for idx, j in enumerate(["right_wrist", "left_wrist", "right_ankle", "left_ankle"]):
            node_name = joint_map[j]
            self.constrained_joints.append(node_name)
            self.joint_weights_map[idx] = self.skeleton.joint_weight_map[node_name]
        self.inv_joint_map = dict()
        if self.skeleton.skeleton_model is not None:
            for j in self.skeleton.skeleton_model["joints"]:
                skel_j = self.skeleton.skeleton_model["joints"][j]
                if skel_j is not None:
                    self.inv_joint_map[skel_j] = j

        self.action_definitions = dict()
        if hasattr(self._graph, "action_definitions") and self._graph.action_definitions is not None:
            print("load action definitions from file")
            self.action_definitions = self._graph.action_definitions
        print("loaded actions", list(self.action_definitions.keys()))

    def generate_walk_dir_constraint(self, dir_vector, n_frames, aligning_transform, w=1.0):
        local_dir_vec = np.dot(aligning_transform, [dir_vector[0], 0, dir_vector[2], 0])[:3]
        length = np.linalg.norm(local_dir_vec)
        if length <= 0:
            return None

        local_dir_vec /= length
        #print("dir", dir_vector, local_dir_vec)
        c_desc = {"joint": self.skeleton.root, "canonical_keyframe": n_frames-1,
                  "dir_vector": local_dir_vec,
                  "n_canonical_frames": n_frames, "semanticAnnotation": {"keyframeLabel": "none"}}
        return Direction2DConstraint(self.skeleton, c_desc, w, 1.0)

    def generate_walk_position_constraint(self, dir_vector, distance, n_frames, aligning_transform, w=1.0):
        local_dir_vec = np.dot(aligning_transform, [dir_vector[0], 0, dir_vector[2], 0])[:3]
        local_dir_vec /= np.linalg.norm(local_dir_vec)
        position = local_dir_vec * distance

        c_desc = {"joint": self.skeleton.root, "canonical_keyframe": n_frames-1,
                  "position": position,
                  "n_canonical_frames": n_frames, "semanticAnnotation": {"keyframeLabel": "none"}}
        return GlobalTransformConstraint(self.skeleton, c_desc, w, 1.0)

    def generate_transform_constraint(self, node, keyframe, joint_name, position, orientation, n_frames, aligning_transform,
                                      offset=None, end_keyframe=None, keep_orientation=False, relative_joint_name=None):
        local_position = np.dot(aligning_transform, [position[0], position[1], position[2], 1])[:3]
        c_desc = {"joint": joint_name, "canonical_keyframe": keyframe,
                  "position": local_position,
                  "n_canonical_frames": n_frames, "semanticAnnotation": {"keyframeLabel": "none"},
                  "canonical_end_keyframe": end_keyframe,
                  "keep_orientation": keep_orientation,
                  "relative_joint_name": relative_joint_name}

        if orientation is not None:
            local_orientation = np.dot(aligning_transform, quaternion_matrix(orientation))
            lq = quaternion_from_matrix(local_orientation)
            lq /= np.linalg.norm(lq)
            c_desc["qOrientation"] = lq
            #print("set orientation", c_desc["qOrientation"])
        if offset is not None:
            c_desc["offset"] = offset
            return RelativeTransformConstraint(self.skeleton, c_desc, 1.0, 1.0)
        else:
            return GlobalTransformConstraint(self.skeleton, c_desc, 1.0, 1.0)

    def generate_mg_constraint_from_unity_constraint(self, constraint, joint_name, n_frames, aligning_transform=None):
        position = constraint.position
        if aligning_transform is not None:
            local_position = np.dot(aligning_transform, [position[0], position[1], position[2], 1])[:3]
        else:
            local_position = np.array([position[0], position[1], position[2]])

        c_desc = {"joint": joint_name, "canonical_keyframe": constraint.keyframe,
                  "position": local_position,
                  "n_canonical_frames": n_frames, "semanticAnnotation": {"keyframeLabel": "none"},
                  "canonical_end_keyframe": constraint.end_keyframe,
                  "relative_joint_name": constraint.relative_joint_name,
                  "mirror_joint_name": constraint.mirror_joint_name,
                  "constrained_parent": constraint.constrained_parent
                  }

        
        c_desc["constrain_position_in_region"] = constraint.constrain_position_in_region
        c_desc["constrain_orientation_in_region"] = constraint.constrain_orientation_in_region
        c_desc["look_at"] = constraint.look_at
        if constraint.orientation is not None:
            orientation = constraint.orientation
            if aligning_transform is not None:
                local_orientation = np.dot(aligning_transform, quaternion_matrix(orientation))
                lq = quaternion_from_matrix(local_orientation)
                lq /= np.linalg.norm(lq)
            else:
                lq = orientation
            c_desc["qOrientation"] = lq
            # print("set orientation", c_desc["qOrientation"])
        if constraint.vector_to_parent is not None:
            vector_to_parent = constraint.vector_to_parent
            if aligning_transform is not None:
                vector_to_parent = np.dot(aligning_transform, [vector_to_parent[0], vector_to_parent[1], vector_to_parent[2], 0])[:3]
            c_desc["vector_to_parent"] = vector_to_parent
        if constraint.src_tool_cos is not None and constraint.dest_tool_cos is not None:
            src_tool_cos = dict()
            dest_tool_cos = dict()
            for a in ["x", "y"]:
                if  a in constraint.src_tool_cos and a in constraint.dest_tool_cos:
                    src_tool_cos[a]= constraint.src_tool_cos[a]     
                    dest_tool_axis = constraint.dest_tool_cos[a]
                    if aligning_transform is not None:
                        dest_tool_axis = np.dot(aligning_transform, [dest_tool_axis[0], dest_tool_axis[1], dest_tool_axis[2], 0])[:3]
                    dest_tool_cos[a] = dest_tool_axis
            c_desc["src_tool_cos"]= src_tool_cos
            c_desc["dest_tool_cos"]= dest_tool_cos

        if constraint.offset is not None: 
            # TODO check if offset needs to be brought into local coordinate system
            c_desc["offset"] = constraint.offset
            return RelativeTransformConstraint(self.skeleton, c_desc, 1.0, 1.0)
        else:
            return GlobalTransformConstraint(self.skeleton, c_desc, 1.0, 1.0)

    def map_keyframe_labels_to_frame_indices(self, frame_constraints):
        for c in frame_constraints:
            c.keyframe = self._get_keyframe_from_label(c.node, c.keyframe_label)
            if c.end_keyframe_label is not None:
                c.end_keyframe = self._get_keyframe_from_label(c.node, c.end_keyframe_label)

    def _get_keyframe_from_label(self, node, keyframe_label):
        return self._graph.node_groups[node[0]].map_label_to_keyframe(node[1], keyframe_label)

    def generate_transition_constraint(self, pose_buffer, aligning_transform):
        last_pose = self.skeleton.convert_quaternion_frame_to_cartesian_frame(pose_buffer[-1], self.constrained_joints)
        for i,p in enumerate(last_pose):
            last_pose[i] = np.dot(aligning_transform[:3,:3], last_pose[i])

        c_desc = {"keyframeLabel": "start",
                "frame_constraint": last_pose,
                "semanticAnnotation": {"keyframeLabel": "start"},
                "node_names": self.constrained_joints,
                "weights": self.joint_weights_map,
                "canonical_keyframe": 0}

        return PoseConstraint(self.skeleton, c_desc, 1.0, 1.0)

    def extract_tool_offset(self, joint_name, constraint_desc):
        tool_offset = None
        src_cos = None
        dest_cos = None
        if "offset" in constraint_desc and "applyOffset" in constraint_desc and constraint_desc["applyOffset"]:
            tool_offset = constraint_desc["offset"]
            if "toolEndPoint" in constraint_desc and "currentPose" in constraint_desc:
                print("try to overwrite offset", tool_offset)
                tp = constraint_desc["toolEndPoint"]
                unity_frame = constraint_desc["currentPose"]
                frame = unity_frame_to_mg_frame(self.skeleton, unity_frame, self.skeleton.animated_joints, 1)
                m = self.skeleton.nodes[joint_name].get_global_matrix(frame)
                #p = self.skeleton.nodes[joint_name].get_global_position(frame)
                #g_offset = tp-p
                tp = np.array([tp[0],tp[1],tp[2],1])
                inv_m = np.linalg.inv(m)
                l_offset = np.dot(inv_m, tp)
                print("overwrite offset", tool_offset, l_offset, tp)
                tool_offset = l_offset
                if "useToolCos" in constraint_desc and constraint_desc["useToolCos"]:
                    src_cos = dict()
                    dest_cos = dict()
                    if "destToolCos"  in constraint_desc and "srcToolCos" in constraint_desc:
                        for a in ["x", "y"]:
                            if a in constraint_desc["srcToolCos"] and a in constraint_desc["destToolCos"]:
                                target_tool_vector = constraint_desc["destToolCos"][a]
                                magnitude = np.linalg.norm(target_tool_vector)
                                if magnitude <= 0:
                                    src_cos = None
                                    dest_cos = None
                                    return tool_offset, None, None
                                target_tool_vector /= magnitude
                                dest_cos[a] = target_tool_vector

                                g_axis_point = constraint_desc["srcToolCos"][a]
                                tp = np.array([g_axis_point[0],g_axis_point[1],g_axis_point[2],1])
                                inv_m = np.linalg.inv(m)
                                tool_axis_offset = np.dot(inv_m, tp)[:3]
                                # remove tool offset to get relative axis
                                tool_axis_offset -= tool_offset[:3]
                                tool_axis_offset /= np.linalg.norm(tool_axis_offset)
                                tool_axis_offset = np.array([tool_axis_offset[0], tool_axis_offset[1],tool_axis_offset[2], 0])
                                src_cos[a] = tool_axis_offset
        return tool_offset, src_cos, dest_cos

    def create_frame_constraint(self, action_name, constraint_desc, look_at=False):
        keyframe_label = constraint_desc["keyframe"]
        print("generate constraint", action_name, keyframe_label)
        joint_name = constraint_desc["joint"]
        position = constraint_desc["position"]
        constrain_orientation_in_region = False
        constrain_position_in_region = False
        if constraint_desc["constrainOrientation"]:
            orientation = constraint_desc["orientation"]
            if "constrainOrientationInRegion" in constraint_desc:
                constrain_orientation_in_region = constraint_desc["constrainOrientationInRegion"]
        else:
            orientation = None
        constraint_slots = self.action_definitions[action_name]["constraint_slots"]
        cycle = 0
        if "cycle" in constraint_desc:
            cycle = constraint_desc["cycle"]

        if "cycle_nodes" in  constraint_slots[keyframe_label]:
            if cycle < len(constraint_slots[keyframe_label]["cycle_nodes"]):
                mp_name = constraint_slots[keyframe_label]["cycle_nodes"][cycle]
            else:
                mp_name = constraint_slots[keyframe_label]["cycle_nodes"][-1]
        else:
           mp_name = constraint_slots[keyframe_label]["node"]
        print("set constraint", mp_name, cycle)
        if joint_name is None:
            joint_name = constraint_slots[keyframe_label]["joint"]
        hold_frame = False
        if "hold" in constraint_desc and constraint_desc["hold"]:
            hold_frame = True

        tool_offset, src_tool_cos, dest_tool_cos= self.extract_tool_offset(joint_name, constraint_desc)

        node = (action_name, mp_name)
        end_keyframe_label = None
        if "constrainPositionInRegion" in constraint_desc:
            constrain_position_in_region = constraint_desc["constrainPositionInRegion"]
        if "endKeyframe" in constraint_desc and constrain_position_in_region or constrain_orientation_in_region:
            if constraint_desc["endKeyframe"] != "":
                end_keyframe_label = constraint_desc["endKeyframe"]
        frame_constraint = UnityFrameConstraint(node, keyframe_label, joint_name, position, orientation, hold_frame, tool_offset, end_keyframe_label)
        frame_constraint.constrain_orientation_in_region = constrain_orientation_in_region
        frame_constraint.constrain_position_in_region = constrain_position_in_region
        frame_constraint.cycle = cycle
        frame_constraint.look_at = look_at

        if "keyframeEvents" in constraint_desc:
            frame_constraint.keyframe_events = constraint_desc["keyframeEvents"]

        if "keepOffsetBetweenBones" in constraint_desc and constraint_desc["keepOffsetBetweenBones"]:
            if constraint_desc["relativeBoneName"] in self.skeleton.nodes:
                frame_constraint.relative_joint_name = constraint_desc["relativeBoneName"]
        if "keepMirrorBoneStatic" in constraint_desc and constraint_desc["keepMirrorBoneStatic"]:
            mirror_joint_name = self.get_mirror_joint_name(joint_name)
            print("set mirror joint name", mirror_joint_name, joint_name)
            frame_constraint.mirror_joint_name = mirror_joint_name
        if "constrainedParent" in constraint_desc and constraint_desc["constrainedParent"] !="" and "vectorToParent" in constraint_desc:
            constraint_parent = constraint_desc["constrainedParent"]
            frame_constraint.vector_to_parent = constraint_desc["vectorToParent"]
            frame_constraint.constrained_parent = constraint_parent
            print("Found constrained parent", frame_constraint.constrained_parent, frame_constraint.vector_to_parent)

        frame_constraint.src_tool_cos = src_tool_cos
        frame_constraint.dest_tool_cos = dest_tool_cos
        return frame_constraint

    def extract_constraints_from_dict(self, action_desc, look_at_constraints=False):
        action_name = action_desc["name"]
        end_direction = None
        if "orientationVector" in action_desc:
            end_direction = action_desc["orientationVector"]
        frame_constraints = list()
        if action_name in self.action_definitions and "constraint_slots" in self.action_definitions[action_name]:
            frame_constraints = self.create_frame_constraints(action_name, action_desc, look_at_constraints)            
        print("frame constraints", action_name)

        look_at_target = None
        spine_target = None
        if "constrainLookAt" in action_desc and action_desc["constrainLookAt"]:
            if "lookAtTarget" in action_desc and np.linalg.norm(action_desc["lookAtTarget"]) > 0:
                look_at_target = action_desc["lookAtTarget"]
            if "spineTarget" in action_desc and np.linalg.norm(action_desc["spineTarget"]) > 0:
                spine_target = action_desc["spineTarget"]
        body_orientation_targets = (look_at_target, spine_target)
        print("body orientation targets", body_orientation_targets)
        return frame_constraints, end_direction, body_orientation_targets

    def create_frame_constraints(self, action_name, action_desc, look_at_constraints):
        frame_constraints = list()
        if "frameConstraints" in action_desc:
            for constraint_desc in action_desc["frameConstraints"]:
                frame_constraint = self.create_frame_constraint(action_name, constraint_desc, look_at_constraints)
                frame_constraints.append(frame_constraint)
        return frame_constraints

    def get_mirror_joint_name(self, joint_name):
        mirror_joint_name = None
        if joint_name in self.inv_joint_map:
            std_joint_name = self.inv_joint_map[joint_name]
            if std_joint_name in STANDARD_MIRROR_MAP:
                std_mirror_joint_name = STANDARD_MIRROR_MAP[std_joint_name]
                if std_mirror_joint_name in self.skeleton.skeleton_model["joints"]:
                    mirror_joint_name = self.skeleton.skeleton_model["joints"][std_mirror_joint_name]
        return mirror_joint_name

    def generate_walk_constraints(self, current_node, aligning_transform, direction_vector, distance, pose_buffer):
        n_frames = self._graph.nodes[current_node].get_n_canonical_frames()
        mp_constraints = MotionPrimitiveConstraints()
        mp_constraints.skeleton = self.skeleton
        mp_constraints.constraints = list()
        pos_constraint = self.generate_walk_position_constraint(direction_vector, distance, n_frames, aligning_transform, self.settings.position_constraint_weight)
        mp_constraints.constraints.append(pos_constraint)
        dir_constraint = self.generate_walk_dir_constraint(direction_vector, n_frames, aligning_transform,
                                                           self.settings.direction_constraint_weight)
        if dir_constraint is not None:
            mp_constraints.constraints.append(dir_constraint)
        if self.settings.add_transition_constraint:
            c = self.generate_transition_constraint(pose_buffer, aligning_transform)
            mp_constraints.constraints.append(c)
        mp_constraints.is_local = True
        return mp_constraints


    def generate_motion_primitive_constraints(self, current_node, aligning_transform, action_constraints, pose_buffer):
        apply_ik = False
        n_frames = self._graph.nodes[current_node].get_n_canonical_frames()
        mp_constraints = MotionPrimitiveConstraints()
        mp_constraints.skeleton = self.skeleton
        mp_constraints.constraints = list()
        for frame_constraint in action_constraints:
            joint_name = None
            if frame_constraint.joint in self.skeleton.skeleton_model["joints"]:
                joint_name = self.skeleton.skeleton_model["joints"][frame_constraint.joint]
            elif frame_constraint.joint in self.skeleton.nodes:
                joint_name = frame_constraint.joint
            else:
                print("Error: could not assign joint", frame_constraint.joint)
            if joint_name is not None:
                c = self.generate_mg_constraint_from_unity_constraint(frame_constraint, joint_name,
                                                                                         n_frames, aligning_transform)
                mp_constraints.constraints.append(c)
                mp_constraints.use_local_optimization = self.algorithm_config["local_optimization_mode"] in ["keyframes",
                                                                                                             "all"]
                apply_ik = self.settings.activate_ik
                print("add end effector constraints", c.joint_name, c.relative_joint_name, c.keyframe_label, c.canonical_keyframe, c.canonical_end_keyframe)

        if self.settings.add_transition_constraint and not apply_ik:
            c = self.generate_transition_constraint(pose_buffer, aligning_transform)
            mp_constraints.constraints.append(c)
        mp_constraints.is_local = True
        return mp_constraints, apply_ik