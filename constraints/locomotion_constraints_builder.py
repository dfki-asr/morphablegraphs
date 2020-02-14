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
from .spatial_constraints.keyframe_constraints import GlobalTransformConstraint, Direction2DConstraint
from .foot_step_constraints_builder import FootStepConstraintsBuilder
from ..utilities.exceptions import PathSearchError
from ..motion_generator.motion_primitive_grounding import MP_CONFIGURATIONS


class LocomotionConstraintsBuilder(object):
    def __init__(self, skeleton, mp_constraint_builder, settings):
        self.skeleton = skeleton
        self.mp_constraint_builder = mp_constraint_builder
        self.settings = settings
        self.precision = {"pos": 1.0, "rot": 1.0, "smooth": 1.0}
        self.use_transition_constraint = self.settings["use_transition_constraint"]
        self.step_model = MP_CONFIGURATIONS
        self.foot_step_constraint_generator = FootStepConstraintsBuilder(self.skeleton, self.step_model, self.precision, self.settings)
        self.generate_half_step_constraint = False
        self.generate_foot_plant_constraints = False
        if "generate_half_step_constraint" in list(self.settings.keys()):
            self.generate_half_step_constraint = self.settings["generate_half_step_constraint"]
        if "generate_foot_plant_constraints" in list(self.settings.keys()):
            self.generate_foot_plant_constraints = self.settings["generate_foot_plant_constraints"]

    def set_algorithm_settings(self, settings):
        self.settings = settings
        self.precision = {"pos": 1.0, "rot": 1.0}
        self.use_transition_constraint = self.settings["use_transition_constraint"]
        if "generate_half_step_constraint" in list(self.settings.keys()):
            self.generate_half_step_constraint = self.settings["generate_half_step_constraint"]

    def add_constraints(self, mp_constraints, node_key, trajectory, prev_arc_length, is_last_step=False):
        # if it is the last step we need to reach the point exactly otherwise
        # make a guess for a reachable point on the path that we have not visited yet
        if not is_last_step:
            goal_arc_length = self._estimate_step_goal_arc_length(node_key, trajectory, prev_arc_length)
        else:
            goal_arc_length = trajectory.full_arc_length

        mp_constraints.goal_arc_length = goal_arc_length

        mp_constraints.step_goal, goal_dir_vector = self._get_point_and_orientation_from_arc_length(trajectory, goal_arc_length)
        mp_constraints.print_status()
        if self.generate_foot_plant_constraints:
            self._add_foot_step_constraints(mp_constraints, node_key, trajectory, prev_arc_length, goal_arc_length)
        else:
            self._add_path_following_goal_constraint(self.skeleton.aligning_root_node, mp_constraints, mp_constraints.step_goal)

        self._add_path_following_direction_constraint(self.skeleton.aligning_root_node, mp_constraints, goal_dir_vector)

        if self.generate_half_step_constraint:
            prev_goal_arc_length = prev_arc_length
            half_step_arc_length = prev_goal_arc_length * 0.5 + mp_constraints.goal_arc_length * 0.5
            half_step_goal, half_step_dir_vector = self._get_point_and_orientation_from_arc_length(trajectory, half_step_arc_length)
            self._add_path_following_half_step_constraint(self.skeleton.aligning_root_node, mp_constraints, half_step_goal)


    def _get_approximate_step_length(self, node_key):
        return self.mp_constraint_builder.motion_state_graph.nodes[node_key].average_step_length * self.settings["heuristic_step_length_factor"]

    def _add_path_following_goal_constraint(self, joint_name, mp_constraints, goal, keyframeLabel="end"):
        if mp_constraints.settings["position_constraint_factor"] > 0.0:
            keyframe_semantic_annotation = {"keyframeLabel": keyframeLabel, "generated": True}
            keyframe_constraint_desc = {"joint": joint_name,
                                        "position": goal,
                                        "semanticAnnotation": keyframe_semantic_annotation}
            keyframe_constraint_desc = self.mp_constraint_builder._map_label_to_canonical_keyframe(keyframe_constraint_desc)
            keyframe_constraint = GlobalTransformConstraint(self.skeleton,
                                                            keyframe_constraint_desc,
                                                            self.precision["pos"],
                                                            mp_constraints.settings["position_constraint_factor"])
            mp_constraints.constraints.append(keyframe_constraint)

    def _add_path_following_half_step_constraint(self, joint_name, mp_constraints, half_step_goal,
                                                 keyframeLabel="middle"):
        if mp_constraints.settings["position_constraint_factor"] > 0.0:
            keyframe_semantic_annotation = {"keyframeLabel": keyframeLabel, "generated": True}
            keyframe_constraint_desc = {"joint": joint_name,
                                        "position": half_step_goal,
                                        "semanticAnnotation": keyframe_semantic_annotation}
            keyframe_constraint_desc = self.mp_constraint_builder._map_label_to_canonical_keyframe(keyframe_constraint_desc)
            keyframe_constraint = GlobalTransformConstraint(self.skeleton,
                                                            keyframe_constraint_desc,
                                                            self.precision["pos"],
                                                            mp_constraints.settings["position_constraint_factor"])
            mp_constraints.constraints.append(keyframe_constraint)

    def _add_path_following_direction_constraint(self, joint_name, mp_constraints, dir_vector):
        if mp_constraints.settings["dir_constraint_factor"] > 0.0:
            dir_semantic_annotation = {"keyframeLabel": "end", "generated": True}
            dir_constraint_desc = {"joint": joint_name, "dir_vector": dir_vector,
                                   "semanticAnnotation": dir_semantic_annotation}
            dir_constraint_desc = self.mp_constraint_builder._map_label_to_canonical_keyframe(dir_constraint_desc)
            direction_constraint = Direction2DConstraint(self.skeleton, dir_constraint_desc, self.precision["rot"],
                                                         mp_constraints.settings["dir_constraint_factor"])
            mp_constraints.constraints.append(direction_constraint)

    def _estimate_step_goal_arc_length(self, node_key, trajectory, prev_arc_length):
        """ Makes a guess for a reachable arc length based on the current position.
            It searches for the closest point on the trajectory, retrieves the absolute arc length
            and its the arc length of a random sample of the next motion primitive
        Returns
        -------
        * arc_length : float
          The absolute arc length of the new goal on the trajectory.
          The goal should then be extracted using get_point_and_orientation_from_arc_length
        """
        step_length = self._get_approximate_step_length(node_key)
        # find closest point in the range of the last_arc_length and max_arc_length
        # closest_point = self.find_closest_point_to_current_position_on_trajectory(step_length)
        # approximate arc length of the point closest to the current position
        # start_arc_length, eval_point = self.action_constraints.root_trajectory.get_absolute_arc_length_of_point(closest_point)

        start_arc_length = prev_arc_length #last arc length is already found as closest point on path to current position
        # update arc length based on the step length of the next motion primitive
        if start_arc_length == -1:
            return trajectory.full_arc_length
        else:
            return start_arc_length + step_length

    def find_closest_point_to_current_position_on_trajectory(self, trajectory, step_length, prev_pos, prev_arc_length):
        """ find closest point in the range of the prev_arc_length and max_arc_length
             approximate arc length of the point closest to the current position"""
        max_arc_length = prev_arc_length + step_length * 4.0
        closest_point, distance = trajectory.find_closest_point(prev_pos, prev_arc_length, max_arc_length)
        if closest_point is None:
            self._raise_closest_point_search_exception(trajectory, max_arc_length, prev_arc_length)
        return closest_point

    def _get_point_and_orientation_from_arc_length(self, trajectory, arc_length):
        """ Returns a point, an orientation and a direction vector on the trajectory
        """
        point = trajectory.query_point_by_absolute_arc_length(arc_length).tolist()
        #reference_vector = np.array([0.0, 1.0])  # is interpreted as x, z
        #start, dir_vector, angle = Trajectory.get_angle_at_arc_length_2d(arc_length, reference_vector)
        #delta = trajectory.full_arc_length - arc_length
        # dir_vector =  trajectory.get_direction_vector_by_absolute_arc_length(arc_length)
        # print "orientation vector", dir_vector,dir_vector1
        dir_vector = trajectory.query_orientation_by_absolute_arc_length(arc_length)
        dir_vector /= np.linalg.norm(dir_vector)
        for i in trajectory.unconstrained_indices:
            point[i] = None
        return point, dir_vector

    def _raise_closest_point_search_exception(self, trajectory, max_arc_length, prev_arc_length):
        parameters = {"last": prev_arc_length, "max": max_arc_length,
                       "full": trajectory.full_arc_length}
        print("Error: Did not find closest point", str(parameters))
        raise PathSearchError(parameters)


    def _add_foot_step_constraints(self, mp_constraints, node_key, trajectory, prev_arc_length, goal_arc_length):
        n_prev_frames = self.mp_constraint_builder.status["n_prev_frames"]
        n_canonical_frames = self.mp_constraint_builder.status["n_canonical_frames"]
        constraints = self.foot_step_constraint_generator.generate_step_constraints(trajectory, node_key[1],
                                                                                             prev_arc_length,
                                                                                             goal_arc_length,
                                                                                             n_prev_frames,
                                                                                             n_canonical_frames)
        mp_constraints.constraints += constraints

