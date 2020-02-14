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
# -*- coding: utf-8 -*-
import collections

OLD_ALGORITHM_CONFIG = {
    "smoothing_settings": {
        "spatial_smoothing": True,
        "time_smoothing": False,
        "spatial_smoothing_method": "smoothing",
        "spatial_smoothing_window": 20,
        "time_smoothing_window": 15,
        "apply_foot_alignment": True,
        "root_filter_window": 0

    },
    "trajectory_following_settings": {
        "spline_type": 0,
        "control_point_filter_threshold": 0,
        "dir_constraint_factor": 0.1,
        "heuristic_step_length_factor": 1.0,
        "position_constraint_factor": 1.0,
        "step_length_approx_method": "arc_length",
        "transition_pose_constraint_factor": 0.6,
        "closest_point_search_accuracy": 0.001,
        "closest_point_search_max_iterations": 5000,
        "look_ahead_distance": 100,
        "end_step_length_factor": 1.0,
        "max_distance_to_path": 500,
        "arc_length_granularity": 1000,
        "use_transition_constraint" : False,
        "spline_super_sampling_factor": 20,
        "constrain_start_orientation": True,
        "constrain_transition_orientation": True,
        "generate_half_step_constraint": True,
        "generate_foot_plant_constraints": False

    },
    "local_optimization_settings": {
        "start_error_threshold": 0.0,
        "error_scale_factor": 1.0,
        "spatial_epsilon": 0.0,
        "quality_scale_factor": 1.0,
        "tolerance": 0.05,
        "method": "leastsq",
        "max_iterations": 500,
        "verbose": False
    },
    "global_spatial_optimization_settings": {
        "max_steps":  3,
        "start_error_threshold": 4.0,
        "error_scale_factor": 1.0,
        "quality_scale_factor": 100.0,
        "tolerance": 0.05,
        "method": "leastsq",
        "max_iterations": 500,
        "position_weight": 1000.0,
        "orientation_weight": 1000.0,
        "verbose": False
    },
    "global_time_optimization_settings": {
        "error_scale_factor": 1.0,
        "quality_scale_factor": 0.0001,
        "tolerance": 0.05,
        "method": "L-BFGS-B",
        "max_iterations": 500,
        "optimized_actions": 2,
        "verbose": False
    },
    "inverse_kinematics_settings":{
        "tolerance": 0.05,
        "optimization_method": "L-BFGS-B",
        "max_iterations": 1000,
        "interpolation_window": 120,
        "transition_window": 60,
        "use_euler_representation": False,
        "solving_method": "unconstrained",
        "activate_look_at": True,
        "max_retries": 5,
        "success_threshold": 5.0,
        "optimize_orientation": True,
        "elementary_action_max_iterations": 5,
        "elementary_action_optimization_eps": 1.0,
        "adapt_hands_during_carry_both": True,
        "constrain_place_orientation": False
    },
    "motion_grounding_settings":{
         "activate_blending": True,
         "generate_foot_plant_constraints": True,
         "foot_lift_search_window": 40,
         "foot_lift_tolerance": 3.0,
         "graph_walk_grounding_window": 4,
         "contact_tolerance": 1.0,
         "constraint_range": 10,
         "smoothing_constraints_window": 8
    },
    "n_random_samples": 100,
    "average_elementary_action_error_threshold": 500,
    "constrained_sampling_mode": "cluster_tree_search",
    "activate_inverse_kinematics": True,
    "activate_motion_grounding": True,
    "n_cluster_search_candidates": 4,
    "use_transition_model": False,
    "local_optimization_mode": "all",
    "activate_parameter_check": False,
    "use_global_time_optimization": True,
    "global_spatial_optimization_mode": "trajectory_end",
    "collision_avoidance_constraints_mode": "direct_connection",
    "optimize_collision_avoidance_constraints_extra": False,
    "use_constrained_gmm": False,
    "use_constraints": True,
    "use_local_coordinates": True,
    "use_semantic_annotation_with_mgrd": False,
    "activate_time_variation": True,
    "debug_max_step": -1,
    "verbose": False
}



DEFAULT_ALGORITHM_CONFIG = collections.OrderedDict({
    "smoothing_settings": {
        "spatial_smoothing" : True,
        "time_smoothing" : False,
        "spatial_smoothing_method": "smoothing",
        "spatial_smoothing_window": 20,
        "time_smoothing_window": 15,
        "apply_foot_alignment": False,
        "root_filter_window": 0

    },
    "trajectory_following_settings": {
        "spline_type": 0,
        "control_point_filter_threshold": 0,
        "dir_constraint_factor": 0.8,
        "heuristic_step_length_factor": 1.0,
        "position_constraint_factor": 1.0,
        "step_length_approx_method": "arc_length",
        "transition_pose_constraint_factor": 0.6,
        "closest_point_search_accuracy": 0.001,
        "closest_point_search_max_iterations": 5000,
        "look_ahead_distance": 100,
        "end_step_length_factor": 1.0,
        "max_distance_to_path": 500,
        "arc_length_granularity": 1000,
        "use_transition_constraint": False,
        "spline_super_sampling_factor": 20,
        "constrain_start_orientation": True,
        "constrain_transition_orientation": True,
        "generate_half_step_constraint": False,
        "generate_foot_plant_constraints": False
    },
    "local_optimization_settings": {
        "start_error_threshold": 0.0,
        "error_scale_factor": 1.0,
        "spatial_epsilon": 0.0,
        "quality_scale_factor": 0.1,
        "tolerance": 0.05,
        "method": "leastsq",#"L-BFGS-B",#
        "max_iterations": 500,
        "verbose": False,
        "diff_eps": 1.0
    },
    "global_spatial_optimization_settings": {
        "max_steps":  3,
        "start_error_threshold": 4.0,
        "error_scale_factor": 1.0,
        "quality_scale_factor": 100.0,
        "tolerance": 0.05,
        "method": "leastsq",
        "max_iterations": 500,
        "position_weight": 1000.0,
        "orientation_weight": 1000.0,
        "verbose": False,
        "diff_eps": 2.0
    },
    "global_time_optimization_settings": {
        "error_scale_factor": 1.0,
        "quality_scale_factor": 0.0001,
        "tolerance": 0.05,
        "method": "L-BFGS-B",
        "max_iterations": 500,
        "optimized_actions": 2,
        "verbose": False,
        "diff_eps": 1.0
    },
    "inverse_kinematics_settings":{
        "tolerance": 0.05,
        "optimization_method": "L-BFGS-B",
        "max_iterations": 1000,
        "interpolation_window": 120,
        "transition_window": 60,
        "use_euler_representation": False,
        "solving_method": "unconstrained",
        "activate_look_at": True,
        "max_retries": 5,
        "success_threshold": 5.0,
        "optimize_orientation": True,
        "elementary_action_max_iterations": 5,
        "elementary_action_optimization_eps": 1.0,
        "adapt_hands_during_carry_both": True,
        "constrain_place_orientation": False,
        "activate_blending": True
    },
    "motion_grounding_settings":{
           "activate_blending": True,
           "generate_foot_plant_constraints": True,
           "foot_lift_search_window": 40,
           "foot_lift_tolerance": 3.0,
           "graph_walk_grounding_window": 4,
           "contact_tolerance": 1.0,
           "constraint_range": 10,
           "smoothing_constraints_window": 8,
           "damp_angle": 0.01,
           "damp_factor": 1.0
     },
    "n_random_samples": 100,
    "average_elementary_action_error_threshold": 500,
    "constrained_sampling_mode": "cluster_tree_search",
    "activate_inverse_kinematics": True,
    "activate_motion_grounding": False,
    "n_cluster_search_candidates": 4,
    "use_transition_model": False,
    "local_optimization_mode": "all",
    "activate_parameter_check": False,
    "use_global_time_optimization": True,
    "global_spatial_optimization_mode": "none",
    "collision_avoidance_constraints_mode": "direct_connection",
    "optimize_collision_avoidance_constraints_extra": False,
    "use_constrained_gmm": False,
    "use_constraints": True,
    "use_local_coordinates": True,
    "use_semantic_annotation_with_mgrd": False,
    "activate_time_variation": True,
    "debug_max_step": -1,
    "verbose": False
})
