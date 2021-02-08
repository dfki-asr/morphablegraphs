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
import numpy
import time
from ..constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
from transformations import quaternion_matrix, quaternion_from_matrix
try:
    from mgrd import Constraint, SemanticConstraint
    from mgrd import CartesianConstraint as MGRDCartesianConstraint
    from mgrd import ForwardKinematics as MGRDForwardKinematics
    from mgrd import score_splines_with_semantic_pose_constraints
    has_mgrd = True
except ImportError:
    print("Import failed")
    pass
    has_mgrd = False


class MGRDSampleFilter(object):
    """ Wrapper around MGRD sample scoring methods to estimate the best fit parameters for one motion primitive.
    """
    def __init__(self, pose_constraint_weights=(1,1)):
        self.pose_constraint_weights = pose_constraint_weights

    @staticmethod
    def transform_coeffs(qs, transform):
        for c in qs.coeffs:
            c[:3] = numpy.dot(transform, c[:3].tolist()+[1])[:3]
            c[3:7] = quaternion_from_matrix(numpy.dot(transform, quaternion_matrix(c[3:7])))

    @staticmethod
    def extract_cartesian_constraints(mp_constraints):
        mgrd_constraints = []
        for c in mp_constraints.constraints:
            if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
                position = [p if p is not None else 0 for p in c.position]
                #print "local", position, c.joint_name
                cartesian_constraint = MGRDCartesianConstraint(position, c.joint_name, c.weight_factor)
                mgrd_constraints.append(cartesian_constraint)
        return mgrd_constraints

    @staticmethod
    def score_samples(motion_primitive, samples, semantic_pose_constraints, cartesian_constraints, weights=(1,1)):
        quat_splines = motion_primitive.create_multiple_spatial_splines(samples, joints=None)
        #print "orientation of motion sample",quat_splines[-1].coeffs[0][3:7]
        # Evaluate semantic and time constraints
        labels = None#list(set(SemanticConstraint.get_constrained_labels(semantic_pose_constraints)) & set(motion_primitive.time.semantic_labels))
        time_splines = [motion_primitive.create_time_spline(svec, labels) for svec in samples]
        scores = numpy.zeros(len(samples))
        if len(semantic_pose_constraints) > 0:
            #print "set weights to ", semantic_pose_constraints[0].weights #TODO general solution needed for the weights
            scores += score_splines_with_semantic_pose_constraints(quat_splines, time_splines, semantic_pose_constraints, semantic_pose_constraints[0].weights)
        if len(cartesian_constraints) > 0:
            scores += MGRDCartesianConstraint.score_splines(quat_splines, cartesian_constraints)
        return scores

    @staticmethod
    def score_samples_cartesian(motion_primitive, samples, mp_constraints):
        """ Scores splines using only cartesian constraints.

        Args:
            motion_primitive (mgrd.MotionPrimitiveModel): the motion primitive to backproject the samples.
            samples(List<Array<float>>): list of samples generated from the motion primitive.
            mp_constraints (MotionPrimitiveConstraints>):  a set of motion primitive constraints.
            transform (Matrix4F):optional transformation matrix of the samples into the global coordinate system.

        Returns:
            Array<float>
        """
        if has_mgrd:
            cartesian_constraints = MGRDSampleFilter.extract_cartesian_constraints(mp_constraints)
            if len(cartesian_constraints) > 0:
                quat_splines = motion_primitive.create_multiple_spatial_splines(samples, joints=None)
                # transform the splines if the constraints are not in the local coordinate system of the motion primitive
                if not mp_constraints.is_local and mp_constraints.aligning_transform is not None:
                    start = time.time()
                    for qs in quat_splines:
                        MGRDSampleFilter.transform_coeffs(qs, mp_constraints.aligning_transform)
                    print("transformed splines in", time.time()-start, "seconds")
                return MGRDCartesianConstraint.score_splines(quat_splines, cartesian_constraints)
        else:
            print ("Error: MGRD was not correctly initialized")
            return [0]

    @staticmethod
    def score_samples_using_cartesian_constraints(motion_primitive, samples, mgrd_constraints, transform=None):
        """ Scores splines using cartesian constraints only.

        Args:
            motion_primitive (mgrd.MotionPrimitiveModel): the motion primitive to backproject the samples.
            samples(List<Array<float>>): list of samples generated from the motion primitive.
            constraints (List<mgrd.CartesianConstraint>):  a list of cartesian constraints each describing the target position of a joint.
            transform (Matrix4F):optional transformation matrix of the samples into the global coordinate system.

        Returns:
            Array<float>
        """
        if has_mgrd:
            quat_splines = motion_primitive.create_multiple_spatial_splines(samples, joints=None)
            if transform is not None:
                for qs in quat_splines:
                    qs.transform_coeffs(transform)
            scores = MGRDCartesianConstraint.score_splines(quat_splines, mgrd_constraints)
            return scores
        else:
            print ("Error: MGRD was not correctly initialized")
            return [0]
