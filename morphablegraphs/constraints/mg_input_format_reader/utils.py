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

CONSTRAINT_TYPES = ["keyframeConstraints", "directionConstraints"]

def _transform_point_from_cad_to_opengl_cs(point, activate_coordinate_transform=False):
    """ Transforms a 3D point represented as a list from a CAD to a
        opengl coordinate system by a -90 degree rotation around the x axis
    """
    if not activate_coordinate_transform:
        return point
    transform_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    return np.dot(transform_matrix, point).tolist()


def _transform_unconstrained_indices_from_cad_to_opengl_cs(indices, activate_coordinate_transform=False):
    """ Transforms a list indicating unconstrained dimensions from cad to opengl
        coordinate system.
    """
    if not activate_coordinate_transform:
        return indices
    new_indices = []
    for i in indices:
        if i == 0:
            new_indices.append(0)
        elif i == 1:
            new_indices.append(2)
        elif i == 2:
            new_indices.append(1)
    return new_indices

