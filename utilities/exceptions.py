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
"""
Created on Tue Jul 14 18:05:23 2015

@author: Erik Herrmann
"""


class SynthesisError(Exception):

    def __init__(self,  quat_frames, bad_samples):
        message = "Could not process input file"
        super(SynthesisError, self).__init__(message)
        self.bad_samples = bad_samples
        self.quat_frames = quat_frames


class PathSearchError(Exception):

    def __init__(self, parameters):
        self.search_parameters = parameters
        message = "Error in the navigation goal generation"
        super(PathSearchError, self).__init__(message)


class ConstraintError(Exception):

    def __init__(self,  bad_samples):
        message = "Could not reach constraint"
        super(ConstraintError, self).__init__(message)
        self.bad_samples = bad_samples
