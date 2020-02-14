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
"""
Created on Wed Mar 10 17:15:22 2015

Wrapper around scipy minimize and error function definition.

@author: Erik Herrmann, Han Du
"""
from .optimizer_base import OptimizerBase
import time
from scipy.optimize import minimize


class NumericalMinimizer(OptimizerBase):
    """ A wrapper class for Scipy minimize module that implements different gradient descent and
        derivative free optimization methods.
        Please see the official documentation of that module for the supported optimization methods:
        http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.minimize.html
    """
    def run(self, initial_guess):
        """ Runs the optimization for a single motion primitive and a list of constraints
        Returns
        -------
        * x : np.ndarray
              optimal low dimensional motion parameter vector
        """
        if self._objective_function is not None and initial_guess is not None:
            if self.verbose:
                start = time.clock()
                print("Start optimization using", self.optimization_settings["method"],
                      self.optimization_settings["max_iterations"],
                      self.optimization_settings["diff_eps"])
        #    jac = error_function_jac(s0, data)
            try:
                result = minimize(self._objective_function,
                                  initial_guess,
                                  args=(self._error_func_params,),
                                  method=self.optimization_settings["method"],
                                  jac=self._jacobian,
                                  tol=self.optimization_settings["tolerance"],
                                  options={'maxiter': self.optimization_settings["max_iterations"],
                                           'disp': self.verbose,
                                           'eps': self.optimization_settings["diff_eps"]})


            except ValueError as e:
                print("Warning:", e.args)
                return initial_guess

            if self.verbose:
                print("Finished optimization in ", time.clock()-start, "seconds")
            return result.x
        else:
            print("Error: No objective function set. Return initial guess instead.")
            return initial_guess
