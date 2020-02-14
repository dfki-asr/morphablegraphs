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
Created on Sun Nov 01 20:46:42 2015

@author: Erik Herrmann

http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/
http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html
http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-curve.html
http://devosaurus.blogspot.de/2013/10/exploring-b-splines-in-python.html
https://chi3x10.wordpress.com/2009/10/18/de-boor-algorithm-in-c/

http://demonstrations.wolfram.com/GeneratingABSplineCurveByTheCoxDeBoorAlgorithm/

"""
import numpy as np
import matplotlib.pyplot as plt


class BSpline(object):
    """
    http://demonstrations.wolfram.com/GeneratingABSplineCurveByTheCoxDeBoorAlgorithm/
    http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html
    """
    def __init__(self,  points, degree=3, domain=None):
        self.points = np.array(points)
        if isinstance(points[0], (int, float, complex)):
            self.dimensions = 1
        else:
            self.dimensions = len(points[0])
        self.degree = degree
        if domain is not None:
            self.domain = domain
        else:
            self.domain = (0.0, 1.0)
        self.knots = None
        self.initiated = False
        self._create_knots()

    def _initiate_control_points(self):
        return

    def clear(self):
        return

    def get_last_control_point(self):
        return self.points[-1]

    def _create_knots(self):
        """
        http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-curve.html
        #To change the shape of a B-spline curve, one can modify one or more of 
        #these control parameters: 
        #the positions of control points, the positions of knots, and the degree of the curve.
        # given n+1 control points and m+1 knots the following property must be true
        #m = n + p + 1. // p+1 = m-n
        # for a clamped curve the last knot must be of multiplicity p+1
     
        If you have n+1 control points (n=9) and p = 3. 
        Then, m must be 13 so that the knot vector has 14 knots
        The remaining 14 - (4 + 4) = 6 knots can be anywhere in the domain. 
        U = { 0, 0, 0, 0, 0.14, 0.28, 0.42, 0.57, 0.71, 0.85, 1, 1, 1, 1 }. 
        how do find the knot points C(ui).
        """
        outer_knots = self.degree+1
        print("multiplicity", outer_knots)
        n = len(self.points) - 1 
        print("control points", len(self.points))
        print("n", n)
        p = self.degree
        m = n + p + 1
        n_knots = m + 1
        inner_knots = n_knots-(outer_knots*2 - 2)
        print("knots", n_knots)
        print("free knots", inner_knots)
        print("domain", self.domain)
        #print np.linspace(0.0, 1.0, 4)
        knots = np.linspace(self.domain[0], self.domain[1], inner_knots).tolist()
        #print self.knots
        self.knots = knots[:1] * (outer_knots-1) + knots +\
                    knots[-1:] * (outer_knots-1)
        print(self.knots)
        print(len(self.knots))
        self.initiated = True

    def query_point_by_parameter(self, u):
        """

        """
        return self.evaluate(u, algorithm="deboor")

    def evaluate(self, u, algorithm="standard"):
        #print "evaluate", u
        if self.domain[0] < u < self.domain[1]:
            if algorithm == "standard":
                value = 0.0#np.zeros(self.dim)
                n = len(self.points)
                w_list = []
                for i in range(n):
                    #i+=self.degree
                    #print "iteration",i, self.basis(u, i, self.degree)
                    #i = self.get_begin_of_knot_range(u)
                    w = self.basis(u, i, self.degree)
                    w_list.append(w)
                    #print temp
                    value += w * self.points[i]
                #print sum(w_list)
                return value
            elif algorithm == "deboor":
                i = self.get_begin_of_knot_range(u)
                #print u
                return self.deboor(self.degree, self.degree, u, i)
        elif u >= self.domain[1]:
            return self.points[-1]
        elif u <= self.domain[0]:
            return self.points[0]

    def basis(self, u, i, p):
        """http://devosaurus.blogspot.de/2013/10/exploring-b-splines-in-python.html
        """
        if p == 0:
            if self.knots[i] <= u < self.knots[i+1]:
                return 1.0
            else:
                return 0.0
        elif p >= 1:
            #print i+p
            #print "knot interval", i, i+p, self.knots[i+p]
            out = 0.0
            w_nom = (u-self.knots[i])
            w_denom  = (self.knots[i+p]-self.knots[i])
            if w_denom > 0.0:
                w = w_nom / w_denom
                out += w * self.basis(u, i, p-1)
                
            w_inv_nom = (self.knots[i+p+1] - u)
            w_inv_denom = (self.knots[i+p+1] - self.knots[i+1])
            if w_inv_denom > 0.0:
                w_inv = w_inv_nom / w_inv_denom
                out += w_inv * self.basis(u, i+1, p-1)
            return out
            
    def get_begin_of_knot_range(self, u):
        begin_of_range = 0        
        for i, u_i in enumerate(self.knots):
            if u_i < u:
                begin_of_range = i
            else:
                break
        #print "begin", begin_of_range
        return begin_of_range
                
    def deboor(self, k, p, u, i):
        """
        https://chi3x10.wordpress.com/2009/10/18/de-boor-algorithm-in-c/
        """
        if k == 0:
            return self.points[i]
        elif k >= 1:

            denom = (self.knots[i+p+1-k] - self.knots[i])
            if denom >0:
                alpha = (u-self.knots[i])/denom
                return (1-alpha) * self.deboor(k-1, p, u, i-1) \
                        + (alpha * self.deboor(k-1, p, u, i))
            else:
                return np.zeros(self.dimensions)

    def draw(self, domain, algorithm="standard", granularity=1000):
        print("draw")
        figure = plt.figure()
        ax = figure.add_subplot(111)
        u_values = []
        samples = []
        for u in np.linspace(domain[0], domain[1], granularity):
            u_values.append(u)
            samples.append(self.evaluate(u, algorithm))
            print(u)
        #samples = np.array(samples)
        ax.plot(u_values, samples, alpha=0.3)
        plt.show()


def main():

    knots = [0.0, 0.0, 0.0, 0.0, \
             4, 5, 6,  \
            10, 10, 10, 10]

    points = [0, 0, 0, 6, 0, 0, 0] # 7 control points => 6+3+1=10=m => 11 knots
    degree = 3

    #print "n knots", len(knots)
    #print "n points", len(points)
    #print "degree", degree
    #print "delta", len(knots) - len(points)
    domain = (0.0, 1.0)
    points = np.array([(1,1), (2,2), (4,4), (6,6), (8,8), (10,11), (12,12), (15,15), (16,16), (20,20)])
    bspline = BSpline(points, degree, domain)
    print(bspline.evaluate(0.999))
    algorithm = "standard"
    #algorithm = "deboor"
    bspline.draw(domain, algorithm)
    return
      
if __name__ == "__main__":
    main()
