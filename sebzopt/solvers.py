# coding=utf-8
"""Convex optimization problems solvers
"""

__author__ = ("Sébastien BARTHÉLEMY <sebastien.barthelemy@crans.org>")

from objectives import Objective
from numpy import zeros
from numpy.linalg import norm

class GradientDescent(object):
    
    def __init__(self, obj):
        assert isinstance(obj, Objective)
        self.f = obj
        self.eps = 1e-6

    def run(self, x):
        kmax = 1e2
        # an array for the results
        dtype = 'u4, {n}f8, {n}f8, f8'.format(n=self.f.ndim) 
        data = zeros(kmax, dtype=dtype)
        data.dtype.names = ('k', 'x', 'r', 't')

        k = 0
        g = self.f.gradient(x)
        while norm(g) > self.eps and k < kmax:
            
            r = -g
            t = .15
            data[k]['k'] = k
            data[k]['x'][:]= x            
            data[k]['r'][:] = r
            data[k]['t'] = t
            x += r*t
            k += 1
            g = self.f.gradient(x)
        return data[0:k]

