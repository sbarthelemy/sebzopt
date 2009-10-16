# coding=utf-8
"""Cost/objective functions for convex optimization problems
"""
__author__ = ("Sébastien BARTHÉLEMY <sebastien.barthelemy@crans.org>")

from abc import ABCMeta, abstractmethod, abstractproperty
from numpy import *


class Objective(object):
    """An objective function, from R^ndim to R."""
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def value(self, x):
        """Return the value of the objective funtion at x."""
        pass
        
    @abstractmethod
    def gradient(self, x):
        """Return the gradient of the objective funtion at x."""
        pass
        
    #@abstractmethod
    #def hessian(self, x):
    #    """Return the hessian of the objective funtion at x.""""
    #    pass

    @abstractproperty
    def ndim(self, x):
        """Number of dimension of the input space of the objective function."""
        pass


    def contour(self, x, y):
        """Contour plot of the function at points x, y (only defined if ndim==2)."""
        import matplotlib.pyplot as plt
        from pylab import show
        if self.ndim != 2:
            raise ValueError()
        def val(x, y):
            return self.value(array([x,y]))
        vvalue = vectorize(val)
        X, Y = meshgrid(x, y)
        Z = vvalue(X, Y)       
        fig = plt.figure()
        CS = plt.contour(X, Y, Z)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title('Contour plot')
        return fig

                 
class QuadraticFunction(Objective):
    def __init__(self, P, q=None, r=None):
        self._ndim = P.shape[0]
        self._P = P
        self._q = q
        self._r = r

    @property
    def ndim(self):
        return self._ndim

    def value(self, x):
        return dot(x, dot(self._P, x))/2. + dot(self._q,x) + self._r
        
    def gradient(self, x):
        return dot(x, self._P) + self._q
        
    def hessian(self, x):
        return self.P
