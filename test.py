# coding=utf-8
__author__ = ("Sébastien BARTHÉLEMY <sebastien.barthelemy@crans.org>")

from sebzopt import *
from pylab import *

def myplot(f,log):
    xmin, ymin = log['x'].min(0)
    xmax, ymax = log['x'].max(0)
    n = 100
    x, y = linspace(xmin, xmax, n), linspace(ymin, ymax, n)
    fig = f.contour(x, y)
    fig.axes[0].plot(log['x'][:,0], log['x'][:,1],'.-')
    #plot(log['x'][:,0], log['x'][:,1])    
    axis('equal')
    show()
    
f = QuadraticFunction(P=diag([10.,2]), 
                      q = array([0.,0]), 
                      r = array(0))

solv = GradientDescent(f)
x = array([ 10.,   30.])
log = solv.run(x)
myplot(f, log)
