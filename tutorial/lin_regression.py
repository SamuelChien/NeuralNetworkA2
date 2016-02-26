import os
import pandas
from matplotlib.pyplot import *
from numpy import *
from numpy.linalg import norm

def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()

    while norm(t - prev_t) >  EPS:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)    
    return t


dat = pandas.read_csv("galaxy.data")



x1 = dat.loc[:,"east.west"].as_matrix()
x2 = dat.loc[:, "north.south"].as_matrix()
y = dat.loc[:, "velocity"].as_matrix()


x = vstack((x1, x2))
theta = array([-3, 2, 1])

#Check the gradient
h = 0.000001
# print (f(x, y, theta+array([0, h, 0])) - f(x, y, theta-array([0, h, 0])))/(2*h)
# print df(x, y, theta)



theta0 = array([0, 10, 20.])

#larger learning rate leads to trouble!
theta = grad_descent(f, df, x, y, theta0, 0.0000010)

print x.shape, y.shape, theta.shape
#Exact solution:  dot(dot(linalg.inv(dot(x, x.T)),x), y)
#array([ 1599.7805884 ,     2.32128786,    -3.53935822])
