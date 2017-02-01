import numpy as np
from numpy.linalg import inv
import pandas as pd
from LinearRegressionSimulator import *
import matplotlib
import pylab as pl
import sys

# my UNI is jl4312
sim = LinearRegressionSimulator(np.array([[2, 1, 3, 4]]), 0)

#http://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html

#Generate training and test set
N=100
M=10
training = np.random.uniform(0, 1, N)
test = np.random.uniform(0, 1, M)
training.shape = (N, 1)

#Set degree
degree = 10

x = np.vander(training.flatten(), degree+1, increasing=True)
print (x)

XInput = pd.DataFrame(training)
y = (sim.SimPoly(XInput))
print (y.T)

# Get coefficients
print ("x shape: ", x.shape)
xtx_inv = inv(np.dot(x.T, x))
print ("xtx_inv shape: ", xtx_inv.shape)
xtx_inv_xt = np.dot(xtx_inv, x.T)
print ("xtx_inv_xt shape: ", xtx_inv_xt.shape)
a = np.dot(xtx_inv_xt, y.T)
print (a)

print ("a shape is ", a.shape)
reg_x = np.linspace(0, 1)
print ("reg_x shape is ", reg_x.shape)
print (reg_x)
vand = np.vander(reg_x, degree+1, increasing=True).T
print ("vand shape: ", vand.shape)
print (vand)
reg_y = np.dot(a.T, np.vander(reg_x, degree+1, increasing=True).T)
print ("xx shapte is ", reg_y.shape)
print (reg_y)
pl.plot(reg_x.flatten(), reg_y.flatten(), "b--")

pl.plot(training.flatten(), y.flatten(), 'ro')
pl.show()
#compare output with numpy.polyfit
