import numpy as np
from numpy.linalg import inv
import pandas as pd
from LinearRegressionSimulator import *


# my UNI is jl4312
sim = LinearRegressionSimulator(np.array([[2, 1, 3, 4]]), 0)

#http://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html

#Generate training and test set
N=3
M=10
training = np.random.uniform(0, 1, N)
test = np.random.uniform(0, 1, M)
training.shape = (N, 1)

#Set degree
degree = 1

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


#compare output with numpy.polyfit
