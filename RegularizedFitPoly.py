import numpy as np
from numpy.linalg import inv
import pandas as pd
from LinearRegressionSimulator import *
import matplotlib
import pylab as pl
import sys
from matplotlib.backends.backend_pdf import PdfPages

'''
1. Get sample from [-1, 1], not [0, 1]
2. Use Legendre instead of simple power of x
3. Add lambda for the regualized and set it to 0. We can just reuse FitCubic this case
4. Pick lamda 2 to 5.
'''

def RegularizedFitPoly():
	# my UNI is jl4312
	sim = LinearRegressionSimulator(np.array([2, 1, 3, 4]), 0.1)

	# Given condition in the problem
	degree = 6
	N = 10
	M = 100

	# Generate training data
	training = np.random.uniform(-1, 1, N)
	training.shape = (N, 1)
	XInput = pd.DataFrame(training)
	training_y = (sim.SimPoly(XInput))
	training_y.shape = (N, 1)

	# Generate test data
	test = np.random.uniform(-1, 1, M)
	test.shape = (M, 1)
	XInput = pd.DataFrame(test)
	test_y = (sim.SimPoly(XInput))
	test_y.shape = (M, 1)

	# Save x and y for train and test to pickel file (problem1.pkl)

	risk_train = []
	risk_test = []

	# TODO: X is from legendre poly
	x = np.vander(training.flatten(), degree+1, increasing=True)
	
	# Five lambda values
	ld = [0, 0, 0, 0, 0]


	# TODO: We need for-loop iterating ld[]
	for idx, ld_val in enumerate(ld)
	if ld_val  == 0:
		# Use given solution for the linear regression
	else:
		# (X.T*X+lambda*N*I) is always invertible if lambda is not zero
		xtx_inv = inv(np.dot(x.T, x)+ ld_val*N*np.identity(degree))

		# thetaStar = (XTX)-1XTy (Nx1 matrix)
		xtx_inv_xt = np.dot(xtx_inv, x.T)
		thetaStar = np.dot(xtx_inv_xt, training_y)

	# TODO: save thetaStar to pickle file
	
	# Get training risk
	diff  = training_y - np.dot(x, a)
	training_R = np.linalg.norm(diff)**2 / (2*N)
	risk_train.append(training_R)

	# Get test risk
	# TODO: legendre
	test_x = np.vander(test.flatten(), degree+1, increasing=True)
	diff = test_y -np.dot(test_x, a)
	test_R = np.linalg.norm(diff)**2 / (2*M)
	risk_test.append(test_R)

	np.savetxt("Risk.train."+str(run_index)+'.txt', np.array(risk_train))
	np.savetxt("Risk.test."+str(run_index)+'.txt', np.array(risk_test))

	# Draw a plot
	r_plot = PdfPages('RiskPlot'+str(run_index)+'.pdf')
	deg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	pl.plot(deg, risk_train, "g--", label="Train Risk")
	pl.plot(deg, risk_test, "r--", label="Test Risk")
	pl.legend()
	pl.xlabel('Dimension')
	pl.xticks(np.linspace(0,10,11,endpoint=True))
	r_plot.savefig()
	pl.close()
	r_plot.close()

