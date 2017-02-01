import numpy as np
from numpy.linalg import inv
import pandas as pd
from LinearRegressionSimulator import *
import matplotlib
import pylab as pl
import sys
from matplotlib.backends.backend_pdf import PdfPages

# my UNI is jl4312
sim = LinearRegressionSimulator(np.array([[2, 1, 3, 4]]), 0.1)

#http://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html

runs = [(10, 10), (100, 10), (10, 100)]
max_degree = 10
for (M,N) in runs:
	# Generate training data
	training = np.random.uniform(0, 1, N)
	training.shape = (N, 1)
	XInput = pd.DataFrame(training)
	training_y = (sim.SimPoly(XInput))

	# Generate test data
	test = np.random.uniform(0, 1, M)
	test.shape = (N, 1)
	XInput = pd.DataFrame(test)
	test_y = (sim.SimPoly(XInput))

	for degree in range(0, max_degree+1):
		x = np.vander(training.flatten(), degree+1, increasing=True)

		# a = (XTX)-1XTy
		xtx_inv = inv(np.dot(x.T, x))
		xtx_inv_xt = np.dot(xtx_inv, x.T)
		a = np.dot(xtx_inv_xt, training_y.T)

		# Plot graph
		reg_x = np.linspace(0, 1)
		vand = np.vander(reg_x, degree+1, increasing=True).T
		reg_y = np.dot(a.T, np.vander(reg_x, degree+1, increasing=True).T)
		pl.plot(reg_x.flatten(), reg_y.flatten(), "g--")

		# Plot Training data
		pl.plot(training.flatten(), training_y.flatten(), 'r.')
		# Plot Test data
		pl.plot(test.flatten(), test_y.flatten(), 'b.')

		pl.show()

	# TODO: remove
	sys.exit()
