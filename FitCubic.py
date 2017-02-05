import numpy as np
from numpy.linalg import inv
import pandas as pd
from LinearRegressionSimulator import *
import matplotlib
import pylab as pl
import sys
from matplotlib.backends.backend_pdf import PdfPages

def FitCubic():
# my UNI is jl4312
	sim = LinearRegressionSimulator(np.array([[2, 1, 3, 4]]), 0.1)
	runs = [(10, 10), (100, 10), (10, 100)]
	max_degree = 10

	for index, (N,M) in enumerate(runs):
		# Generate training data
		training = np.random.uniform(0, 1, N)
		training.shape = (N, 1)
		XInput = pd.DataFrame(training)
		training_y = (sim.SimPoly(XInput))
		training_y.shape = (N, 1)

		# Generate test data
		test = np.random.uniform(0, 1, M)
		test.shape = (M, 1)
		XInput = pd.DataFrame(test)
		test_y = (sim.SimPoly(XInput))
		test_y.shape = (M, 1)

		# run_index starts from 1
		run_index = index+1

		# Save x and y
		np.savetxt('x.train.'+str(run_index)+'.txt', training)
		np.savetxt('x.test.'+str(run_index)+'.txt', test)
		np.savetxt('y.train.'+str(run_index)+'.txt', training_y)
		np.savetxt('y.test.'+str(run_index)+'.txt', test_y)

		risk_train = []
		risk_test = []
		for degree in range(0, max_degree+1):
			x = np.vander(training.flatten(), degree+1, increasing=True)

			# Check if the XTX is invertible.
			# http://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
			if np.linalg.cond(np.dot(x.T, x)) < 1/sys.float_info.epsilon:
				xtx_inv = inv(np.dot(x.T, x))
			else:
				xtx_inv = np.linalg.pinv(np.dot(x.T, x))
			# a = (XTX)-1XTy (Nx1 matrix)
			xtx_inv_xt = np.dot(xtx_inv, x.T)
			a = np.dot(xtx_inv_xt, training_y)

			np.savetxt('ThetaStar.'+str(run_index)+"."+str(degree)+'.txt', a)
			
			# Get training risk
			diff  = training_y - np.dot(x, a)
			training_R = np.linalg.norm(diff)**2 / (2*N)
			risk_train.append(training_R)

			# Get test risk
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

