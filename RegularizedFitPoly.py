import numpy as np
from numpy.linalg import inv
import pandas as pd
from HW1_sol import *
import matplotlib
import pylab as pl
import sys
from matplotlib.backends.backend_pdf import PdfPages

# Generate K random inputs and K correspoding outputs from SimPoly function
def gen_data(sim, K):
	data_x = np.random.uniform(-1, 1, K)
	data_x.shape = (K, 1)
	XInput = pd.DataFrame(data_x)
	data_y = (sim.SimPoly(XInput))
	data_y.shape = (K, 1)
	
	return data_x, data_y

# Computes the empirical risk of the given params Theta on the dataset
def EmpiricalRisk(x, y, Theta, N):
        return ( np.linalg.norm( y - np.dot( x, Theta ) ) ** 2 ) / (2. * N)

def Plot_Original():
	# my UNI is jl4312
	sim_orig = LinearRegressionSimulator(np.array([2, 1, 3, 4]), 0)
	orig_x = np.linspace(-1, 1)
	XInput = pd.DataFrame(orig_x)
	orig_y = (sim_orig.SimPoly(XInput))
	pl.plot(orig_x, orig_y, label="original", color="black")

def RegularizedFitPoly():
	# my UNI is jl4312
	sim = LinearRegressionSimulator(np.array([2, 1, 3, 4]), 0.1)

	# Given condition in the problem
	degree = 6
	N = 10
	M = 100

	training, training_y = gen_data(sim, N)
	test, test_y = gen_data(sim, M)
	# TODO: Save x and y for train and test to pickel file (problem1.pkl)

	# x: Nx7 array
	# rows represent each input
	# columns represent Legendre polinomial degree of i where i: [0, 6]
	x = np.polynomial.legendre.legvander(training.flatten(), degree)

	# Five lambda values
	lambda_array = [0, 0.01, 0.1, 1, 10]
	
	# Plot original and training data.
	Plot_Original()
	pl.plot(training, training_y, "ro",  label="Training data")
	# Prepare for regression plots
	cmap = pl.get_cmap('jet')
	colors = cmap(np.linspace(0, 1, len(lambda_array)))

	# For each lamba, we have traning risk and test risk
	risk_train = np.full( len(lambda_array) , fill_value=np.nan )
	risk_test = np.full( len(lambda_array) , fill_value=np.nan )

	# loop over different lambda values
	for idx, lambda_val in enumerate(lambda_array):
		if lambda_val == 0:
			thetaStar = np.dot( np.linalg.pinv( x ), training_y )
		else:
			# (X.T*X+lambda*N*I) is always invertible if lambda is not zero
			xtx_inv = inv(np.dot(x.T, x)+ lambda_val*N*np.identity(degree+1))

			# thetaStar = (XTX)-1XTy (Nx1 matrix)
			xtx_inv_xt = np.dot(xtx_inv, x.T)
			thetaStar = np.dot(xtx_inv_xt, training_y)

		# TODO: save thetaStar to pickle file
		
		# Get training risk
		risk_test[idx] = EmpiricalRisk(x, training_y, thetaStar, N)
		# Get test risk
		test_x = np.polynomial.legendre.legvander(test.flatten(), degree)
		risk_test[idx] = EmpiricalRisk(test_x, test_y, thetaStar, M)

		# Plot the regression with lambda
		reg_x = np.linspace(-1, 1)
		reg_x_x = np.polynomial.legendre.legvander(reg_x.flatten(), degree)
		pl.plot(reg_x, np.dot( reg_x_x, thetaStar), \
			label="lambda: "+str(lambda_val),color=colors[idx])
		x1,x2,y1,y2 = pl.axis()
		# Max y value of true data is 10 (4+3+1+2)
		pl.axis((x1,x2,0,15))

	pl.legend()
	pl.show()
	# TODO: save risk 

RegularizedFitPoly()
