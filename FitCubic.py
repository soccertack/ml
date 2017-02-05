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

def dump_data_each_degree(prefix, run_index, degree_index, header, nparray):
	return
	f = open(prefix+str(run_index)+'.'+str(degree_index)+'.txt', 'w')
	f.write(output_header)
	if nparray != None:
		f.write(np.savetxt(nparray))
	f.close()

def dump_data(prefix, header, nparray):
	f = open(prefix+'.txt', 'w')
	f.write(output_header)
	f.close()
	if nparray != None:
		np.savetxt(prefix+'.txt', nparray)

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

	# Prepare pdf
	pp = PdfPages('DataPlot'+str(run_index)+'.pdf')


	'''
	output_header = "Run #"+str(run_index)+" (N=" + str(N) + ", M=" + str(M) + ")\n"
	output_header += "Iteration 1\n"
	'''
	np.savetxt('x.train.'+str(run_index)+'.txt', training)
	np.savetxt('x.test.'+str(run_index)+'.txt', test)
	np.savetxt('y.train.'+str(run_index)+'.txt', training_y)
	np.savetxt('y.test.'+str(run_index)+'.txt', test_y)

	prev_R = 0

	risk_train = []
	risk_test = []
	for degree in range(0, max_degree+1):
		x = np.vander(training.flatten(), degree+1, increasing=True)

		# Check if the XTX is invertible.
		# http://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
		if np.linalg.cond(np.dot(x.T, x)) < 1/sys.float_info.epsilon:
			xtx_inv = inv(np.dot(x.T, x))
			print("Invertible")
		else:
			xtx_inv = np.linalg.pinv(np.dot(x.T, x))
			print("NOT Invertible")
		# a = (XTX)-1XTy (Nx1 matrix)
		xtx_inv_xt = np.dot(xtx_inv, x.T)
		a = np.dot(xtx_inv_xt, training_y)

		np.savetxt('ThetaStar.'+str(run_index)+"."+str(degree)+'.txt', a)
		print ("degree: ", degree)
		
		diff  = training_y - np.dot(x, a)
		training_R = np.linalg.norm(diff)**2 / (2*N)
		print ("Training R: ", training_R)
		risk_train.append(training_R)

		#print (diff)
		'''
		if prev_R!=0 and (R > prev_R):
			print ("========== R is BIG ===============")
		prev_R = R
		'''
			
		test_x = np.vander(test.flatten(), degree+1, increasing=True)
		diff = test_y -np.dot(test_x, a)
		test_R = np.linalg.norm(diff)**2 / (2*M)
		print ("Test R: ", test_R)
		risk_test.append(test_R)

		# Plot graph
		reg_x = np.linspace(0, 1)
		vand = np.vander(reg_x, degree+1, increasing=True).T
		reg_y = np.dot(a.T, np.vander(reg_x, degree+1, increasing=True).T)
		pl.plot(reg_x.flatten(), reg_y.flatten(), "g--")

		# Plot Training data
		pl.plot(training.flatten(), training_y.flatten(), 'r.')
		# Plot Test data
		pl.plot(test.flatten(), test_y.flatten(), 'b.')

		'''
		# This is for the verification of coefficient
		z = np.polyfit(training.flatten(), training_y.flatten(), degree)
		p = np.poly1d(z)
		pl.plot(reg_x.flatten(), p(reg_x.flatten()), 'r-')
		'''

		#pl.show()
		pp.savefig()
		pl.close()

	risk_train_np = np.array(risk_train)
	risk_train_np.shape = (max_degree+1, 1)
	risk_test_np = np.array(risk_test)
	risk_test_np.shape = (max_degree+1, 1)
	np.savetxt("Risk.train."+str(run_index)+'.txt', risk_train_np)
	np.savetxt("Risk.test."+str(run_index)+'.txt', risk_test_np)
	pp.close()

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
