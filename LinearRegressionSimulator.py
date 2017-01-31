import numpy as np
import pandas as pd

class LinearRegressionSimulator(object):

	def __init__(self, theta, stddev):
		self.theta = theta
		self.stddev= stddev

	# XInput is pandas Data Frame
	def SimData(self, XInput):

		# Convert Xinput to ndarray
		NDArray = XInput.values.astype(float)

		# Convert NxD array to Nx(D+1) by adding a new column of 1s as a first column
		ND1Array = np.insert(NDArray, 0, 1, axis=1)

		# Get means for each data (i.e. row)
		mean_array = np.dot(ND1Array, self.theta.T)

		# Get normal distribution
		return np.random.normal(mean_array, self.stddev)

	# XInput is a single column pandas vector
	def SimPoly(self, XInput):

		# Convert Xinput to ndarray
		NDArray = XInput.values.astype(float)
	
		num_rows = NDArray.shape[0]

		# Convert Nx1 array to (N+1)x1 by adding 1 as a new first row
		N1Array = np.insert(NDArray, 0, 1, axis=0)

		mult_N1Array = N1Array
		mean_array = N1Array

		for x in range(1, num_rows):
			# Set each element to 1 once it is multipled by d times
			mult_N1Array[x] = 1
			mean_array = np.multiply(mean_array, mult_N1Array)

		print ("mean array is ")

		# Get normal distribution
		return np.random.normal(mean_array, self.stddev)


