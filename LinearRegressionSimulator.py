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
		input_array = XInput.values.astype(float)
	
		# Xinput is a column vector
		num_input = input_array.shape[0]

		# vander_input is (num_input x D+1) where ith rows is input_array.T^i
		vander_input = np.vander(input_array.flatten(), self.theta.size, increasing=True).T
		mean_array = np.dot(self.theta, vander_input)

		# Get normal distribution
		return np.random.normal(mean_array, self.stddev)


