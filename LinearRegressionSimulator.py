import numpy as np
import pandas as pd

class LinearRegressionSimulator(object):

	def __init__(self, Theta, std):
		"""
		Inputs:
		Theta - array of coefficients (nonempty 1xD+1 numpy array)
		std   - standard deviation (float)
		"""

		assert len(Theta) != 0

		self.Theta = Theta
		self.std = std

	def SimData(self, XInput):
		"""
		Input:
		XInput - (NxD pandas dataframe)
		Returns: outarray - (N-dim vector)
		"""
		N,D = XInput.shape

		assert D+1 == len(self.Theta)

		self.means = self.Theta[0]+np.matmul(XInput, self.Theta[1:])
		outarray = self.std*np.random.randn(N)+self.means

		return outarray

	# XInput is a single column pandas vector
	def SimPoly(self, XInput):

		# Convert Xinput to ndarray
		input_array = XInput.values.astype(float)
	
		# Xinput is a column vector
		num_input = input_array.shape[0]

		# vander_input is (num_input x D+1) where ith rows is input_array.T^i
		vander_input = np.vander(input_array.flatten(), self.Theta.size, increasing=True).T
		# Make sure that Theta is a column vector 
		Theta = self.Theta
		Theta.shape = (1, self.Theta.size)
		mean_array = np.dot(Theta, vander_input)

		# Get normal distribution
		return np.random.normal(mean_array, self.std)


