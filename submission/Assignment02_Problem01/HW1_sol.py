#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pylab import *

class LinearRegressionSimulator:
	def __init__(self, Theta, StdDev):
		# Theta must be a vector, StdDev must be non-negative
		assert len(Theta.shape) == 1
		assert StdDev >= 0.
		self.Theta = Theta
		self.StdDev = StdDev
	def SimData(self, XInput):
		# Augment the dataset by appending a column of ones to the input array
		XAug = np.concatenate( (np.ones( (XInput.shape[0],1) ), XInput), axis=1)
		mus = np.dot( XAug, self.Theta )
		return np.random.normal( mus, self.StdDev )
	def SimPoly(self, XInput):
		# Copy XInput in increasing powers, then outsource augmented data back to SimData.
		powers = np.repeat( np.arange( 1, self.Theta.shape[0] ).reshape(1,-1), XInput.shape[0], axis=0 )
		XAug = np.power( np.repeat( XInput.as_matrix(), powers.shape[1], axis=1 ), powers )
		return self.SimData( XAug )

# Assuming XInput is a single column of values, constructs a d-degree polynomial for each value
def DesignMatrix(XInput, d):
	if d == 0:
		return np.ones( (XInput.shape[0],1) )
	else:
		powers = np.repeat( np.arange( 1, d+1 ).reshape(1,-1), XInput.shape[0], axis=0 )
		powered = np.power( np.repeat( XInput.as_matrix(), powers.shape[1], axis=1 ), powers )
		return np.concatenate( (np.ones( (XInput.shape[0],1) ), powered), axis=1)

# Performs polynomial regression on X by fitting a polynomial of degree d
# Returns ThetaStar, a vector of coefficients, of length d+1 (including the bias)
def FitPoly(XInput, y, d):
	X = DesignMatrix( XInput, d )
	return np.dot( np.linalg.pinv( X ), y )

