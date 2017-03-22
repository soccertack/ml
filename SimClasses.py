'''
Create a simulator python class SimClasses for data that is drawn from two
D-dimensional circular Gaussians with prescribed distance. Include a method
[X,Y] = GetData(N, D, Distance) that produces N points X (numpy matrix), whose
category assignments are fair coin- flips Y (numpy array), where the Heads
category is drawn from a standard normal distribution, and the Tails category
is drawn from an equal-covariance distribution whose mean is Distance standard
deviations away along the first coordinate in the positive direction.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random as rd

from pylab import *

class SimClasses:
	def GetData(self, N, D, Distance):

		X = np.empty([N, D])
		Y = np.empty([N])
		for i in range(0, N):
			# HEAD (1)
			if rd.random() > 0.5: 
				X[i] = np.random.multivariate_normal(np.zeros(D), np.identity(D))
				Y[i] = 1
			# TAIL (0)
			else:
				mean = np.zeros(D)
				mean[0] = Distance
				X[i] = np.random.multivariate_normal(mean, np.identity(D))
				Y[i] = 2
		return X, Y

