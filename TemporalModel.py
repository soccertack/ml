import numpy as np
import random as rd

class TemporalModel:
	def __init__(self, alpha, mu, sigma):
		# TODO: verify params
		self.alpha= alpha
		self.mu = mu
		self.sigma = sigma
		self.K = self.alpha.shape[0]

	def Simulate(self, T):
		
		# q0 = 0
		curr_idx = 0
		Y = np.empty([T, 2])

		# Generate T samples
		for j in range(0, T):
			r = rd.random()
			acc_prob = 0	
			next_idx = self.K

			for i in range(0, self.K):
				# Compare from the first column
				acc_prob += self.alpha[curr_idx][i]
				if r <= acc_prob:
					next_idx = i
					break;

			assert next_idx != self.K, "next_idx is not set"
			curr_idx = next_idx
				
		
			#Generate y_i
			Y[j] = np.random.multivariate_normal(self.mu[curr_idx], self.sigma[curr_idx])

		print (Y)
		return Y
