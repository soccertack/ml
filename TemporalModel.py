import numpy as np
import random as rd
import scipy.stats

class TemporalModel:
	def __init__(self, alpha, mu, sigma):
		# TODO: verify params
		# TODO: Check the sum of each row of alpha is 1
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

			if j < 3:
				print ("%dth state: %d" % (j, curr_idx))
			#Generate y_j
			Y[j] = np.random.multivariate_normal(
				self.mu[curr_idx], self.sigma[curr_idx])

			#Get next state
			r = rd.random()
			acc_prob = 0	
			next_idx = self.K

			for i in range(0, self.K):
				# Compare from the first column
				acc_prob += self.alpha[curr_idx][i]
				if r <= acc_prob:
					next_idx = i
					#print ("next_idx: %d" % next_idx)
					break;

			assert next_idx != self.K, "next_idx is not set"
		
			curr_idx = next_idx

		#print (Y)
		return Y

	def Probability_of(self, y):

		prob = np.empty([self.K])
		for j in range(0, self.K):
			prob[j] = scipy.stats.multivariate_normal(
					self.mu[j], self.sigma[j]).pdf(y)
		print ("prob: ", prob)
		return prob


	def Posterior(self, y, prior):

		prob = self.Probability_of(y)
		posterior = np.multiply(prob, prior)
		norm = np.sum(posterior)
		posterior /= norm
		print ("post: ", posterior)
		return posterior











