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
		Y, states = self.Simulate_states(T)
		return Y

	# prob: a 1xK array
	def Sample_state(self, prob):
		r = rd.random()
		acc_prob = 0	
		state_sampled = self.K

		for i in range(0, self.K):
			# Compare from the first element
			acc_prob += prob[i]
			if r <= acc_prob:
				state_sampled = i
				break;

		assert state_sampled != self.K, "state_sampled is not set"
		return state_sampled

	def Simulate_states(self, T):
		
		# q0 = 0
		curr_idx = 0
		Y = np.empty([T, 2])
		states = np.empty(T, dtype=np.int)

		# Generate T samples
		for j in range(0, T):

			#Generate y_j
			Y[j] = np.random.multivariate_normal(
				self.mu[curr_idx], self.sigma[curr_idx])
			states[j] = curr_idx

			# Update the next state
			curr_idx = self.Sample_state(self.alpha[curr_idx])

		#print (Y)
		return Y, states

	def Probability_of(self, y):

		prob = np.empty([self.K])
		for j in range(0, self.K):
			prob[j] = scipy.stats.multivariate_normal(
					self.mu[j], self.sigma[j]).pdf(y)
		#print ("prob: ", prob)
		return prob


	def Posterior(self, y, prior):

		prob = self.Probability_of(y)
		posterior = np.multiply(prob, prior)
		norm = np.sum(posterior)
		posterior /= norm
		#print ("post: ", posterior)
		return posterior

	# return T-long array of states
	def SampleStates(self, Y):
		
		T = Y.shape[0]
		states = np.empty([T])
		states = states.astype(int)

		# We know that the first state is 0
		states[0] = 0
		curr_state = 0
		
		# TODO: how to determine prior?
		prior = np.array([1/3, 1/3, 1/3])

		for j in range(1, T):
			# calc probability of jth state based on jth input and j-1th state
			mul = np.multiply(self.Probability_of(Y[j]), self.alpha[states[j-1]])
			# Normalize
			norm = np.sum(mul)
			mul/= norm

			states[j] = self.Sample_state(mul)

		return states

