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

	# prob: a 1xK array, sum(prob) should be 1
	# return state [0, K-1]
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

	# return T-long array of states [0,K-1]
	def SampleStates(self, Y):
		
		T = Y.shape[0]
		states = np.empty([T])
		states = states.astype(int)

		# We know that the first state is 0
		states[0] = 0
		curr_state = 0
		
		for j in range(1, T):
			# calc probability of jth state based on jth input and j-1th state
			posterior = self.Posterior(Y[j], self.alpha[states[j-1]])
			states[j] = self.Sample_state(posterior)

		return states

	def SampleGibbsLike(self, Y):

		#TODO: this should be loop

		# At least, we know K.
		K = self.K

		#TODO: temporarily assume K = 3, and hardcode all model param
		K = 3
		alpha = np.empty([K, K])
		alpha[0] = [0.2, 0.2, 0.6]
		alpha[1] = [0.2, 0.2, 0.6]
		alpha[2] = [0.2, 0.2, 0.6]

		# Set mu and sigma the same. Check we get better alpha
		Distance = 10
		mu = np.empty([K, 2])
		mu[0] = [0, 0]
		mu[1] = [Distance, Distance]
		mu[2] = [0, Distance]

		sigma = np.empty([K, 2, 2])
		sigma[0] = np.identity(2)
		sigma[1] = np.identity(2)
		sigma[2] = np.identity(2)

		t = TemporalModel(alpha, mu, sigma)
		# We are sampling using true alpha, not the prior we would get
		sampled_states = self.SampleStates(Y)
		print ("From SampleGibbsLike")
		print (sampled_states)

		# Tally pseudocounts
		prior = np.zeros([self.K])
		avg = np.zeros([self.K, 2])
		sigma = np.zeros([self.K])

		new_alpha = np.zeros([K, K])

		T = Y.shape[0]
		for j in range(0, T):
			s_state = sampled_states[j]
			prior[s_state] += 1
			avg[s_state] += Y[j]

			if j == 0:
				continue
			
			p_state = sampled_states[j-1]
			if p_state == 0:
				print (s_state)
			new_alpha[p_state][s_state] +=1

		print (new_alpha)
		prior = prior/T
		avg = avg/T

		# TODO: covariance matrix 
		# use np.cov



		
