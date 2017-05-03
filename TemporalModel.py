import numpy as np
import random as rd
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib

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
		if norm != 0:
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

	def GibbsInit(self, Y):
		# At least, we know K.
		K = self.K
		T = Y.shape[0]

		alpha = np.empty([K, K])
		mu = np.empty([K, 2])
		sigma = np.empty([K, 2, 2])

		x_max = float("-inf") 
		x_min = float("inf")
		y_max = float("-inf") 
		y_min = float("inf")
		for j in range(0, K):
			alpha[j] = np.random.dirichlet(np.ones(K),size=1)
			sigma[j] = np.identity(2)

		for j in range(0, T):
			if (Y[j][0] > x_max):
				x_max = Y[j][0]
			if (Y[j][0] < x_min):
				x_min = Y[j][0]

			if (Y[j][1] > y_max):
				y_max = Y[j][1]
			if (Y[j][1] < y_min):
				y_min = Y[j][1]

		print (x_min, x_max)
		print (y_min, y_max)

		for j in range(0, K):
			x = np.random.uniform(x_min.astype(int), x_max.astype(int), 1)
			y = np.random.uniform(y_min.astype(int), y_max.astype(int), 1)
			print (x, y)
			mu[j] = np.array([x, y]).reshape(1,2)
		print("init alpha")
		print(alpha.astype(float))

		# Set mu and sigma the same. Check we get better alpha
		print ("init mu")
		print(mu.astype(float))

		return alpha, mu, sigma

	def SampleGibbsLike(self, Y):

		K = self.K
		T = Y.shape[0]
		alpha, mu, sigma = self.GibbsInit(Y)
		max_iterations = 100

		alpha_list = []
		mu_list = []
		sigma_list = []
		sampled_states_list = []

		
		for j in range(0, max_iterations):

			# We are sampling using true alpha, not the prior we would get

			alpha_list.append(alpha)
			mu_list.append(mu)
			sigma_list.append(sigma)

			t = TemporalModel(alpha, mu, sigma)
			sampled_states = t.SampleStates(Y)
			sampled_states_list.append(sampled_states)

			if j!=0:
				diff = np.count_nonzero(sampled_states != prev_states)
				print ("diff: %d" % diff)
				if diff < T*0.05:
					break;

			mu_class=[1, 2, 3]
			colors = ['red','green','blue','purple']

			plt.scatter(Y[:,0],Y[:,1], c=sampled_states,cmap=matplotlib.colors.ListedColormap(colors),marker='+', s=10)
			plt.scatter(mu[:,0],mu[:,1], c=mu_class,cmap=matplotlib.colors.ListedColormap(colors), s = 50)

			plt.title('%d sampled states give mu' %j)

			plt.axis([-2, 30, -2, 30])
			#plt.show()
			plt.close()

			# TODO: Don't touch sigma yet
			#sigma = np.zeros([self.K])

			new_alpha = np.zeros([K, K])
			new_mu = np.zeros([K, 2])

			for l in range(0, T):
				s_state = sampled_states[l]
				#print ("%dth class %d, " % (j, s_state), Y[j])
				new_mu[s_state] += Y[l]

				# Don't sample state trasition if this is the first state
				if l == 0:
					continue
				
				p_state = sampled_states[l-1]
				new_alpha[p_state][s_state] +=1

			#print ("new mu before norm")
			#print (new_mu.astype(float))

			l1_norm = np.linalg.norm(new_alpha, axis=1, ord=1)
			alpha = new_alpha/l1_norm.reshape(K,1)
			mu = new_mu/l1_norm.reshape(K,1)
			#print ("new normalized alpha")
			#print (alpha.astype(float))
			#print ("new mu")
			#print (mu.astype(float))

			plt.scatter(Y[:,0],Y[:,1], c=sampled_states,cmap=matplotlib.colors.ListedColormap(colors),marker='+', s=1)
			plt.scatter(mu[:,0],mu[:,1], c=mu_class,cmap=matplotlib.colors.ListedColormap(colors), s = 100)

			plt.title('%d New mu given data' % j)

			plt.axis([-2, 30, -2, 30])
			plt.show()
			plt.close()

			prev_states = sampled_states
			# TODO: covariance matrix 
			# use np.cov
	
		print ("Gibbs sampling took %d iterations. Diff: %d" % (j, diff))
		print ("final alpha")
		print (alpha)
		print ("final mu")
		print (mu)
		return 	alpha_list, mu_list, sigma_list, sampled_states_list, j
