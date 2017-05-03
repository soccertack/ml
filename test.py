from TemporalModel import *
import matplotlib.pyplot as plt

'''
# The most simple case
# K = 1 (one class)
K = 1
alpha = np.ones([K,K])
print (alpha.shape)
mu = np.zeros(2)
mu = np.expand_dims(mu, axis=0)
sigma = np.identity(2)
sigma = np.expand_dims(sigma, axis=0)
print (sigma.shape)

t = TemporalModel(alpha, mu, sigma)
Y = t.Simulate(1000)

plt.scatter(Y[:,0],Y[:,1], marker='+', s=1)
plt.show()
'''


# More realistic case 
K = 3
alpha = np.empty([K, K])
alpha[0] = [0.5, 0.3, 0.2]
alpha[1] = [0.4, 0.2, 0.4]
alpha[2] = [0.0, 0.1, 0.9]
print (alpha.shape)

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
Y, states = t.Simulate_states(10000)

print("0th y :", Y[0])
print("1th y :", Y[1])
print("2th y :", Y[2])
print("0th y state :", states[0])
print("1th y state :", states[1])
print("2th y state :", states[2])

plt.scatter(Y[:,0],Y[:,1], marker='+', s=1)
#plt.show()

'''
t.Probability_of(Y[0])
t.Probability_of(Y[1])
t.Probability_of(Y[2])
'''

prior = np.array([1/3, 1/3, 1/3])
t.Posterior(Y[0], prior)
t.Posterior(Y[1], prior)
t.Posterior(Y[2], prior)

sampled_states = t.SampleStates(Y)

print (states[0:20])
print (sampled_states[0:20])
print (np.array_equal(states, sampled_states))

print ("Gibbs")
t.SampleGibbsLike(Y)
