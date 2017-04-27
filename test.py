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
alpha[0] = [0.9, 0.1, 0.0]
alpha[1] = [0.4, 0.2, 0.4]
alpha[2] = [0.0, 0.1, 0.9]
print (alpha.shape)
mu = np.empty([K, 2])
mu[0] = [0, 0]
mu[1] = [10, 10]
mu[2] = [0, 10]

sigma = np.empty([K, 2, 2])
sigma[0] = np.identity(2)
sigma[1] = np.identity(2)
sigma[2] = np.identity(2)

t = TemporalModel(alpha, mu, sigma)
Y = t.Simulate(1000)

plt.scatter(Y[:,0],Y[:,1], marker='+', s=1)
plt.show()
