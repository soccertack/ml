from TemporalModel import *
import matplotlib.pyplot as plt

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
