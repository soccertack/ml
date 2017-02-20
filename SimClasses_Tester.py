from SimClasses import *

a = SimClasses()
N = 10000	# Data size
D = 2		# Dimension
Distance = 10	# stdev for Tail distribution
X, Y = a.GetData(N, D, Distance)

# Print input data
print (X)
print (X[:,0])
print (X[:,1])

# Plot Gaussian pdf (X with Head Y)
'''
X = np.sort(X, axis=None)
print (X)
Xmean = np.mean(X)
Xstd = np.std(X)
pdf = stats.norm.pdf(X, Xmean, Xstd)
plt.plot(X, pdf)
'''

# Plot input data
X_x = X[:,0]
X_x.shape = (N, 1)
X_y = X[:,1]
X_y.shape = (N, 1)

plt.scatter(X_x, X_y, marker='+', s=1)
plt.show()

