from SimClasses import *
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import perceptron
from sklearn import metrics

class Classifier_B:
	def __init__(self, train_X, train_Y):
		self.train_X = train_X
		self.train_Y = train_Y
		# Need more iterations?
		self.model = perceptron.Perceptron()
		self.model = self.model.fit(train_X, train_Y)
		print("Training accuracy: ", self.model.score(train_X, train_Y))

	def Classify(self, test_X):
		predicted = self.model.predict(test_X)
		test_Y = 1
		return predicted

	
'''
# References
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
http://stamfordresearch.com/scikit-learn-perceptron/
'''
a = SimClasses()
N = 10000	# Data size
D = 2		# Dimension
# Distance starts from 1, to 10, which gives a perfect training and test error
Distance = 2	# stdev for Tail distribution
train_X, train_Y = a.GetData(N, D, Distance)

# Training data has the same setting.
N = 1000	# Data size
test_X, test_Y = a.GetData(N, D, Distance)

class_b = Classifier_B(train_X, train_Y)
predicted_Y = class_b.Classify(test_X)
print (metrics.accuracy_score(test_Y, predicted_Y))

# Plot input data
N = 10000	# Data size
X_x = train_X[:,0]
X_x.shape = (N, 1)
X_y = train_X[:,1]
X_y.shape = (N, 1)

plt.scatter(X_x, X_y, marker='+', s=1)
plt.show()

