from Classifier import *

class Classifier_B(Classifier):
	def __init__(self, train_X, train_Y):
		self.train_X = train_X
		self.train_Y = train_Y
		# Need more iterations?
		self.model = perceptron.Perceptron()
		start = timer()
		self.model = self.model.fit(train_X, train_Y)
		end = timer()
		self.time = (end - start)
		self.coef = self.model.get_params()

	def Classify(self, test_X):
		predicted = self.model.predict(test_X)
		return predicted

	
'''
# References
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
http://stamfordresearch.com/scikit-learn-perceptron/
'''
