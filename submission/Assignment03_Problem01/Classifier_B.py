from Classifier import *

class Classifier_B(Classifier):
	def __init__(self, train_X, train_Y):
		self.model = perceptron.Perceptron()

		start = timer()
		self.model = self.model.fit(train_X, train_Y)
		end = timer()

		self.time = (end - start)

	
'''
# References
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
http://stamfordresearch.com/scikit-learn-perceptron/
'''
