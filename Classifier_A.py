from Classifier import *

class Classifier_A(Classifier):
	def __init__(self, train_X, train_Y):
		self.model = LogisticRegression()

		start = timer()
		self.model = self.model.fit(train_X, train_Y)
		end = timer()

		self.time = (end - start)
		self.coef = self.model.get_params()

'''
# References
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976
'''

