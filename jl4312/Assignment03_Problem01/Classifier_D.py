from Classifier import *
from sklearn import preprocessing
import random

class Classifier_D(Classifier):
	def __init__(self, train_X, train_Y):

		start = timer()
		train_X = np.insert(train_X, 0, 1, axis = 1)
		w = np.zeros(train_X.shape[1])
		train_size = train_X.shape[0]

		for idxx in range(0, train_size * 100):
			idx = random.randint(0, train_size-1)
			y = train_Y[idx]
			if y == 0:
				y = -1

			if y * np.dot(w, train_X[idx]) <= 0 :
				w = w + y*train_X[idx]
		end = timer()

		self.time = (end - start)
		self.coef_ = w

	def Get_Params(self):
		return self.coef_

	def Classify(self, test_X):
		test_X = np.insert(test_X, 0, 1, axis = 1)
		return (np.sign(np.dot(test_X, self.coef_.T))+1)/2


