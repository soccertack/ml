from timeit import default_timer as timer
from SimClasses import *
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import perceptron
from sklearn.svm import LinearSVC
from sklearn import metrics
import numpy as np

class Classifier:
	def __init__(self, train_X, train_Y):
		self.time = 0
		self.coef = 0

	def Get_Params(self):
		return np.array(self.model.coef_)

	def Get_Compute_Times(self):
		return self.time

	def Classify(self, test_X):
		predicted = self.model.predict(test_X)
		return predicted

