from timeit import default_timer as timer
from SimClasses import *
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import perceptron
from sklearn import metrics

class Classifier:
	def __init__(self, train_X, train_Y):
		self.time = 0
		self.coef = 0

	def Get_Params(self):
		return self.coef

	def Get_Compute_Times(self):
		return self.time

