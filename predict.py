##############################################
# COMS 4771 Machine Learning Midterm
# Name: Jin Tack Lim
# UNI: jl4312
##############################################

# import all necessary modules here
# see midterm instruction for requirements

import numpy as np
from sklearn.externals import joblib

CLASSIFIER_FILE="Trained_classifier.pkl"

def predict(X_test):
	"""This function takes a dataset and predict the class label for each data point
	should be in.

	Parameters
	----------
	dataset: M X D numpy array
		A dataset represented by numpy-array

	Returns
	-------
	M x 1 numpy array
		Returns a numpy array of predicted class labels
	"""

	clf = joblib.load(CLASSIFIER_FILE) 
	test_y = clf.predict(X_test)
	return test_y 

