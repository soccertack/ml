import numpy as np
from sklearn.externals import joblib

CLASSIFIER_FILE="Trained_classifier.pkl"

def predict(test_x):

	# Sample line to get the trained classifier
	clf = joblib.load(CLASSIFIER_FILE) 
	test_y = clf.predict(test_x)
	return test_y 


