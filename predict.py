import numpy as np
from sklearn.externals import joblib

def predict(test_x):

	# Sample line to get the trained classifier
	clf = joblib.load('filename.pkl') 
	test_y = clf.predict(test_x)
	return test_y 


