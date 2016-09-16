#!/usr/bin/python
from scipy.io import loadmat
import random
import numpy
import sys

#TODO: return predicted labels preds
def knn(training_data, training_labels, test_data, test_labels):
	err = 0
	for y in range (0, test_labels.size):
	#for y in range (0, 10):
		min_val = sys.maxint
		min_label = 0
		for x in range (0, training_labels.size):
			diff = numpy.linalg.norm(abs(training_data[x] - test_data[y]))
			if diff < min_val:
				min_val = diff
				min_label = training_labels[x]

		ret = "[O]"
		if min_label != test_labels[y]:
			err += 1
			ret = "[X]"
		#print ret, "pred: ", min_label, "real: ", test_labels[y]
	print err
	return

training_size = 8000
ocr = loadmat('ocr.mat')
#print ocr['data'].shape
sel = random.sample(xrange(60000), training_size)
'''
print sel
print ocr['data'][sel]
print ocr['labels'][sel]
'''
#print "testdata size: ", ocr['testlabels'].size 
knn(ocr['data'][sel], ocr['labels'][sel], ocr['testdata'], ocr['testlabels'])
