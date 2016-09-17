#!/usr/bin/python
from scipy.io import loadmat
import random
import numpy
import sys

#TODO: return predicted labels preds
def knn(training_data, training_labels, test_data, test_labels):
	training_data = training_data.astype('float')
	test_data = test_data.astype('float')
	err = 0

	for y in range (0, test_labels.size):
		min_val = sys.maxint
		min_label = 0
		for x in range (0, training_labels.size):
			diff = numpy.linalg.norm(abs(training_data[x] - test_data[y]))
			if diff < min_val:
				min_val = diff
				min_label = training_labels[x]

		if min_label != test_labels[y]:
			err += 1
	print err
	return

ocr = loadmat('ocr.mat')
#print ocr['data'].shape
training_size = [1000, 2000, 4000, 8000]
for i in range(len(training_size)):
	for j in range(0, 10):
		print j, "th run with size ", training_size[i]
		sel = random.sample(xrange(60000), training_size[i])
		knn(ocr['data'][sel], ocr['labels'][sel], ocr['testdata'], ocr['testlabels'])
