#!/usr/bin/python
from scipy.io import loadmat
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm 
import time

def knn(training_data, training_labels, test_data, test_labels):
	training_data = training_data.astype('float')
	test_data = test_data.astype('float')

	data_size = training_labels.size
	input_size = test_labels.size

	# Get sum of squared values for each rows
	t_data_sq_sum = np.sum(np.square(training_data), axis=1)
	# Extend array
	a_sq = np.broadcast_to(t_data_sq_sum, (input_size, data_size))

	# Get sum of squared values for each rows
	i_data_sq_sum = np.sum(np.square(test_data), axis=1)
	i_data_sq_sum = np.expand_dims(i_data_sq_sum, axis=0)
	# Extend array
	b_sq = np.repeat(i_data_sq_sum, data_size).reshape(input_size, data_size)
	
	ab = np.dot(test_data, training_data.T)

	# (a - b)^2 = a^2 + b^2 -2ab
	eu_dist = a_sq + b_sq - 2*ab

	return training_labels[np.argmin(eu_dist, axis = 1)]

def err_rate(test_labels,  knn_labels):
	err = 0
	for y in range (0, test_labels.size):
		if test_labels[y] != knn_labels[y]:
			err += 1
	#print err
	return err

def prototype_sel(training_data, training_labels, m):

	partitioned_data = [[] for i in range(10)]
	label_mean = [[] for i in range(10)]
	partitioned_label = [0 for i in range(10)]
	for d in range (0, training_labels.size):
		idx = training_labels[d]
		idx = idx[0]
		partitioned_data[idx].append(training_data[d])
		partitioned_label[idx] += 1

	print partitioned_label[0]

	# Get average for each label
	for i in range (0, 10):
		label_mean[i] = np.mean(partitioned_data[i], axis=0)
	
	sel = random.sample(xrange(60000), 2*m)
	# Get kNN from 2x input

	return

ocr = loadmat('ocr.mat')
#print ocr['data'].shape
training_size = [1000, 2000, 4000, 8000]
iteration = 10
for i in range(len(training_size)):
	err = []
	for j in range(0, iteration):
		#print j, "th run with size ", training_size[i]
		sel = random.sample(xrange(60000), training_size[i])
	#	prototype_sel(ocr['data'][sel], ocr['labels'][sel], training_size)
		input_labels = ocr['testlabels']
		knn_labels = knn(ocr['data'][sel], ocr['labels'][sel], ocr['testdata'], input_labels)
		err.append(err_rate(input_labels, knn_labels))
	# At the end of iteratioin, print avg and stdev
	div = input_labels.size/100	
	print training_size[i], np.mean(err)/div, np.std(err)/div
