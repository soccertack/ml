#!/usr/bin/python
from __future__ import division
from scipy.io import loadmat
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm 
import time

def partition(t_data, t_labels):

	t_size = t_labels.size
	class_size = np.unique(t_labels).size

	partitioned_data = [[] for i in range(class_size)]
	param = [0 for i in range(class_size)]

	for d in range (0, t_size):
		idx = t_labels[d]
		idx = idx[0]
		idx -= 1 # class label is from 1 to 20. Convert it from 0 to 19
		partitioned_data[idx].append(t_data[d])

	return partitioned_data


def get_class_prior(partitioned_data):
	class_size = len(partitioned_data)
	data_size = sum(len(x) for x in partitioned_data)

	class_prior = []
	for y in range (class_size):
		class_prior.append(len(partitioned_data[y])/data_size)
			
	print class_prior
	print sum(class_prior)
	return class_prior

def get_ccdist(partitioned_data):
	class_size = len(partitioned_data)

	ccdist = []
	for y in range (class_size):
		partitioned_data[y] = np.array(partitioned_data[y])
		# sum for each xi where i is from 0 to d
		column_sum = np.sum(partitioned_data[y], axis=0)
		column_sum = column_sum.todense()
		numerator = column_sum + 1
		denominator = (len(partitioned_data[y]) +2)
		ccdist.append(numerator/denominator)

	print ccdist
	return ccdist

news = loadmat('news.mat')
# partitioned_data[i] contains data with label i
partitioned_data = partition(news['testdata'], news['testlabels'])
print "class size is ", len(partitioned_data)

class_prior = get_class_prior(partitioned_data)
ccdist = get_ccdist(partitioned_data)


sys.exit(0)

