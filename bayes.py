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

	for n in range (0, t_size):
		idx = t_labels[n]
		idx = idx[0]
		idx -= 1 # class label is from 1 to 20. Convert it from 0 to 19
		partitioned_data[idx].append(t_data[n])

	return partitioned_data


def get_class_prior(partitioned_data):
	class_size = len(partitioned_data)
	data_size = sum(len(x) for x in partitioned_data)

	class_prior = []
	for y in range (class_size):
		class_prior.append(len(partitioned_data[y])/data_size)
			
	print "class prior", class_prior
	return class_prior

def get_ccdist(partitioned_data):
	class_size = len(partitioned_data)

	ccdist = []
	for y in range (class_size):
		partitioned_data[y] = np.array(partitioned_data[y])
		# sum for each xi where i is from 0 to n
		column_sum = np.sum(partitioned_data[y], axis=0)
		column_sum = column_sum.todense()
		numerator = column_sum + 1
		denominator = (len(partitioned_data[y]) +2)
		ccdist.append(numerator/denominator)

	ccdist = np.array(ccdist)
	return ccdist

def get_ccdist_log(ccdist):

	ccdist_con = []
	for y in range (len(ccdist)):
		ccdist_rev = np.subtract(1, ccdist[y])
		ccdist_con.append(np.vstack((ccdist[y], ccdist_rev)))

	ccdist_con = np.array(ccdist_con)
	ccdist_log = np.log(ccdist_con)

	return ccdist_log

def get_class_raw(x, prior, ccdist):
	
	x = x.todense()
	x = np.array(x)

	max_val = float('-inf')
	max_idx = 0
	for y in range (len(prior)):
		z = prior[y]
		for k in range (x.shape[1]):
			if x[0][k] == 1:
				z *= ccdist[y][0][k]
			else:
				z *= (1- ccdist[y][0][k])
			
		if (z > max_val):
			max_val = z
			max_idx = y

	# add 1 to convert range from [0:19] to [1:20]
	return max_idx+1


def get_class(x, prior, ccdist):
	
	x = x.todense()
	x_con = np.vstack((x, 1-x))

	max_val = float('-inf')
	max_idx = 0
	for y in range (len(prior)):
		z = prior[y] + np.sum(np.inner(ccdist[y], x_con))
		if (z > max_val):
			max_val = z
			max_idx = y

	# add 1 to convert range from [0:19] to [1:20]
	return max_idx+1

def run(test_labels, test_data, class_prior, ccdist):
	err = 0
	total = 0
	for x in range(len(test_labels)):
		total += 1
		approx = get_class_raw(test_data[x], class_prior, ccdist)
		if approx != test_labels[x]:
			err += 1
		print "error rate: ", err/total, " err: ", err, " total: ", total

	print "err", err
	print "total", (len(test_labels))


news = loadmat('news.mat')

# partitioned_data[i] contains data with label i
partitioned_data = partition(news['data'], news['labels'])
print "class size is ", len(partitioned_data)

class_prior = get_class_prior(partitioned_data)
ccdist = get_ccdist(partitioned_data)

class_prior_log = np.log(class_prior)
ccdist_log = get_ccdist_log(ccdist)

print "run with training data"
run(news['labels'], news['data'], class_prior, ccdist)
print "run with test data"
run(news['testlabels'], news['testdata'], class_prior, ccdist)
sys.exit(0)

