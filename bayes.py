#!/usr/bin/python
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

	# Test label 1 and 2 only 
	t_size = 707
	class_size = 2

	partitioned_data = [[] for i in range(class_size)]
	param = [0 for i in range(class_size)]

	for d in range (0, t_size):
		idx = t_labels[d]
		idx = idx[0]
		idx -= 1 # class label is from 1 to 20. Convert it from 0 to 19
		partitioned_data[idx].append(t_data[d])

	return partitioned_data
'''
	print len(partitioned_data[0])
	print [len(x) for x in partitioned_data]
	print sum(len(x) for x in partitioned_data)

	for y in range (class_size):
		partitioned_data[y] = np.array(partitioned_data[y])
		column_sum = np.sum(partitioned_data[y], axis=0)
		print column_sum.shape
		print column_sum
		column_sum = column_sum.todense()
		column_sum += 1
		column_sum /= (len(partitioned_data[y]) +2)
		param[y] = column_sum

	print param
'''

news = loadmat('news.mat')
partitioned_data = partition(news['testdata'], news['testlabels'])

sys.exit(0)

