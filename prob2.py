#!/usr/bin/python
from __future__ import division
from scipy.io import loadmat
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm 
import time
import csv
from collections import Counter

TR_FILE="reviews_tr.csv"

# return avg of n+1 classifiers
def avg_pct_1pass(data, label, w)
	w_avg = w
# w = (0, ... 0)
# w_avg = (0, ..., 0)
# for each data
#  if y * ( w dot xi) < 0
#      w = w + y*x
#  w_avg += w
	return w, w_avg

def avg_pct_2pass(data, label)
	#suffle data
	w = 0
	w, w_avg =  avg_pct_1pass(data, level, w)
	#suffle data
	w, w_avg =  avg_pct_1pass(data, level, w)
	return w_avg

wordcount={}
with open(TR_FILE) as csvfile:
	reader = csv.reader(csvfile)

	#Skip the header
	reader.next()
	cnt = Counter()

	ngram = 2
	prepared = False
	for row in reader:
		prev_word = ""
		for word in row[1].split():
			if len(prev_word.split()) < ngram -1:
				prev_word = prev_word + " " + word
				continue;

			if prev_word.partition(' ')[2] != "":
				new_word = prev_word.partition(' ')[2] + ' ' + word
			else:
				new_word = word
			prev_word = new_word
			if new_word not in wordcount:
				wordcount[new_word] = 1
			else:
				wordcount[new_word] += 1
	print wordcount

# tf = get_tf()
# format of tf = ( (1, 2, 0, 0, 0, ... 3), label1), ((1,...0), label2)
# or just two matrix just like data and label

w_avg = avg_perceptron_2pass(data, label)

# training data
hw3_test(w_avg, data, label)

# test data
hw3_test(w_avg, test_data, test_label)


