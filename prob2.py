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

