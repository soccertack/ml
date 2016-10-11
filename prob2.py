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

with open(TR_FILE) as csvfile:
	reader = csv.reader(csvfile)

	#Skip the header
	reader.next()
	cnt = Counter()

	for row in reader:
		cnt.update(row[1].split())
	print cnt
	

