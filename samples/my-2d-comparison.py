import numpy as np
import pickle
from sklearn.model_selection import KFold
from SimClasses import *
import itertools
from sklearn.externals import joblib
from predict import *
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from matplotlib.backends.backend_pdf import PdfPages
from timeit import default_timer as timer
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

colors = ['dummy', 'blue', 'red']

f = open("class1.pkl", 'rb')
class1 = pickle.load(f)
f.close()
f = open("class2.pkl", 'rb')
class2 = pickle.load(f)
f.close()

for ii in range(0,500):
	plot_y = class1[100*ii:100*(ii+1)]
	plot_x = np.arange(len(plot_y))
	plt.plot(plot_x, plot_y, marker='.', c=colors[1])

	plot_y = class2[100*ii:100*(ii+1)]
	plot_x = np.arange(len(plot_y))
	plt.plot(plot_x, plot_y, marker='.', c=colors[2])
	plt.show()
