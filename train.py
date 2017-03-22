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

INPUT_X_FILE="Data_x.pkl"
INPUT_Y_FILE="Data_y.pkl"
CLASSIFIER_FILE="Trained_classifier.pkl"

def train(tr_x, tr_y):
	print ("X data dimension is ", tr_x.shape)
	print ("Y data dimension is ", tr_y.shape)

	# This is the line YOU need to play around
	clf = LinearSVC()

	clf = clf.fit(tr_x, tr_y)
	predicted_Y = clf.predict(tr_x)
	print ("accuracy from training: ", metrics.accuracy_score(tr_y, predicted_Y))

	cross_validation = 5
	# Cross validation method 1
	score = cross_val_score(clf, tr_x, tr_y, cv=cross_validation)
	print ("cross validation score", score)

	# Cross validation method 2
	#k_fold = KFold(n_splits=cross_validation)
	#manual_kfold_score = [clf.fit(tr_x[tr_idx], tr_y[tr_idx]).score(tr_x[test_idx], tr_y[test_idx]) for tr_idx, test_idx in k_fold.split(tr_x)]
	#print ("manual kfold score", manual_kfold_score)

	joblib.dump(clf, 'CLASSIFIER_FILE') 

	return

start = timer()

f2 = open(INPUT_X_FILE, 'rb')
x_array = pickle.load(f2)
f2.close()
f2 = open(INPUT_Y_FILE, 'rb')
y_array = pickle.load(f2)
f2.close()

# Ensure Dimensions
print (x_array.shape)
print (y_array.shape)

colors = ['dummy', 'blue', 'orange']
for dims in itertools.combinations(range(100), 2):
	dim1 = dims[0]
	dim2 = dims[1]
	for i in range (1,3):
		# Pick rows with the given class i
		X_x = x_array[y_array == i][:,dim1]
		X_y = x_array[y_array == i][:,dim2]
		plt.scatter(X_x, X_y, marker='.', c=colors[i], s=1)
		plt.xlabel("Dimension "+str(dim1))
		plt.ylabel("Dimension "+str(dim2))
	plot = PdfPages("Plot_" + str(dim1) +"_" + str(dim2) +".pdf")
	plot.savefig()
	plot.close()
	plt.close()

train(x_array, y_array)
predicted_Y = predict(x_array)
print ("accuracy from dup: ", metrics.accuracy_score(y_array, predicted_Y))

end = timer()
print (end - start)
