from SimClasses import *
from Classifier_A import *
from Classifier_B import *
from Classifier_C import *
from sklearn import metrics
import collections
import pickle

PICKLE_RESULT = "Results.pkl"
PICKLE_PARAMS = "Parameters.pkl"

def Get_Classifier(classifier, train_X, train_Y):

	if classifier == 'A':
		class_obj = Classifier_A(train_X, train_Y)
	elif classifier == 'B':
		class_obj = Classifier_B(train_X, train_Y)
	elif classifier == 'C':
		class_obj = Classifier_C(train_X, train_Y)
	else:
		print ("Undefined class", classifier)
		sys.exit()
	return class_obj

def RunTest(N, D, Distance, classifier):

	test_N = 100
	a = SimClasses()
	train_X, train_Y = a.GetData(N, D, Distance)
	test_X, test_Y = a.GetData(test_N, D, Distance)
	class_obj = Get_Classifier(classifier, train_X, train_Y)
	predicted_Y = class_obj.Classify(test_X)
	return metrics.accuracy_score(test_Y, predicted_Y), class_obj.Get_Params(), \
		class_obj.Get_Compute_Times()

def SaveResult(outDict, paramDict, classifier, item, var,\
				accuracy, e_time, params):
	outDict[('i', classifier, item, var)] = accuracy 
	outDict[('ii', classifier, item, var)] = e_time
	paramDict[('i', classifier, item, var)] = params
	paramDict[('ii', classifier, item, var)] = params

def RunAndSave(N, D, Distance, classifier, item, var, outDict, paramDict):
	accuracy, params, e_time = RunTest(N, D, Distance, classifier)
	SaveResult(outDict, paramDict, classifier, item, var,\
			accuracy, e_time, params)

def TestClassifiers():
	
	classes = ["A", "B", 'C']
	outDict = collections.OrderedDict()
	paramDict = collections.OrderedDict()

	for classifier in classes:

		# a) Fixed N and Fixed Distance
		N = 10000	# Data size
		Distance = 1	# stdev for Tail distribution
		D_array = [1, 2, 3, 4, 5]
		item = 'a'
		for D in D_array:
			RunAndSave(N, D, Distance, classifier, item, D, outDict, paramDict)

		# b) Fixed D and Fixed Distance
		Distance = 1	# stdev for Tail distribution
		D = 2
		N_array = [5, 10, 100, 1000, 10000]
		item = 'b'
		for N in N_array:
			RunAndSave(N, D, Distance, classifier, item, N, outDict, paramDict)

		# c) Fixed D and Fixed N 
		N = 1000	# Data size
		D = 2
		Dist_array = [0, 1, 2, 3, 4]
		item = 'c'
		for Distance in Dist_array:
			RunAndSave(N, D, Distance, classifier, item, Distance, outDict, paramDict)

	for k, v in outDict.items():
		if k[0] == 'i':
			print (k, v)

	with open(PICKLE_RESULT, 'wb') as f:
		pickle.dump(outDict, f)
	with open(PICKLE_PARAMS, 'wb') as f:
		pickle.dump(paramDict, f)

# Plot input data
'''
	N = 10000	# Data size
	X_x = train_X[:,0]
	X_x.shape = (N, 1)
	X_y = train_X[:,1]
	X_y.shape = (N, 1)

	plt.scatter(X_x, X_y, marker='+', s=1)
	plt.show()
'''

TestClassifiers()
