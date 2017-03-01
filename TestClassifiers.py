from SimClasses import *
from Classifier_A import *
from Classifier_B import *
from Classifier_C import *
from Classifier_D import *
from sklearn import metrics
import collections
import pickle
from matplotlib.backends.backend_pdf import PdfPages

PICKLE_RESULT = "Results.pkl"
PICKLE_PARAMS = "Parameters.pkl"
test_dict = {}
training_dict = {}

def Get_Classifier(classifier, train_X, train_Y):

	if classifier == 'A':
		class_obj = Classifier_A(train_X, train_Y)
	elif classifier == 'B':
		class_obj = Classifier_B(train_X, train_Y)
	elif classifier == 'C':
		class_obj = Classifier_C(train_X, train_Y)
	elif classifier == 'D':
		class_obj = Classifier_D(train_X, train_Y)
	else:
		print ("Undefined class", classifier)
		sys.exit()
	return class_obj

def GetData(N, D, Distance, data_dict):
	if (N, D, Distance) not in data_dict:
		a = SimClasses()
		X, Y = a.GetData(N, D, Distance)
		data_dict[(N, D, Distance)] = (X, Y)
	return data_dict[(N, D, Distance)][0], data_dict[(N, D, Distance)][1]
		
def GetTrainingData(N, D, Distance):
	return GetData(N, D, Distance, training_dict)

def GetTestData(N, D, Distance):
	return GetData(N, D, Distance, test_dict)


def RunTest(N, D, Distance, classifier):

	test_N = 100
	a = SimClasses()
	train_X, train_Y = GetTrainingData(N, D, Distance)
	test_X, test_Y = GetTestData(test_N, D, Distance)
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
	
	classes = ["A", "B", 'C', 'D']
	items = ['a', 'b', 'c']
	measurements = ['i', 'ii']
	outDict = collections.OrderedDict()
	paramDict = collections.OrderedDict()
	varDict = {}
	item_label_Dict = {}
	fixed_Dict = {}

	for classifier in classes:

		# a) Fixed N and Fixed Distance
		item = 'a'
		N = 1000	# Data size
		Distance = 1	# stdev for Tail distribution
		D_array = [1, 2, 3, 4, 5]
		varDict[item] = D_array
		item_label_Dict[item] = 'D'
		fixed_Dict[item] = "N: " + str(N) + ", Distance: " + str(Distance)
		for D in D_array:
			RunAndSave(N, D, Distance, classifier, item, D, outDict, paramDict)

		# b) Fixed D and Fixed Distance
		item = 'b'
		Distance = 1	# stdev for Tail distribution
		D = 2
		N_array = [10, 100, 500, 1000, 10000]
		varDict[item] = N_array 
		item_label_Dict[item] = 'N'
		fixed_Dict[item] = "D: " + str(D) + ", Distance: " + str(Distance)
		for N in N_array:
			RunAndSave(N, D, Distance, classifier, item, N, outDict, paramDict)

		# c) Fixed D and Fixed N 
		item = 'c'
		N = 1000	# Data size
		D = 2
		Dist_array = [0, 1, 2, 3, 4]
		varDict[item] = Dist_array 
		item_label_Dict[item] = 'Distance'
		fixed_Dict[item] = "D: " + str(D) + ", N: " + str(N)
		for Distance in Dist_array:
			RunAndSave(N, D, Distance, classifier, item, Distance, outDict, paramDict)

	for k, v in outDict.items():
		if k[0] == 'i':
			print (k, v)

	with open(PICKLE_RESULT, 'wb') as f:
		pickle.dump(outDict, f)
	with open(PICKLE_PARAMS, 'wb') as f:
		pickle.dump(paramDict, f)

	
	x = [0, 1, 2, 3, 4]
	for measurement in measurements:
		for item in items:
			X_x =varDict[item] 
			plt.title("Measurement " + measurement + ", item " + item + "\n" +  fixed_Dict[item])
			plt.xlabel(item_label_Dict[item])
			for classifier in classes:
				X_y = []
				for var in varDict[item]:
					X_y.append(outDict[(measurement, classifier, item, var)])
				plt.plot(x, X_y, "ro")
				plt.plot(x, X_y, label=classifier)
				plt.xticks(x,X_x)
			plt.legend()

			result_plot = PdfPages("Plot_" + measurement +"_" + item +".pdf")
			result_plot.savefig()
			result_plot.close()

			plt.show()

	sys.exit()

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
