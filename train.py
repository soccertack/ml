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

INPUT_X_FILE="Data_x.pkl"
INPUT_Y_FILE="Data_y.pkl"
CLASSIFIER_FILE="Trained_classifier.pkl"

def get_training_data():
	f = open(INPUT_X_FILE, 'rb')
	x_array = pickle.load(f)
	f.close()

	f = open(INPUT_Y_FILE, 'rb')
	y_array = pickle.load(f)
	f.close()
	
	return x_array, y_array

#Print out basic info
def basic_info(x_array, y_array):
	print (x_array.shape)
	print (y_array.shape)

	# Checked that number of sample for each class is the same
	print (np.bincount(y_array.astype(int)))

	# Check the range of each dimension
	# Max: around 0.5
	print (np.amax(x_array, axis = 0))
	# Min: exactly -9.332
	print (np.amin(x_array, axis = 0))


def check_dimension(x_array, y_array):
	for dim in range(0,num_of_dim):
		for cls in (1, 2):
			# take data for class i
			plot_y = x_array[y_array == cls][:,dim]
			plot_x = np.arange(plot_y.shape[0])
			
			# print out number of samples greater than -5
			print(np.bincount(np.greater(plot_y, -5)))

			# Plot distribution
			# plt.scatter(plot_x, plot_y, marker='.', c=colors[cls], s=0.1)
			# plt.show()
			# plt.close()

def train(tr_x, tr_y):
	print ("X data dimension is ", tr_x.shape)
	print ("Y data dimension is ", tr_y.shape)

	cov = np.cov(tr_x)
	print ("cov")
	print (cov)
	# This is the line YOU need to play around
	clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

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

x_array, y_array = get_training_data()
num_of_data = x_array.shape[0]
num_of_dim = x_array.shape[1]

# plot preparation
colors = ['dummy', 'blue', 'red']

# Print out basic information about the training set
#basic_info(x_array, y_array)

# Check the range of each dimension
#check_dimension(x_array, y_array)

# --------- Up to this point, data is intact ---------


# TODO: remove this. Sample 1000 data
sample_size = 10000
x_1 = x_array[0:sample_size]
#print ("x[0:sample]")
#print (x_1)
y_1 = y_array[0:sample_size]
x_2 = x_array[-sample_size:]
y_2 = y_array[-sample_size:]
x_array = np.concatenate((x_1, x_2), axis=0)
y_array = np.concatenate((y_1, y_2), axis=0)
#print (x_array.shape)
#print (y_array.shape)
#print ("x[0:sample] + x[-sample:]")
#print (x_array)

D = 100

# normalize
# x_array= (x_array - x_array.mean()) / x_array.std()

#Remove outliers
num_of_rows = x_array.shape[0]
cond_array = x_array 
orig_x = x_array
orig_y = y_array

plt_idx = 0
pic_num = 1
class_1 = []
class_2 = []
for dims in itertools.combinations(range(D), 2):
	dim1 = dims[0]
	dim2 = dims[1]
	plt_idx += 1
	#plt.subplot(2, 5, plt_idx)
	print ("Dimension "+str(dim1) + " Dimension "+str(dim2))
	for i in range (1,3):
		# Pick rows with the given class i
		X_x = x_array[y_array == i][:,dim1]
		num_of_rows = X_x.shape[0]
		X_y = x_array[y_array == i][:,dim2]

		cond_array = X_x
		orig_x = X_x
		orig_y = X_y
		X1 = orig_x[[np.all(cond_array[k] < -5) for k in range(0,num_of_rows)]]
		Y1 = orig_y[[np.all(cond_array[k] < -5) for k in range(0,num_of_rows)]]

		cond_array = Y1
		orig_x = X1
		orig_y = Y1
		num_of_rows = Y1.shape[0]
		X2 = orig_x[[np.all(cond_array[k] < -5) for k in range(0,num_of_rows)]]
		Y2 = orig_y[[np.all(cond_array[k] < -5) for k in range(0,num_of_rows)]]

		if i == 1:
			class_1.append(X2.shape[0])
		else:
			class_2.append(X2.shape[0])
		#plt.scatter(X2, Y2, marker='.', c=colors[i], s=0.1)
		#plt.xlabel("Dimension "+str(dim1))
		#plt.ylabel("Dimension "+str(dim2))

	'''
	if (plt_idx == 10):
		#plot = PdfPages("plots/Plot_" + str(dim1) +"_" + str(dim2) +".pdf")
		plot = PdfPages("plots/" + str(pic_num) +".pdf")
		plot.savefig()
		plot.close()
		plt.close()
		pic_num += 1
		plt_idx = 0
	'''

plot_y = class_1
plot_x = np.arange(len(plot_y))
plt.plot(plot_x, plot_y, marker='.', c=colors[1])

plot_y = class_2
plot_x = np.arange(len(plot_y))
plt.plot(plot_x, plot_y, marker='.', c=colors[2])
plt.show()

sys.exit()
x_array = x_sanitized
y_array = y_sanitized
train(x_array, y_array)
predicted_Y = predict(x_array)
print ("accuracy from dup: ", metrics.accuracy_score(y_array, predicted_Y))

end = timer()
print (end - start)
