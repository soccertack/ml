import numpy as np
import pickle
from sklearn.model_selection import KFold
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
from sklearn.naive_bayes import *
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.kernel_approximation import RBFSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

INPUT_X_FILE="StudentData/Data_x.pkl"
INPUT_Y_FILE="StudentData/Data_y.pkl"
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
			plt.scatter(plot_x, plot_y, marker='.', c=colors[cls], s=0.1)
			plt.show()
			plt.close()
def check_rows(x_array, y_array):
	print (x_array[0])
	print (x_array[0].sum())
	print (x_array[0:50000].sum())
	print (x_array[-50000:].sum())
	
	
	sys.exit()

def train(tr_x, tr_y, x_array, y_array, inlier):
	print ("X data dimension is ", tr_x.shape)
	print ("Y data dimension is ", tr_y.shape)

	classifiers = {
		"BernoulliNB": BernoulliNB(),
		#"SGDClassifier": SGDClassifier(loss="hinge", penalty="l2", shuffle=True),
		#"Decision Tree": tree.DecisionTreeClassifier(),
		#"KNN": KNeighborsClassifier(n_neighbors=3),
		#"Logistic": LogisticRegression(penalty='l1', tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100),
		#"Linear SVM": svm.SVC(kernel='linear', C=0.025),
		"AdaBoost": AdaBoostClassifier(),
		#"GaussianNB": GaussianNB(),
		#"Poly SVM":  svm.SVC(kernel='poly'),
		#"RBF SVM": SVC(gamma-2, C=1).
		}


	for i, (clf_name, clf) in enumerate(classifiers.items()):
		print (clf_name)
		clf = clf.fit(tr_x, tr_y)
		predicted_Y = clf.predict(tr_x)
		print ("accuracy from training: ", metrics.accuracy_score(tr_y, predicted_Y))

		cross_validation = 3
		# Cross validation method 1
		score = cross_val_score(clf, tr_x, tr_y, cv=cross_validation)
		print ("cross validation score", score)
		predicted_Y = clf.predict(x_array)
		print ("accuracy from dup: ", metrics.accuracy_score(y_array, predicted_Y))

	
		X_train, X_test, y_train, y_test = train_test_split(
		tr_x, tr_y, test_size=0.4, random_state=0)
		clf = clf.fit(X_train, y_train)
		print("6:4 validation: ", clf.score(X_test, y_test))

		X_train, X_test, y_train, y_test = train_test_split(
		x_array, y_array, test_size=0.2, random_state=0)
		clf = clf.fit(X_train, y_train)
		print("8:2 validation: ", clf.score(X_test, y_test))

		print("ShuffleSplit")
		ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
		for train_index, test_index in ss.split(tr_x):
			score = clf.fit(tr_x[train_index], tr_y[train_index]).score(tr_x[test_index], tr_y[test_index])
			print("score", score)

		print("ShuffleSplit - train with good data, test with orig data")
		ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
		for train_index, test_index in ss.split(x_array):

			inlier_tr = inlier[train_index]
			selected_x = x_array[train_index]
			inlier_selected_x = selected_x[inlier_tr]
			selected_y = y_array[train_index]
			inlier_selected_y = selected_y[inlier_tr]

			score = clf.fit(inlier_selected_x, inlier_selected_y).score(x_array[test_index], y_array[test_index])
			print("score", score)

	joblib.dump(clf, CLASSIFIER_FILE) 

	return

start = timer()

x_array, y_array = get_training_data()
num_of_data = x_array.shape[0]
num_of_dim = x_array.shape[1]

#check_rows(x_array, y_array)

#Removing features with low variance
#p = 0.90
#sel = VarianceThreshold(threshold=(p * (1 - p)))
#selected_x = sel.fit_transform(x_array)
#x_array = selected_x

#Univariate feature selection
#x_array = SelectKBest(chi2, k=2).fit_transform(x_array, y_array)
#print("after selection")
#print (x_array.shape)


# plot preparation
colors = ['dummy', 'blue', 'red']

# Print out basic information about the training set
#basic_info(x_array, y_array)

# Check the range of each dimension
#check_dimension(x_array, y_array)

#sys.exit()
# --------- Up to this point, data is intact ---------


# TODO: remove this. Sample 1000 data
D = 100

# normalize -9 to 1, rest to 0
# x_array= (x_array - x_array.mean()) / x_array.std()
#x_array[x_array > -5] = 0
#x_array[x_array < -0] = 1

#Remove outliers


'''
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

	if (plt_idx == 10):
		#plot = PdfPages("plots/Plot_" + str(dim1) +"_" + str(dim2) +".pdf")
		plot = PdfPages("plots/" + str(pic_num) +".pdf")
		plot.savefig()
		plot.close()
		plt.close()
		pic_num += 1
		plt_idx = 0

plot_y = class_1
plot_x = np.arange(len(plot_y))
plt.plot(plot_x, plot_y, marker='.', c=colors[1])

plot_y = class_2
plot_x = np.arange(len(plot_y))
plt.plot(plot_x, plot_y, marker='.', c=colors[2])

with open("class1_result.pkl", 'wb') as f:
	pickle.dump(class_1, f)
with open("class2_result.pkl", 'wb') as f:
	pickle.dump(class_2, f)

plot = PdfPages("plots/all.pdf")
plot.savefig()
plot.close()
plt.close()
'''

f = open('inlier.pkl', 'rb')
inlier = pickle.load(f)
f.close()

tr_x = x_array[inlier]
tr_y = y_array[inlier]

train(tr_x, tr_y, x_array, y_array, inlier)
#predicted_Y = predict(x_array)
#print ("accuracy from dup: ", metrics.accuracy_score(y_array, predicted_Y))

end = timer()
print (end - start)
