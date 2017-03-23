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
			#plt.scatter(plot_x, plot_y, marker='.', c=colors[cls], s=0.1)
			#plt.show()
			#plt.close()

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
		#"BernoulliNB": BernoulliNB(),
		#"SGDClassifier": SGDClassifier(loss="hinge", penalty="l2", shuffle=True),
		#"Decision Tree": tree.DecisionTreeClassifier(),
		#"KNN": KNeighborsClassifier(n_neighbors=3),
		#"Logistic": LogisticRegression(penalty='l1', tol=0.0001, C=1,
	#			fit_intercept=True, intercept_scaling=1,
	#			class_weight=None, random_state=None,
	#			solver='liblinear', max_iter=100),
		#"Linear SVM": svm.SVC(kernel='linear', C=0.025),
		#"AdaBoost": AdaBoostClassifier(),
		#"GaussianNB": GaussianNB(),
		#"Poly SVM":  svm.SVC(kernel='poly'),
		"RBF SVM": svm.SVC(gamma=2, C=1),
		}


	for i, (clf_name, clf) in enumerate(classifiers.items()):
		print (clf_name)
		clf = clf.fit(tr_x, tr_y)
		predicted_Y = clf.predict(tr_x)
		print ("accuracy from training: ", metrics.accuracy_score(tr_y, predicted_Y))

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

# -------------------------------------------------------------
#TODO: Do feature selection
#Removing features with low variance
#p = 0.90
#sel = VarianceThreshold(threshold=(p * (1 - p)))
#selected_x = sel.fit_transform(x_array)
#x_array = selected_x

#Univariate feature selection
#x_array = SelectKBest(chi2, k=2).fit_transform(x_array, y_array)
#print("after selection")
#print (x_array.shape)
# -------------------------------------------------------------

# Print out basic information about the training set
#basic_info(x_array, y_array)

# Check the range of each dimension
#check_dimension(x_array, y_array)

#TODO: check if this transform helps performance
# normalize -9 to 1, rest to 0
# x_array= (x_array - x_array.mean()) / x_array.std()
#x_array[x_array > -5] = 0
#x_array[x_array < -0] = 1

# Remove outliers
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
