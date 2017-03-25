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
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from util import *
from mahal_hist import *
import argparse

INPUT_X_FILE="StudentData/Data_x.pkl"
INPUT_Y_FILE="StudentData/Data_y.pkl"
CLASSIFIER_FILE="Trained_classifier.pkl"
SCALER_FILE="Scaler.pkl"

def get_cov(X):
	cov = np.cov(X)
	print (cov)

def check_pca(X):
	pca = PCA(n_components=100).fit(X)
	print(pca.explained_variance_ratio_)

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

def train_mahal(x_array, y_array):

	rng = np.random.RandomState(1)

	with open("robust_cov_class1.pkl", 'rb') as f:
		robust_cov1 = pickle.load(f)
	with open("robust_cov_class2.pkl", 'rb') as f:
		robust_cov2 = pickle.load(f)

	# Open mahal-dist
	with open("maha-dist_class1.pkl", 'rb') as f:
		maha_dist1 = pickle.load(f)
	with open("maha-dist_class2.pkl", 'rb') as f:
		maha_dist2 = pickle.load(f)

	print ("maha dist : ", maha_dist1)
	print ("maha dist size : ", maha_dist1.shape)

	class1_hist, bins = np.histogram(maha_dist1,bins=np.arange(20000))
	class2_hist, bins = np.histogram(maha_dist2,bins=np.arange(20000))


	print("ShuffleSplit")
	ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
	for train_index, test_index in ss.split(x_array):

		# Remove outliers from training and test set 
		robust_scaler = RobustScaler()
		inlier_selected_x = robust_scaler.fit_transform(x_array[train_index])
		inlier_test_x = robust_scaler.transform(x_array[test_index])

		outlier = 0
		correct = 0
		for i in range(0,8000):
			print("-------")
			print ("idx: "+ str(i))
			print ("shape: "+ str(inlier_selected_x.shape))
			dist1 = robust_cov1.mahalanobis(inlier_selected_x[i:i+1])
			print ("Dist 1: ", str(dist1))
			dist1_int = int(dist1)
			prob1 = class1_hist[dist1_int-2:dist1_int+3]
			print("Prob: ", prob1)
			dist2 = robust_cov2.mahalanobis(inlier_selected_x[i:i+1])
			print ("Dist 2: ", str(dist2))
			dist2_int = int(dist2)
			prob2 = class2_hist[dist2_int-2:dist2_int+3]
			print("Prob: ", prob2)
			if (prob1.sum() > prob2.sum()) and y_array[i] == 1:
				correct +=1
				print("correct!")
			if (prob1.sum() < prob2.sum()) and y_array[i] == 2:
				correct +=1
				print("correct!")
			print ("y: ", y_array[i])
				
		print ("corect", correct)
	return

def train(x_array, y_array):

	rng = np.random.RandomState(1)
	classifiers = {
		#"BernoulliNB": BernoulliNB(),
		#"SGDClassifier": SGDClassifier(loss="hinge", penalty="l2", shuffle=True),
		#"Decision Tree": tree.DecisionTreeClassifier(),
		#"KNN": KNeighborsClassifier(n_neighbors=3),
		#"Logistic": LogisticRegression(penalty='l2', fit_intercept=True, solver='liblinear', max_iter=100),
		#"Linear SVM": svm.SVC(kernel='linear', C=0.025),
		#"AdaBoost": AdaBoostClassifier(),
		"AdaBoost estimate 500": AdaBoostClassifier(n_estimators=500),
		#"AdaBoost depth 4": AdaBoostClassifier(DecisionTreeClassifier(max_depth=4)),
		#"AdaBoost 300 & 4": AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
		#			n_estimators=300),
		#"AdaBoost rng": AdaBoostClassifier(random_state=rng),
		#"AdaBoost 300 & 4 & rng": AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
		#			n_estimators=300, random_state=rng),
		#"GaussianNB": GaussianNB(),
		#"Poly SVM":  svm.SVC(kernel='poly'),
		#"RBF SVM": svm.SVC(gamma=2, C=1),
		}


	for i, (clf_name, clf) in enumerate(classifiers.items()):
		print (clf_name)
		'''
		clf = clf.fit(tr_x, tr_y)
		predicted_Y = clf.predict(tr_x)
		print ("accuracy from training: ", metrics.accuracy_score(tr_y, predicted_Y))
		'''

		print("ShuffleSplit")
		ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
		for train_index, test_index in ss.split(x_array):

			# Remove outliers from training and test set 
			robust_scaler = RobustScaler()
			inlier_selected_x = robust_scaler.fit_transform(x_array[train_index])
			inlier_test_x = robust_scaler.transform(x_array[test_index])
	
			score = clf.fit(inlier_selected_x, y_array[train_index]).score(inlier_test_x, y_array[test_index])
			print("score", score)

	joblib.dump(clf, CLASSIFIER_FILE) 
	joblib.dump(robust_scaler, SCALER_FILE) 

	return robust_scaler

###################################################################
#	Start
##################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mahal", help="get mahal-dist histogram", action='store_true')
parser.add_argument("-b", "--basic", help="print basic info", action='store_true')
parser.add_argument("-p", "--pca", help="run pca", action='store_true')
parser.add_argument("-t", "--two", help="draw pair-wise 2D graphs", action='store_true')
parser.add_argument("-e", "--exp", help="Training based on mahal-dist", action='store_true')
args = parser.parse_args()

start = timer()
x_array, y_array = get_training_data()
num_of_data = x_array.shape[0]

# Print out PCA result
if args.pca:
	check_pca(x_array)
	sys.exit()

# Print out basic information about the training set
if args.basic:
	basic_info(x_array, y_array)
	sys.exit()

# Get Mahanobis Distance
if args.mahal:
	get_mahal_hist(x_array)
	sys.exit()

if args.two:
	draw_pairwise_plot(x_array, y_array)
	sys.exit()

if args.exp:
	# Start training with mahal-distance.(Experimental feature)
	train_mahal(x_array, y_array)
else:
	# Start training. Result is saved in CLASSIFIER_FILE and SCALER_FILE
	train(x_array, y_array)

# predict() test-run
predict_start = timer()
predicted_Y = predict(x_array[rd])
predict_end = timer()
print ("predict time: ", predict_end - predict_start) 
print ("accuracy: ", metrics.accuracy_score(y_array[rd], predicted_Y))
end = timer()

print (end - start)
