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
import matplotlib.pyplot as plt

def draw_pairwise_plot(x_array, y_array):
# plot preparation
	colors = ['dummy', 'blue', 'red']
	cls = ['dummy','class1', 'class2']
	D = 100
	plt_idx = 0
	pic_num = 1
	class_1 = []
	class_2 = []
	for dims in itertools.combinations(range(D), 2):
		dim1 = dims[0]
		dim2 = dims[1]
		plt_idx += 1
		#plt.subplot(2, 3, plt_idx)
		print ("Dimension "+str(dim1) + " Dimension "+str(dim2))
		for i in range (1,3):
			# Pick rows with the given class i
			X_x = x_array[y_array == i][:,dim1]
			num_of_rows = X_x.shape[0]
			X_y = x_array[y_array == i][:,dim2]

			'''
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
			'''
		
			X_plot = X_x
			Y_plot = X_y
			plt.xlim([-2,2])
			plt.ylim([-2,2])
			plt.scatter(X_plot, Y_plot, marker='.', c=colors[i], label=cls[i], s=0.1)
			plt.xlabel("Dimension "+str(dim1))
			plt.ylabel("Dimension "+str(dim2))
		plt.legend()
		plt.show()
		plt.close()

		'''
		if (plt_idx == 6):
			#plot = PdfPages("plots/Plot_" + str(dim1) +"_" + str(dim2) +".pdf")
			plot = PdfPages("plots/" + str(pic_num) +".pdf")
			plot.savefig()
			plot.close()
			plt.close()
			pic_num += 1
			plt_idx = 0
		'''

'''
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
