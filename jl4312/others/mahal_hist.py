from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from basic import *
import sys
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope

# x_array is the original one
def get_mahal_hist(x_array):

	robust_scaler = RobustScaler()
	x_array = robust_scaler.fit_transform(x_array)

	nData = x_array.shape[0]

	#Sample for each class
	#sample = 10000
	#rd_1 = np.random.random_integers(0, nData/2-1, sample)
	#rd_2 = np.random.random_integers(nData/2, nData-1, sample)

	# Use the whole data
	half = int(nData/2 -1)
	X1 = x_array[0:half]
	X2 = x_array[-half:]

	print("Start fit 1")
	start = timer()
	#Calc Dist class 1
	robust_cov1 = EllipticEnvelope().fit(X1)
	end = timer()
	print("Took ", int(end-start), " seconds")
	print("Start mahal dist 1")
	maha_dist = robust_cov1.mahalanobis(X1)
	
	#Plot class 1
	plot_y1 = maha_dist

	#Calc Dist class 2
	print("Start fit 2")
	start = timer()
	robust_cov2 = EllipticEnvelope().fit(X2)
	end = timer()
	print("Took ", int(end-start), " seconds")
	print("Start mahal dist 2")
	maha_dist = robust_cov2.mahalanobis(X1)

	#Plot class 1
	plot_y2 = maha_dist

	# Plot data without outliers
	bins = np.linspace(0, 200, 400)
	data = np.vstack([plot_y1, plot_y2]).T
	plt.hist(data, bins, label=['class1', 'class2'])
	plt.ylabel('Number of occurance')
	plt.xlabel('Mahalanobis Distance')
	plt.title('Mahalanobis Distance Histogram')
	plt.legend()
	plt.show()
	plt.close()

	# Plot all data
	bins = np.linspace(0, 20000, 400)
	data = np.vstack([plot_y1, plot_y2]).T
	plt.hist(data, bins, label=['class1', 'class2'])
	plt.legend()

	plt.ylabel('Number of occurance')
	plt.xlabel('Mahalanobis Distance')
	plt.title('Mahalanobis Distance Histogram')
	plt.show()
	plt.close()

	with open("maha-dist_class1.pkl", 'wb') as f:
		pickle.dump(plot_y1, f)
	with open("maha-dist_class2.pkl", 'wb') as f:
		pickle.dump(plot_y2, f)

	with open("robust_cov_class1.pkl", 'wb') as f:
		pickle.dump(robust_cov1, f)
	with open("robust_cov_class2.pkl", 'wb') as f:
		pickle.dump(robust_cov2, f)

