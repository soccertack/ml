print(__doc__)
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from basic import *
import sys

from sklearn.covariance import EmpiricalCovariance, MinCovDet

# This function spots outliers and returns list of inliners
# Note that this function assumes that the first half and the second half
# is the different class, so calculate them separately.
def get_inliers(x_array):

	rows = x_array.shape[0]
	rows = 100000
	half_rows = int(rows/2)

	# This threashold is predetermined by looking at mahalanobis distance
	threshold = 1000
	
	maha_dist = {}
	inlier_list = {}
	X = x_array

	# fit a Minimum Covariance Determinant (MCD) robust estimator to data
	start = timer()
	robust_cov = MinCovDet().fit(X)
	end = timer()
	print("fit time: ", end-start)
	maha_dist[0] = robust_cov.mahalanobis(X)
	end2 = timer()
	print("transform time: ", end2-end)
#	with open("inlier.pkl", 'wb') as f:
#		pickle.dump(inlier, f)
	with open("maha_dist-all.pkl", 'wb') as f:
		pickle.dump(maha_dist[0], f)
#	with open("maha_dist-2.pkl", 'wb') as f:
#		pickle.dump(maha_dist[1], f)


	plot_x = np.arange(rows)
	plot_y = maha_dist[0]
	plt.plot(plot_x, plot_y, marker='.', c='blue')
	plt.show()

	return 1

np.set_printoptions(threshold=np.inf)
#x_array, y_array = get_training_data()
#inlier = get_inliers(x_array)
with open("maha_dist-all.pkl", 'rb') as f:
	maha_dist = pickle.load(f)

#sample = 100000
plot_y = maha_dist
plot_y = plot_y[plot_y>1000]
plot_x = np.arange(plot_y.shape[0])
plt.scatter(plot_x, plot_y, marker='.', c='blue')
plt.show()
