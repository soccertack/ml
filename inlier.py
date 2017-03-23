print(__doc__)

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
	for i in range(0,2):
		if i == 0:
			X = x_array[0:half_rows,:]
		else:
			X = x_array[-half_rows:,:]

		# fit a Minimum Covariance Determinant (MCD) robust estimator to data
		robust_cov = MinCovDet().fit(X)
		maha_dist[i] = robust_cov.mahalanobis(X)
		inlier_list[i] = maha_dist[i] < 1000

	inlier = np.concatenate([inlier_list[0], inlier_list[1]])
	maha = np.concatenate([maha_dist[0], maha_dist[1]])
	with open("inlier.pkl", 'wb') as f:
		pickle.dump(inlier, f)
	with open("maha_dist.pkl", 'wb') as f:
		pickle.dump(maha, f)

	print (inlier[0:50,])
	print (maha[0:50,])
	
	return inlier

np.set_printoptions(threshold=np.inf)
x_array, y_array = get_training_data()
inlier = get_inliers(x_array)
#with open("inlier.pkl", 'wb') as f:
#	pickle.dump(inlier, f)
