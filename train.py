import numpy as np
import pickle
from sklearn.model_selection import KFold

# TODO before exam: 
# 1. Add cross validation code
# (hold-out validation or K-fold cross-validation)

# During exam:
# 1. Add any training I've tried.

INPUT_X_FILE="Data_x.pkl"
INPUT_Y_FILE="Data_y.pkl"

def train(tr_x, tr_y):
	print ("X data is ", tr_x.shape)
	print ("Y data is ", tr_y.shape)
	print (x_array)

	kf = KFold(n_splits=5, shuffle=True)
	for train, test in kf.split(tr_x):
		print("%s %s" % (train, test))
		print ("Print test data")
		print (tr_x[test])
	
	print ("Train function is supposed to return coefficients")
	return

# Create a dummy input file
b=np.identity(11)
f=open(INPUT_X_FILE,'wb')
pickle.dump(b, f)
f.close()

f2 = open(INPUT_X_FILE, 'rb')
x_array = pickle.load(f2)
f2.close()
f2 = open(INPUT_X_FILE, 'rb')
y_array = pickle.load(f2)
f2.close()

train(x_array, y_array)


