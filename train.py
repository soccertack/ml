import numpy as np
import csv
from PIL import Image
from numpy import genfromtxt
from sklearn import svm, datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#http://stackoverflow.com/questions/22902040/convert-black-and-white-array-into-an-image-in-python
# mylist is an array of 126 numbers (0 and 1)
def save_img(data, idx):
	im = Image.fromarray(data[idx].reshape((16,8)).astype('uint8')*255)
	im.save("result"+str(idx)+".jpg")

def read_data(filename):
	y = []
	with open(filename) as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in reader:
			y.append(row[0])

	my_data = genfromtxt(filename, delimiter=',')
	train_x = my_data[:,1:]
	return train_x, np.asarray(y)

def my_kernel(X, Y):
	return (10*np.dot(X, Y.T))**3

train_x, train_y = read_data('dataset1.csv')
print (train_x.shape)
#print (len(train_y))

# Confirm if we read correctly by checking the picture
sample_idx = 111
print (train_y[sample_idx])
print (train_x[sample_idx])
save_img(train_x, sample_idx)

# This is rbf but running forever.
'''
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(train_x, train_y)
print("The best parameters are %s with a score of %0.2f"
	% (grid.best_params_, grid.best_score_))

sys.exit()
'''

ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
for train_index, test_index in ss.split(train_x):
	cv_train_x = train_x[train_index]
	cv_train_y = train_y[train_index]
	cv_test_x = train_x[test_index]
	cv_test_y = train_y[test_index]
	# gamma 2 gives 0.89
	# gamma .001 gives 0.10
	# gamma 3 gives 0.89, too
	# 10 gives 0.89, too

	svc = svm.SVC(kernel='poly', gamma=100).fit(cv_train_x, cv_train_y)
	score = svc.score(cv_test_x, cv_test_y)
	print("score", score)
