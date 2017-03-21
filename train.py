import numpy as np
import pickle

INPUT_X_FILE="Data_x.pkl"
INPUT_Y_FILE="Data_y.pkl"

def train(tr_x, tr_y):
	print ("X data is ", tr_x.shape)
	print ("Y data is ", tr_y.shape)
	print (x_array)
	print ("Train function is supposed to return coefficients")
	return

# Create a dummy input file
b=np.identity(10)
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


