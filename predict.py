import numpy as np
import pickle

INPUT_FILE="Data_y.pkl"

# Create a dummy input file
b=np.identity(10)
f=open(INPUT_FILE,'wb')
pickle.dump(b, f)
f.close()

f2 = open(INPUT_FILE, 'rb')
s = pickle.load(f2)
f2.close()
print (s)
