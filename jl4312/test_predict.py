import pickle
from timeit import default_timer as timer
from predict import *
from sklearn import metrics

INPUT_X_FILE="others/Data_x.pkl"
INPUT_Y_FILE="others/Data_y.pkl"

def get_training_data():
        f = open(INPUT_X_FILE, 'rb')
        x_array = pickle.load(f)
        f.close()

        f = open(INPUT_Y_FILE, 'rb')
        y_array = pickle.load(f)
        f.close()

        return x_array, y_array

x_array, y_array = get_training_data()

predict_start = timer()
predicted_Y = predict(x_array)
predict_end = timer()

print ("accuracy from dup: ", metrics.accuracy_score(y_array, predicted_Y))
print ("predict time: ", predict_end - predict_start) 
