import pickle

INPUT_X_FILE="StudentData/Data_x.pkl"
INPUT_Y_FILE="StudentData/Data_y.pkl"
CLASSIFIER_FILE="Trained_classifier.pkl"

def get_training_data():
        f = open(INPUT_X_FILE, 'rb')
        x_array = pickle.load(f)
        f.close()

        f = open(INPUT_Y_FILE, 'rb')
        y_array = pickle.load(f)
        f.close()

        return x_array, y_array
