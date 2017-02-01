import numpy as np
import pandas as pd
from LinearRegressionSimulator import *


# my UNI is jl4312
x = LinearRegressionSimulator(np.array([[2, 1, 3, 4]]), 0)

N=3
M=10
training = np.random.uniform(0, 1, N)
test = np.random.uniform(0, 1, M)
training.shape = (N, 1)

print(np.vander(training.flatten(), 5, increasing=True))

print (training.shape)
training = np.array([10, 100, 1000])
XInput = pd.DataFrame(training)

print (XInput)
print (x.SimPoly(XInput))



#compare output with numpy.polyfit
