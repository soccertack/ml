import numpy as np
import pandas as pd
from LinearRegressionSimulator import *

x = LinearRegressionSimulator(np.array([[1, 1, 1, 1, 1, 1, 1]]), 0)
data = np.array([['','Col1'],
                ['Row1',1],
                ['Row1',2],
                ['Row1',3],
                ['Row1',4],
                ['Row1',5],
                ['Row2',6]])
XInput = pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:])
print (XInput)
print (x.SimPoly(XInput))
