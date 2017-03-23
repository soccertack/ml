from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import pandas as pd
from basic import get_training_data

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

x_array, y_array = get_training_data()

x_array = -x_array
y_array = -y_array
sample = 1000
x_array_sub = {}
y_array_sub = {}
x_array_sub[1] = x_array[0:sample]
y_array_sub[1] = y_array[0:sample]

x_array_sub[2] = x_array[-sample:]
y_array_sub[2] = y_array[-sample:]

x_array_sub[1] = np.random.randn(1000,100)
x_array_sub[2] = np.random.randn(1000,100)
n = 100
for j in range(0, 95, 1):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	i = 1
	for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
	    '''
	    xs = randrange(n, 23, 32)
	    ys = randrange(n, 0, 100)
	    zs = randrange(n, zl, zh)
	    '''
	    cri  = x_array_sub[i][:,j]
	    xs = x_array_sub[i][:,j]
	    ys = x_array_sub[i][:,j+1]
	    zs = x_array_sub[i][:,j+2]
	    xs = xs[[cri<5]]
	    ys = ys[[cri<5]]
	    zs = zs[[cri<5]]

	    cri  = ys
	    xs = xs[[cri<5]]
	    ys = ys[[cri<5]]
	    zs = zs[[cri<5]]

	    cri  = zs
	    xs = xs[[cri<5]]
	    ys = ys[[cri<5]]
	    zs = zs[[cri<5]]

	    ax.scatter(xs, ys, zs, c=c, marker=m)
	    i+= 1

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()

plt.close()
