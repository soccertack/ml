'''

# plot preparation
colors = ['dummy', 'blue', 'red']
D = 100
plt_idx = 0
pic_num = 1
class_1 = []
class_2 = []
for dims in itertools.combinations(range(D), 2):
	dim1 = dims[0]
	dim2 = dims[1]
	plt_idx += 1
	#plt.subplot(2, 5, plt_idx)
	print ("Dimension "+str(dim1) + " Dimension "+str(dim2))
	for i in range (1,3):
		# Pick rows with the given class i
		X_x = x_array[y_array == i][:,dim1]
		num_of_rows = X_x.shape[0]
		X_y = x_array[y_array == i][:,dim2]

		cond_array = X_x
		orig_x = X_x
		orig_y = X_y
		X1 = orig_x[[np.all(cond_array[k] < -5) for k in range(0,num_of_rows)]]
		Y1 = orig_y[[np.all(cond_array[k] < -5) for k in range(0,num_of_rows)]]

		cond_array = Y1
		orig_x = X1
		orig_y = Y1
		num_of_rows = Y1.shape[0]
		X2 = orig_x[[np.all(cond_array[k] < -5) for k in range(0,num_of_rows)]]
		Y2 = orig_y[[np.all(cond_array[k] < -5) for k in range(0,num_of_rows)]]

		if i == 1:
			class_1.append(X2.shape[0])
		else:
			class_2.append(X2.shape[0])
		#plt.scatter(X2, Y2, marker='.', c=colors[i], s=0.1)
		#plt.xlabel("Dimension "+str(dim1))
		#plt.ylabel("Dimension "+str(dim2))

	if (plt_idx == 10):
		#plot = PdfPages("plots/Plot_" + str(dim1) +"_" + str(dim2) +".pdf")
		plot = PdfPages("plots/" + str(pic_num) +".pdf")
		plot.savefig()
		plot.close()
		plt.close()
		pic_num += 1
		plt_idx = 0

plot_y = class_1
plot_x = np.arange(len(plot_y))
plt.plot(plot_x, plot_y, marker='.', c=colors[1])

plot_y = class_2
plot_x = np.arange(len(plot_y))
plt.plot(plot_x, plot_y, marker='.', c=colors[2])

with open("class1_result.pkl", 'wb') as f:
	pickle.dump(class_1, f)
with open("class2_result.pkl", 'wb') as f:
	pickle.dump(class_2, f)

plot = PdfPages("plots/all.pdf")
plot.savefig()
plot.close()
plt.close()
'''
