import io
import matplotlib.pyplot as plt
import numpy as np

def multiple_plots(file_path):
	y1, y2, y3 = np.loadtxt(file_path, delimiter=',', unpack = True)
	print("y1.size=", y1.size)
	x = np.zeros(y1.size)
	for i in range(y1.size) :
		x[i] = i + 1
	plt.subplot(1, 2, 1)
	plt.hist(y1, color = 'blue', bins = np.arange(0, 20000, 500))
	plt.hist(y2, color = 'red', bins = np.arange(0, 20000, 500), alpha=0.7)
	plt.legend(('train', 'sift'), loc = 'upper left')
	plt.xlabel('corner loss')
	plt.ylabel('number of images')
	plt.subplot(1, 2, 2)
	plt.hist(y1, color = 'blue', bins = np.arange(0, 20000, 500))
	plt.hist(y3, color = 'green', bins = np.arange(0, 20000, 500), alpha=0.7)
	plt.legend(('train', 'vgg16'), loc = 'upper left')
	plt.title('')
	plt.xlabel('corner loss')
	plt.ylabel('number of images')
	plt.show()
	

multiple_plots("/home/user/sat/models/output_test.txt")