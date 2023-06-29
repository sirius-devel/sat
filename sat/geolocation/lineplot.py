import io
import matplotlib.pyplot as plt
import numpy as np

def multiple_plots(file_path):
	x, y = np.loadtxt(file_path, delimiter=',', unpack = True)
	y = y/20
	print("y.size=", y.size)
	plt.plot(x, y, 'r')
	plt.xlabel('images')
	plt.ylabel('corner loss')
	plt.title('Loss on valid data')
	plt.show()
	

multiple_plots("/home/user/sat/geolocation/valid_250.txt")