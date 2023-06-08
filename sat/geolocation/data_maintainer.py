import geolocation as geo
import torchvision
from torchvision import transforms
import numpy as np

DATAPATH = "/home/user/sat/sat_data/"
DATASET_PATH = "/home/user/sat/"

def write_dataset(dataset_path, data_path, generated_img_count, start_index):
	FOLDER_NAME = "woodbridge"
	FOLDER = FOLDER_NAME + '/'
	img_batch, template_batch, param_batch = geo.generate_data(generated_img_count, data_path, FOLDER)
	for i in range(generated_img_count):
		img_path = dataset_path + "/" + str(start_index + i) + "_transform.jpg"
		transforms.ToPILImage()(img_batch[i].data[:,:,:]).save(img_path)
		template_path = dataset_path + "/" + str(start_index + i) + "_template.jpg"
		transforms.ToPILImage()(template_batch[i].data[:,:,:]).save(template_path)

	for i in range(generated_img_count):
		parameters_path = dataset_path + "/" + str(start_index + i) + "_params.csv"
		with open(parameters_path, 'w', encoding='UTF8', newline='\n') as f:
			#write row
			ar = param_batch[i].cpu().reshape(1, 8).numpy()
			np.savetxt(f, ar, delimiter=',', fmt='%f')

def create_dataset(dataset_path, data_path):
	img_count=500
	start_index = 0
	for i in range(50):
		start_index = i*img_count
		write_dataset(dataset_path + "train", data_path, img_count, start_index)
	start_index = 0
	for i in range(10):
		start_index = i*img_count
		write_dataset(dataset_path + "valid", data_path, img_count, start_index)
		start_index = 0
	start_index = 0	
	for i in range(10):
		start_index = i*img_count
		write_dataset(dataset_path + "test", data_path, img_count, start_index)

create_dataset(DATASET_PATH, DATAPATH)
