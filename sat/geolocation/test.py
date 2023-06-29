import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import io
import requests
from PIL import Image, ImageFont, ImageDraw
from torch.autograd import Variable
from torch.nn.functional import grid_sample
import pdb
from sys import argv
import argparse
import os
import random
from torchvision import transforms
from torchvision.models import VGG16_Weights
import geolocation as geo


#DATASET_PATH = "/home/user/sat/"
#MODEL_PATH = "/home/user/sat/models/trained_model_output.pth"
#TEST_DATA_SAVE_PATH = "/home/user/sat/models/output1.txt"


# USAGE:
# python3 test.py DATASET_PATH MODEL1_PATH MODEL2_PATH --TEST_DATA_SAVE_PATH

# TEST:
# python3 test.py /home/user/sat/ /home/user/sat/models/epoch38_model.pth -t /home/user/sat/models/output1.txt

###--- TESTING PARAMETERS
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("DATASET_PATH")
	parser.add_argument("MODEL1_PATH")
	parser.add_argument("MODEL2_PATH")
	parser.add_argument("-t", "--TEST_DATA_SAVE_PATH")

	args = parser.parse_args()

	DATASET_PATH = args.DATASET_PATH
	MODEL1_PATH = args.MODEL1_PATH
	MODEL2_PATH = args.MODEL2_PATH
	if args.TEST_DATA_SAVE_PATH == None:
		exit('Must supply TEST_DATA_SAVE_PATH argument')
	else:
		TEST_DATA_SAVE_PATH = args.TEST_DATA_SAVE_PATH
			
	geo.test_optflow_model(DATASET_PATH, MODEL1_PATH, MODEL2_PATH, TEST_DATA_SAVE_PATH)

