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

#FOLDER_NAME = "woodbridge"
#FOLDER = FOLDER_NAME + '/'
#DATAPATH = "/home/user/sat/sat_data/"
#MODEL_PATH = "/home/user/sat/models/trained_model_output.pth"
#DATASET_PATH = /home/user/sat




# USAGE:
# python3 train.py MODEL_PATH DATASET_PATH IS_RESNET

# TRAIN:
# python3 train.py /home/user/sat/models/trained_model_output.pth /home/user/sat 1

###--- TRAINING PARAMETERS
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("MODEL_PATH")
	parser.add_argument("DATASET_PATH")
	parser.add_argument("IS_RESNET")
	args = parser.parse_args()

	MODEL_PATH = args.MODEL_PATH
	DATASET_PATH = args.DATASET_PATH
	IS_RESNET = args.IS_RESNET
	if (IS_RESNET):
		print("load resnet model")
		geo.train_optflow_model(MODEL_PATH, DATASET_PATH, True)
	else:
		print("load vgg model")
		geo.train_optflow_model(MODEL_PATH, DATASET_PATH, False)

