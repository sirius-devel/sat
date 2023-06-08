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
# python3 train.py FOLDER_NAME DATAPATH MODEL_PATH DATASET_PATH

# TRAIN:
# python3 train.py woodbridge /home/user/sat/sat_data/ /home/user/sat/models/trained_model_output.pth /home/user/sat

###--- TRAINING PARAMETERS
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("FOLDER_NAME")
	parser.add_argument("DATAPATH")
	parser.add_argument("MODEL_PATH")
	parser.add_argument("DATASET_PATH")
	args = parser.parse_args()

	FOLDER_NAME = args.FOLDER_NAME
	FOLDER = FOLDER_NAME + '/'
	DATAPATH = args.DATAPATH
	MODEL_PATH = args.MODEL_PATH
	DATASET_PATH = args.DATASET_PATH
	geo.train_optflow_model(DATAPATH, FOLDER, MODEL_PATH, DATASET_PATH)
