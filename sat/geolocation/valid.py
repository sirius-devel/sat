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



USE_CUDA = torch.cuda.is_available()
# USAGE:
# python3 valid.py FOLDER_NAME DATAPATH MODEL_PATH DATASET_PATH

# TRAIN:
# python3 valid.py woodbridge /home/user/sat/sat_data/ /home/user/sat/models/trained_model_output.pth /home/user/sat

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
	validbatch_sz = 1
	valid_dataset = geo.SatImageDataset(DATASET_PATH + "/valid", 5000)
	valid_loader = geo.DataLoader(valid_dataset, batch_size = validbatch_sz, shuffle = True, pin_memory=True)

	if USE_CUDA:
		optf_net = geo.optf.OptFlow(geo.optf.PretrainedNet("/home/user/sat/models/epoch50_resnet.pth")).cuda()
		optf_net150 = geo.optf.OptFlow(geo.optf.PretrainedNet("/home/user/sat/models/epoch50_resnet_150.pth")).cuda()
		optf_net200 = geo.optf.OptFlow(geo.optf.PretrainedNet("/home/user/models/epoch50_resnet_200.pth")).cuda()
	else:
		optf_net = geo.optf.OptFlow(geo.optf.PretrainedNet("/home/user/models/epoch50_resnet.pth.pth"))
		optf_net150 = geo.optf.OptFlow(geo.optf.PretrainedNet("/home/user/models/epoch50_resnet_150.pth")).cuda()
		optf_net200 = geo.optf.OptFlow(geo.optf.PretrainedNet("/home/user/models/epoch50_resnet_200.pth")).cuda()
	optf_net.eval()
	geo.validate(optf_net, valid_loader, 1)

	optf_net150.eval()
	geo.validate(optf_net150, valid_loader, 1)
	
	optf_net200.eval()
	geo.validate(optf_net200, valid_loader, 1)		
