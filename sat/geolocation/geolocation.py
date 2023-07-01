import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import io
import requests
from PIL import Image, ImageFont, ImageDraw
from torch.nn.functional import grid_sample
import pdb
from sys import argv
import argparse
import os
import random
import deep_optflow as optf
import glob
from math import cos, sin, pi, sqrt
import time
import sys
import gc
import numpy as np
import ransac_algorithm as ralg
import argparse
import csv
from torchvision import models, transforms
from torchvision.models import VGG16_Weights
import pandas as pd

# size scale range
min_scale = 0.75
max_scale = 1.25

# rotation range (-angle_range, angle_range)
angle_range = 15 # degrees

# projective variables (p7, p8)
projective_range = 0

# translation (p3, p6)
translation_range = 10 # pixels

# possible segment sizes
lower_sz = 200 # pixels, square
upper_sz = 220

# amount to pad when cropping segment, as ratio of size, on all 4 sides
warp_pad = 0.4

# normalized size of all training pairs
training_sz = 175
training_sz_pad = round(training_sz + training_sz * 2 * warp_pad)

USE_CUDA = torch.cuda.is_available()

###---

def generate_data(batch_size, datapath, folder):
	# create batch of normalized training pairs
	# batch_size [in, int] : number of pairs
	# img_batch [out, Tensor N x 3 x training_sz x training_sz] : batch of images
	# template_batch [out, Tensor N x 3 x training_sz x training_sz] : batch of templates
	# param_batch [out, Tensor N x 8 x 1] : batch of ground truth warp parameters
	# randomly choose 2 aligned images
	FOLDERPATH = datapath + folder
	FOLDERPATH = FOLDERPATH + 'images/'
	images_dir = glob.glob(FOLDERPATH + '*.png')
	random.shuffle(images_dir)
	img = Image.open(images_dir[0])
	template = Image.open(images_dir[1])
	in_W, in_H = img.size
	# pdb.set_trace()
	# initialize output tensors
	if USE_CUDA:
		img_batch = torch.zeros(batch_size, 3, training_sz, training_sz).cuda()
		template_batch = torch.zeros(batch_size, 3, training_sz, training_sz).cuda()
		param_batch = torch.zeros(batch_size, 8, 1).cuda()
	else:
		img_batch = torch.zeros(batch_size, 3, training_sz, training_sz)
		template_batch = torch.zeros(batch_size, 3, training_sz, training_sz)
		param_batch = torch.zeros(batch_size, 8, 1)
	for i in range(batch_size):
		# randomly choose size and top left corner of image for sampling
		seg_sz = random.randint(lower_sz, upper_sz)
		seg_sz_pad = round(seg_sz + seg_sz * 2 * warp_pad)
		loc_x = random.randint(0, (in_W - seg_sz_pad) - 1)
		loc_y = random.randint(0, (in_H - seg_sz_pad) - 1)
		img_seg_pad = img.crop((loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad))
		img_seg_pad = img_seg_pad.resize((training_sz_pad, training_sz_pad))
		template_seg_pad = template.crop((loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad))
		template_seg_pad = template_seg_pad.resize((training_sz_pad, training_sz_pad))
		if USE_CUDA:
			img_seg_pad = transforms.ToTensor()(img_seg_pad).cuda()
			template_seg_pad = transforms.ToTensor()(template_seg_pad).cuda()
		else:
			img_seg_pad = transforms.ToTensor()(img_seg_pad)
			template_seg_pad = transforms.ToTensor()(template_seg_pad)
		# create random ground truth
		scale = random.uniform(min_scale, max_scale)
		angle = random.uniform(-angle_range, angle_range)
		projective_x = random.uniform(-projective_range, projective_range)
		projective_y = random.uniform(-projective_range, projective_range)
		translation_x = random.uniform(-translation_range, translation_range)
		translation_y = random.uniform(-translation_range, translation_range)
		rad_ang = angle / 180 * pi
		if USE_CUDA:
			p_gt = torch.Tensor([scale + cos(rad_ang) - 2,
									   -sin(rad_ang),
									   translation_x,
									   sin(rad_ang),
									   scale + cos(rad_ang) - 2,
									   translation_y,
									   projective_x, 
									   projective_y]).cuda()
		else:
			p_gt = torch.Tensor([scale + cos(rad_ang) - 2,
									   -sin(rad_ang),
									   translation_x,
									   sin(rad_ang),
									   scale + cos(rad_ang) - 2,
									   translation_y,
									   projective_x, 
									   projective_y])
		p_gt = p_gt.reshape(8,1)
		p_gt = p_gt.repeat(1,1,1)
		img_seg_pad_w, _ = optf.warp_with_homography(img_seg_pad.unsqueeze(0), optf.hmg_to_param(optf.BatchInversion.apply(optf.param_to_hmg(p_gt))))
		img_seg_pad_w.squeeze_(0)
		pad_side = round(training_sz * warp_pad)
		img_seg_w = img_seg_pad_w[:,
							pad_side : pad_side + training_sz,
							pad_side : pad_side + training_sz]

		template_seg = template_seg_pad[:,
							pad_side : pad_side + training_sz,
							pad_side : pad_side + training_sz]

		img_batch[i, :, :, :] = img_seg_w
		template_batch[i, :, :, :] = template_seg
		param_batch[i, :, :] = p_gt[0, :, :].data
	return img_batch, template_batch, param_batch
    

def calc_corner_loss(p, p_gt):
	# p [in, torch tensor] : batch of regressed warp parameters
	# p_gt [in, torch tensor] : batch of gt warp parameters
	# loss [out, float] : sum of corner loss over minibatch
	batch_size, _, _ = p.size()
	# compute corner loss
	H_p = optf.param_to_hmg(p)
	H_gt = optf.param_to_hmg(p_gt)
	if USE_CUDA:
		corners = torch.Tensor([[-training_sz_pad/2, training_sz_pad/2, training_sz_pad/2, -training_sz_pad/2],
								[-training_sz_pad/2, -training_sz_pad/2, training_sz_pad/2, training_sz_pad/2],
								[1, 1, 1, 1]]).cuda()
	else:
		corners = torch.Tensor([[-training_sz_pad/2, training_sz_pad/2, training_sz_pad/2, -training_sz_pad/2],
								[-training_sz_pad/2, -training_sz_pad/2, training_sz_pad/2, training_sz_pad/2],
								[1, 1, 1, 1]])

	corners = corners.repeat(batch_size, 1, 1)
	corners_w_p = H_p.bmm(corners)
	corners_w_gt = H_gt.bmm(corners)
	corners_w_p = corners_w_p[:, 0:2, :] / corners_w_p[:, 2:3, :]
	corners_w_gt = corners_w_gt[:, 0:2, :] / corners_w_gt[:, 2:3, :]
	loss = ((corners_w_p - corners_w_gt) ** 2).sum()
	return loss

def test_optflow_model(DATASET_PATH, MODEL1_PATH, MODEL2_PATH, TEST_DATA_SAVE_PATH):
	if USE_CUDA:
		optf_vgg16 = optf.OptFlow(optf.Vgg16("")).cuda()
		optf_resnet50 = optf.OptFlow(optf.PretrainedNet(MODEL2_PATH)).cuda()
		optf_trained = optf.OptFlow(optf.PretrainedNet(MODEL1_PATH)).cuda()
	else:
		optf_vgg16 = optf.OptFlow(optf.Vgg16(""))
		optf_resnet50 = optf.OptFlow(optf.PretrainedNet(MODEL2_PATH))
		optf_trained = optf.OptFlow(optf.PretrainedNet(MODEL1_PATH))

	testbatch_sz = 1 # keep as 1 in order to compute corner error accurately
	test_num = 500
	test_dataset = SatImageDataset(DATASET_PATH + "/test", test_num)
	test_loader = DataLoader(test_dataset, batch_size = testbatch_sz, shuffle = True, pin_memory=True)
	print('Testing...')
	print('TEST DATA SAVE PATH: ', TEST_DATA_SAVE_PATH)
	print('MODEL1 PATH: ', MODEL1_PATH)
	print('MODEL2_PATH', MODEL2_PATH)
	print('USE CUDA: ', USE_CUDA)
	print('min_scale: ',  min_scale)
	print('max_scale: ', max_scale)
	print('angle_range: ', angle_range)
	print('projective_range: ', projective_range)
	print('translation_range: ', translation_range)
	print('lower_sz: ', lower_sz)
	print('upper_sz: ', upper_sz)
	print('warp_pad: ', warp_pad)
	print('test batch size: ', testbatch_sz, ' number of test images: ', test_num)
	test_results = np.zeros((test_num, 4), dtype=float)
		
	if USE_CUDA:
		test_img_batch = torch.zeros(testbatch_sz, 3, training_sz, training_sz).cuda()
		test_template_batch = torch.zeros(testbatch_sz, 3, training_sz, training_sz).cuda()
		test_param_batch = torch.zeros(testbatch_sz, 8, 1).cuda()
	else:
		test_img_batch = torch.zeros(testbatch_sz, 3, training_sz, training_sz)
		test_template_batch = torch.zeros(testbatch_sz, 3, training_sz, training_sz)
		test_param_batch = torch.zeros(testbatch_sz, 8, 1)
	test_train_loss = 0
	test_resnet_loss = 0
	test_vgg_loss = 0
	test_ralg_loss = 0
	test_batches = 0
	optf_vgg16.eval()
	optf_resnet50.eval()
	optf_trained.eval()
	i = 0
	for img_test_data, template_test_data, param_test_data in test_loader:
		test_batches = test_batches + 1
		if USE_CUDA:
			test_img_batch = optf.normalize_batch(img_test_data).cuda()
			test_template_batch = optf.normalize_batch(template_test_data).cuda()
			param_test_data = param_test_data.cuda()
		else:
			test_img_batch = optf.normalize_batch(img_test_data)
			test_template_batch = optf.normalize_batch(template_test_data)
		#forward pass
		trained_param, _ = optf_trained(test_img_batch, test_template_batch, tol = 1e-3, max_itr = 1, conv_flag = 1)
		trained_loss = calc_corner_loss(trained_param, param_test_data)
		resnet50_param, _ = optf_resnet50(test_img_batch, test_template_batch, tol = 1e-3, max_itr = 1, conv_flag = 1)
		resnet50_loss = calc_corner_loss(resnet50_param, param_test_data)
		
		vgg_param, _ = optf_vgg16(test_img_batch, test_template_batch, tol=1e-3, max_itr=1, conv_flag=1)
		vgg_loss = calc_corner_loss(vgg_param, param_test_data)
		
		ralg_param = ralg.get_homography(test_img_batch, test_template_batch, training_sz)
		ralg_loss = calc_corner_loss(ralg_param, param_test_data)
		
		test_results[i, 0] = trained_loss
		test_results[i, 1] = vgg_loss
		test_results[i, 2] = ralg_loss
		test_results[i, 3] = resnet50_loss
		#print('test: ', i, 
		#	' trained loss: ', round(float(trained_loss), 2),
		#	' vgg16 loss: ', round(float(vgg_loss), 2),
		#	' ralg loss: ', round(float(ralg_loss), 2),
		#	' resnet50 loss', round(float(resnet50_loss), 2))
		#sys.stdout.flush()		
		test_train_loss += float(trained_loss)
		test_vgg_loss += float(vgg_loss)
		test_ralg_loss += float(ralg_loss)
		test_resnet_loss += float(resnet50_loss)
		i = i + 1
	test_train_loss = float(test_train_loss/len(test_loader))
	test_vgg_loss = float(test_vgg_loss/len(test_loader))
	test_ralg_loss = float(test_vgg_loss/len(test_loader))
	test_resnet_loss = float(test_resnet_loss/len(test_loader))
	print('Average test train loss: ', float(test_train_loss), " average vgg loss: ", float(test_vgg_loss), 
		" average ralg loss: ", float(test_ralg_loss), " average resnet loss: ", float(test_resnet_loss))
	np.savetxt(TEST_DATA_SAVE_PATH, test_results, delimiter=',')

class SatImageDataset(Dataset):
	def __init__(self, dataset_path, generated_img_count):
		super(Dataset, self).__init__()
		self.dataset_path = dataset_path
		self.img_count = generated_img_count


	def __len__(self):
		return self.img_count

	def __getitem__(self, index):
		transform_path = self.dataset_path + "/" + str(index) + "_transform.jpg"
		template_path = self.dataset_path + "/" + str(index) + "_template.jpg"
		parameters_path = self.dataset_path + "/" + str(index) + "_params.csv"
		transform_img = Image.open(transform_path)
		template_img = Image.open(template_path)
		transform_img_tensor = transforms.ToTensor()(transform_img)
		template_img_tensor = transforms.ToTensor()(template_img)
		params = pd.read_csv(parameters_path)
		params_tensor = torch.from_numpy(np.genfromtxt(parameters_path, delimiter=",", dtype=float)).float()
		params_tensor = params_tensor.unsqueeze(0)
		params_tensor = params_tensor.reshape(8, 1)
		return transform_img_tensor, template_img_tensor, params_tensor


def validate(optf_net, valid_loader, validbatch_sz):
	val_loss = 0
	valid_batches = 0
	if USE_CUDA:
		valid_img_batch = torch.zeros(validbatch_sz, 3, training_sz, training_sz).cuda()
		valid_template_batch = torch.zeros(validbatch_sz, 3, training_sz, training_sz).cuda()
		param_valid_data = torch.zeros(validbatch_sz, 8, 1).cuda()
	else:
		valid_img_batch = torch.zeros(validbatch_sz, 3, training_sz, training_sz)
		valid_template_batch = torch.zeros(validbatch_sz, 3, training_sz, training_sz)
		param_valid_data = torch.zeros(validbatch_sz, 8, 1)
		
	for img_valid_data, template_valid_data, param_valid_data in valid_loader:
		valid_batches = valid_batches + 1
		if USE_CUDA:
			valid_img_batch = optf.normalize_batch(img_valid_data).cuda()
			valid_template_batch = optf.normalize_batch(template_valid_data).cuda()
			param_valid_data = param_valid_data.cuda()
		else:
			valid_img_batch = optf.normalize_batch(img_valid_data)
			valid_template_batch = optf.normalize_batch(template_valid_data)
		#forward pass of training minibatch through 
		optf_valid_param_batch, _ = optf_net(valid_img_batch, valid_template_batch, tol = 1e-3, max_itr = 1, conv_flag = 1)
		valid_loss = calc_corner_loss(optf_valid_param_batch, param_valid_data)
		v_loss = valid_loss.item()
		#print('batch validation loss: ', v_loss)
		val_loss += v_loss
	val_loss = float(val_loss/len(valid_loader))
	print('Average validation loss: ', float(val_loss))
	
	
def train_optflow_model(MODEL_PATH, DATASET_PATH, RESNET):

	minibatch_sz = 20
	validbatch_sz = 1
	train_dataset = SatImageDataset(DATASET_PATH + "/train", 25000)
	train_loader = DataLoader(train_dataset, batch_size = minibatch_sz, shuffle = True, pin_memory=True)
	valid_dataset = SatImageDataset(DATASET_PATH + "/valid", 5000)
	valid_loader = DataLoader(valid_dataset, batch_size = validbatch_sz, shuffle = True, pin_memory=True)

	if USE_CUDA:
		if (RESNET):
			optf_net = optf.OptFlow(optf.Resnet50("")).cuda()
		else:
			optf_net = optf.OptFlow(optf.Vgg16("")).cuda()
		#optf_net = optf.OptFlow(optf.PretrainedNet("/home/user/sat/models/trained_model_output.pth")).cuda()
	else:
		if (RESNET):
			optf_net = optf.OptFlow(optf.Resnet50(""))
		else:
			optf_net = optf.OptFlow(optf.Vgg(""))
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, optf_net.conv_func.parameters()), lr=0.001)
	
	print('Training...')
	print('MODEL_PATH: ', MODEL_PATH)
	print('USE CUDA: ', USE_CUDA)
	print('min_scale: ',  min_scale)
	print('max_scale: ', max_scale)
	print('angle_range: ', angle_range)
	print('projective_range: ', projective_range)
	print('translation_range: ', translation_range)
	print('lower_sz: ', lower_sz)
	print('upper_sz: ', upper_sz)
	print('warp_pad: ', warp_pad)
	print('training_sz: ', training_sz_pad)

	if USE_CUDA:
		img_train_data = torch.zeros(minibatch_sz, 3, training_sz, training_sz).cuda()
		template_train_data = torch.zeros(minibatch_sz, 3, training_sz, training_sz).cuda()
		param_train_data = torch.zeros(minibatch_sz, 8, 1).cuda()
	else:
		img_train_data = torch.zeros(minibatch_sz, 3, training_sz, training_sz)
		template_train_data = torch.zeros(minibatch_sz, 3, training_sz, training_sz)
		param_train_data = torch.zeros(minibatch_sz, 8, 1)
	
	print_every = 100
	for epoch in range(50):
		print("Epoch ", epoch)
		batches = 0
		running_loss = 0
		start = time.time()
		optf_net.train()
		for img_train_data, template_train_data, param_train_data in train_loader:
			batches = batches + 1
			optimizer.zero_grad()
			if USE_CUDA:
				training_img_batch = optf.normalize_batch(img_train_data).cuda()
				training_template_batch = optf.normalize_batch(template_train_data).cuda()
				param_train_data = param_train_data.cuda()
			else:
				training_img_batch = optf.normalize_batch(img_train_data)
				training_template_batch = optf.normalize_batch(template_train_data)
			# forward pass of training minibatch through 
			optf_param_batch, _ = optf_net(training_img_batch, training_template_batch, tol=0.001, max_itr=1, conv_flag=1)
			loss = calc_corner_loss(optf_param_batch, param_train_data)
			b_loss = loss.item()
			loss.backward()
			optimizer.step()
			running_loss += b_loss
			if batches % print_every == 0:
				print(f"{time.asctime()}.."
				f"Time elapsed = {time.time() - start:.3f}.."
				f"Batch {batches}/{len(train_loader)}.."
				f"Average training loss on batch:{running_loss/(batches):.3f}.."
				f"Batch training loss: {b_loss:.3f}.."
				)
		torch.save(optf_net.conv_func, MODEL_PATH)
		start = time.time()
		gc.collect()
		sys.stdout.flush()
		if epoch % 5 == 0:
			optf_net.eval()
			validate(optf_net, valid_loader, validbatch_sz)
		end = time.time()
		


