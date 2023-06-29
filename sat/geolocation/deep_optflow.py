import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torchvision.models import VGG16_Weights, ResNet50_Weights
import io
import requests
from PIL import Image
from torch.autograd import Variable
from torch.nn.functional import grid_sample
from pdb import set_trace as st
from sys import argv
import argparse
import time
from math import cos, sin, pi, sqrt
import sys
import time

USE_CUDA = torch.cuda.is_available()

class BatchInversion(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        batch_size, h, w = input.size()
        assert(h == w)
        H = torch.Tensor(batch_size, h, h).type_as(input)
        for i in range(0, batch_size):
            H[i, :, :] = input[i, :, :].inverse()
        self.H = H
        return H
    @staticmethod
    def backward(self, grad_output):
        H = self.H
        [batch_size, h, w] = H.size()
        assert(h == w)
        Hl = H.transpose(1,2).repeat(1, 1, h).reshape(batch_size*h*h, h, 1)
        Hr = H.repeat(1, h, 1).reshape(batch_size*h*h, 1, h)
        r = Hl.bmm(Hr).reshape(batch_size, h, h, h, h) * \
            grad_output.contiguous().reshape(batch_size, 1, 1, h, h).expand(batch_size, h, h, h, h)
        return -r.sum(-1).sum(-1)

class GradientFunc(nn.Module):

	def __init__(self):
		super(GradientFunc, self).__init__()
		wx = torch.FloatTensor([-.5, 0, .5]).reshape(1, 1, 1, 3)
		wy = torch.FloatTensor([[-.5], [0], [.5]]).reshape(1, 1, 3, 1)
		self.register_buffer('wx', wx)
		self.register_buffer('wy', wy)
		self.padx_func = torch.nn.ReplicationPad2d((1,1,0,0))
		self.pady_func = torch.nn.ReplicationPad2d((0,0,1,1))

	def forward(self, img):
		batch_size, k, h, w = img.size()
		img_ = img.reshape(batch_size * k, h, w)
		img_ = img_.unsqueeze(1)
		img_padx = self.padx_func(img_)
		img_dx = torch.nn.functional.conv2d(input=img_padx,
											weight=self.wx,
											stride=1,
											padding=0).squeeze(1)
		img_pady = self.pady_func(img_)
		img_dy = torch.nn.functional.conv2d(input=img_pady,
											weight=self.wy,
											stride=1,
											padding=0).squeeze(1)
		img_dx = img_dx.reshape(batch_size, k, h, w)
		img_dy = img_dy.reshape(batch_size, k, h, w)
		if not isinstance(img, torch.Tensor):
			img_dx = img_dx.data
			img_dy = img_dy.data
		return img_dx, img_dy

def normalize_batch(img):
	# per-channel zero-mean and unit-variance of image batch
	N, C, H, W = img.size()
	img_vec = img.reshape(N, C, H * W, 1)
	mean = img_vec.mean(dim=2, keepdim=True)
	img_ = img - mean
	std_dev = img_vec.std(dim=2, keepdim=True)
	img_ = img_ / std_dev
	return img_

def mesh_grid(x, y):
	imW = x.size(0)
	imH = y.size(0)
	x = x - x.max()/2
	y = y - y.max()/2
	X = x.unsqueeze(0).repeat(imH, 1)
	Y = y.unsqueeze(1).repeat(1, imW)
	return X, Y


def bilinear_sampling(A, x, y):
	batch_size, k, h, w = A.size()
	x_norm = x/((w-1)/2) - 1
	y_norm = y/((h-1)/2) - 1
	grid = torch.cat((x_norm.reshape(batch_size, h, w, 1), y_norm.reshape(batch_size, h, w, 1)), 3)
	Q = grid_sample(A, grid, mode='bilinear', align_corners=True)
	if isinstance(A, torch.Tensor):
		if USE_CUDA:
			in_view_mask = ((x_norm.data > -1+2/w) & (x_norm.data < 1-2/w) & (y_norm.data > -1+2/h) & (y_norm.data < 1-2/h)).type_as(A.data).cuda()
		else:
			in_view_mask = ((x_norm.data > -1+2/w) & (x_norm.data < 1-2/w) & (y_norm.data > -1+2/h) & (y_norm.data < 1-2/h)).type_as(A.data)
	else:
		in_view_mask = ((x_norm > -1+2/w) & (x_norm < 1-2/w) & (y_norm > -1+2/h) & (y_norm < 1-2/h)).type_as(A)
		Q = Q.data

	return Q.reshape(batch_size, k, h, w), in_view_mask

def warp_with_homography(img, p):
	# perform warping of img batch using homography transform with batch of parameters p
	# img [in, Tensor N x C x H x W] : batch of images to warp
	# p [in, Tensor N x 8 x 1] : batch of warp parameters
	# img_warp [out, Tensor N x C x H x W] : batch of warped images
	# mask [out, Tensor N x H x W] : batch of binary masks indicating valid pixels areas

	batch_size, k, h, w = img.size()
	if isinstance(img, torch.Tensor):
		if USE_CUDA:
			x = torch.arange(w).cuda()
			y = torch.arange(h).cuda()
		else:
			x = torch.arange(w)
			y = torch.arange(h)
	else:
		x = torch.arange(w)
		y = torch.arange(h)

	X, Y = mesh_grid(x, y)

	H = param_to_hmg(p)
	if isinstance(img, torch.Tensor):
		if USE_CUDA:
		# create xy matrix, 2 x N
			xy = torch.cat((X.reshape(1, X.numel()), Y.reshape(1, Y.numel()), torch.ones(1, X.numel()).cuda()), 0)
		else:
			xy = torch.cat((X.reshape(1, X.numel()), Y.reshape(1, Y.numel()), torch.ones(1, X.numel())), 0)
	else:
		xy = torch.cat((X.reshape(1, X.numel()), Y.reshape(1, Y.numel()), torch.ones(1, X.numel())), 0)
	xy = xy.repeat(batch_size, 1, 1)
	xy_warp = H.bmm(xy)
	# extract warped X and Y, normalizing the homog coordinates
	X_warp = xy_warp[:,0,:] / xy_warp[:,2,:]
	Y_warp = xy_warp[:,1,:] / xy_warp[:,2,:]
	X_warp = X_warp.reshape(batch_size,h,w) + (w-1)/2
	Y_warp = Y_warp.reshape(batch_size,h,w) + (h-1)/2
	img_warp, mask = bilinear_sampling(img, X_warp, Y_warp)
	return img_warp, mask


def param_to_hmg(p):
	# batch parameters to batch homography
	batch_size, _, _ = p.size()
	if isinstance(p, torch.Tensor):
		if USE_CUDA:
			z = torch.zeros(batch_size, 1, 1).cuda()
		else:
			z = torch.zeros(batch_size, 1, 1)
	else:
		z = torch.zeros(batch_size, 1, 1)
	p_ = torch.cat((p, z), 1)
	if isinstance(p, torch.Tensor):
		if USE_CUDA:
			I = torch.eye(3,3).repeat(batch_size, 1, 1).cuda()
		else:
			I = torch.eye(3,3).repeat(batch_size, 1, 1)
	else:
		I = torch.eye(3,3).repeat(batch_size, 1, 1)

	H = p_.reshape(batch_size, 3, 3) + I
	return H

def hmg_to_param(H):
	# batch homography to batch parameters
	batch_size, _, _ = H.size()
	if isinstance(H, torch.Tensor):
		if USE_CUDA:
			I = torch.eye(3,3).repeat(batch_size, 1, 1).cuda()
		else:
			I = torch.eye(3,3).repeat(batch_size, 1, 1)
	else:
		I = torch.eye(3,3).repeat(batch_size, 1, 1)
	p = H - I
	p = p.reshape(batch_size, 9, 1)
	p = p[:, 0:8, :]
	return p

class Vgg16(nn.Module):
	def __init__(self, model_path):
		super(Vgg16, self).__init__()

		print('Loading pretrained network...',end='')
		if model_path == "":
			vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)#torch.load(model_path)
			vgg16
		else:
			vgg16 = torch.load(model_path)
			vgg16
		print('done')

		self.features = nn.Sequential(
			*(list(vgg16.features.children())[0:15]),
		)

		# freeze conv1, conv2
		#for p in self.parameters():
		#	if p.size()[0] < 256:
		#		p.requires_grad=False

		'''
	    (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (1): ReLU(inplace)
	    (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (3): ReLU(inplace)
	    (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (6): ReLU(inplace)
	    (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (8): ReLU(inplace)
	    (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (11): ReLU(inplace)
	    (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (13): ReLU(inplace)
	    (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (15): ReLU(inplace)
	    (16): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (17): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (18): ReLU(inplace)
	    (19): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (20): ReLU(inplace)
	    (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (22): ReLU(inplace)
	    (23): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (24): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (25): ReLU(inplace)
	    (26): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (27): ReLU(inplace)
	    (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (29): ReLU(inplace)
	    (30): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    '''

	def forward(self, x):
		# print('CNN stage...',end='')
		x = self.features(x)
		# print('done')
		return x
    
def flatten_model(modules):
    def flatten_list(_2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list

    ret = []
    try:
        for _, n in modules:
            ret.append(loopthrough(n))
    except:
        try:
            if str(modules._modules.items()) == "odict_items([])":
                ret.append(modules)
            else:
                for _, n in modules._modules.items():
                    ret.append(loopthrough(n))
        except:
            ret.append(modules)
    return flatten_list(ret)
       
class Resnet50(nn.Module):
	def __init__(self, model_path):
		super(Resnet50, self).__init__()

		print('Loading pretrained network...',end='')
		if model_path == "":
			resnet50 = torchvision.models.resnet50(ResNet50_Weights.IMAGENET1K_V2)#weights=None
			
		else:
			resnet50 = torch.load(model_path)
			
		print('done')

		self.features = nn.Sequential(
			*(list(resnet50.children())[:-2]),
		)

		target_layers =[]
		module_list = [module for module in self.modules()] # this is needed
		flatted_list= flatten_model(module_list)
		conv_count = 0
		for count, value in enumerate(flatted_list): 
			if isinstance(value, (nn.Conv2d,nn.AvgPool2d,nn.BatchNorm2d)):
		#		for p in value.parameters():
		#			p.requires_grad = True
				if isinstance(value, (nn.Conv2d)):
					conv_count = conv_count + 1
			if (conv_count >= 45):
				for p in value.parameters():
					p.requires_grad = True
			#else:
			#	for p in value.parameters():
			#		p.requires_grad = False
		#	print(count, conv_count, value)
			target_layers.append(value)
		#print(target_layers)

	def forward(self, x):
		# print('CNN stage...',end='')
		x = self.features(x)
		# print('done')
		return x
		
		
class PretrainedNet(nn.Module):
	def __init__(self, model_path):
		super(PretrainedNet, self).__init__()

		print('Loading pretrained network...',end='')
		self.custom = torch.load(model_path, map_location=lambda storage, loc: storage)
		print('done')

	def forward(self, x):
		x = self.custom(x)
		return x


class OptFlow(nn.Module):
	def __init__(self, conv_net):
		super(OptFlow, self).__init__()
		self.img_gradient_func = GradientFunc()
		self.conv_func = conv_net
		self.inv_func = BatchInversion()

	def forward(self, img, temp, init_param=None, tol=1e-3, max_itr=500, conv_flag=0, ret_itr=False):
		if conv_flag:
			start = time.time()
			Ft = self.conv_func(temp)
			stop = time.time()
			Fi = self.conv_func(img)
		else:
			Fi = img
			Ft = temp
		batch_size, k, h, w = Ft.size()
		Ftgrad_x, Ftgrad_y = self.img_gradient_func(Ft)
		dIdp = self.compute_dIdp(Ftgrad_x, Ftgrad_y)
		dIdp_t = dIdp.transpose(1, 2)
		invH = self.inv_func.apply(dIdp_t.bmm(dIdp))
		invH_dIdp = invH.bmm(dIdp_t)
		if USE_CUDA:
			if init_param is None:
				p = torch.zeros(batch_size, 8, 1).cuda()
			else:
				p = init_param
			dp = torch.ones(batch_size, 8, 1).cuda() # ones so that the norm of each dp is larger than tol for first iteration
		else:
			if init_param is None:
				p = torch.zeros(batch_size, 8, 1)
			else:
				p = init_param
			dp = torch.ones(batch_size, 8, 1)
		itr = 1
		r_sq_dist_old = 0
		while (float(dp.norm(p=2,dim=1,keepdim=True).max()) > tol or itr == 1) and (itr <= max_itr):
			Fi_warp, mask = warp_with_homography(Fi, p)
			mask.unsqueeze_(1)
			mask = mask.repeat(1, k, 1, 1)
			Ft_mask = Ft.mul(mask)
			r = Fi_warp - Ft_mask
			r = r.reshape(batch_size, k * h * w, 1)
			dp_new = invH_dIdp.bmm(r)
			dp_new[:,6:8,0] = 0
			if USE_CUDA:
				dp = (dp.norm(p=2,dim=1,keepdim=True) > tol).type(torch.FloatTensor).cuda() * dp_new
			else:
				dp = (dp.norm(p=2,dim=1,keepdim=True) > tol).type(torch.FloatTensor) * dp_new
			p = p - dp
			itr = itr + 1
		#print('finished at iteration ', itr)
		if (ret_itr):
			return p, param_to_hmg(p), itr
		else:
			return p, param_to_hmg(p)

	def compute_dIdp(self, Ftgrad_x, Ftgrad_y):
		batch_size, k, h, w = Ftgrad_x.size()
		x = torch.arange(w)
		y = torch.arange(h)
		X, Y = mesh_grid(x, y)
		X = X.reshape(X.numel(), 1)
		Y = Y.reshape(Y.numel(), 1)
		X = X.repeat(batch_size, k, 1)
		Y = Y.repeat(batch_size, k, 1)
		if USE_CUDA:
			X = X.cuda()
			Y = Y.cuda()
		#else:
		#	X = Variable(X)
		#	Y = Variable(Y)
		Ftgrad_x = Ftgrad_x.reshape(batch_size, k * h * w, 1)
		Ftgrad_y = Ftgrad_y.reshape(batch_size, k * h * w, 1)
		dIdp = torch.cat((
			X.mul(Ftgrad_x), 
			Y.mul(Ftgrad_x),
			Ftgrad_x,
			X.mul(Ftgrad_y),
			Y.mul(Ftgrad_y),
			Ftgrad_y,
			-X.mul(X).mul(Ftgrad_x) - X.mul(Y).mul(Ftgrad_y),
			-X.mul(Y).mul(Ftgrad_x) - Y.mul(Y).mul(Ftgrad_y)),2)
		# dIdp size = batch_size x k*h*w x 8
		return dIdp















