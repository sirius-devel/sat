import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import pyplot as plt
import pysift
import time

def get_homography(img_batch, template_batch, image_sz):
	template = template_batch.data.squeeze(0).cpu().numpy()
	img = img_batch.data.squeeze(0).cpu().numpy()

	if template.shape[0] == 3:
		template = np.swapaxes(template, 0, 2)
		template = np.swapaxes(template, 0, 1)
		img = np.swapaxes(img, 0, 2)
		img = np.swapaxes(img, 0, 1)

		template = (template * 255).astype('uint8')
		img = (img * 255).astype('uint8')


	template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	start = time.time()
	kp1, des1 = pysift.computeKeypointsAndDescriptors(template_gray)
	print("Time elapsed for first image= ", time.time() - start)
	start = time.time()
	kp2, des2 = pysift.computeKeypointsAndDescriptors(img_gray)
	print("Time elapsed for second image= ", time.time() - start)

	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	matches = bf.match(des1,des2)

	if (len(kp1) >= 2) and (len(kp2) >= 2):

		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1,des2,k=2)

		# store all the good matches as per Lowe's ratio test
		good = []
		for m,n in matches:
		    if m.distance < 0.7*n.distance:
		        good.append(m)
		print("good.size()=",len(good))
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		
		src_pts = src_pts - image_sz/2
		dst_pts = dst_pts - image_sz/2
		print("src.size()=",src_pts.size)
		print("dst.size()=",dst_pts.size)
		if (src_pts.size < 4) or (dst_pts.size < 4) or (len(good) <4):
			H_found = np.eye(3)
		else:
			H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

		if H_found is None:
			H_found = np.eye(3)

	else:
		H_found = np.eye(3)

	H = torch.from_numpy(H_found).float()
	I = torch.eye(3,3)

	p = H - I

	p = p.reshape(1, 9, 1)
	p = p[:, 0:8, :]

	if torch.cuda.is_available():
		return p.cuda()
	else:
		return p












