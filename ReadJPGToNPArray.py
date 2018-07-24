from __future__ import division, print_function, absolute_import
import os
import logging
import numpy as np
from PIL import Image
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base

# The ConvertImg function open image saved in imgFolder
# Image size is set as 512*512 squared. If your raw image size differs to this one, you have to compress it first.

def ConvertImg(imgFolder):
	RawImgSize = (512,512)
	if os.path.isdir(imgFolder) is False:
		logging.warning('Raw image folder doesn\'t exist')
	train_directory = os.path.join(imgFolder)
	all_entries = os.listdir(train_directory)
	dirnames = []
	for entry in all_entries:
		if os.path.isdir(os.path.join(train_directory, entry)):
			dirnames.append(entry)
	
	arr = []
	label = []
	for dirname in dirnames:
		files = os.listdir(os.path.join(train_directory, dirname))
		
		for file in files:
			# read file as gray image
			img = Image.open(os.path.join(train_directory, dirname,file)).convert('L')
			if img.size[0] != RawImgSize[0] or img.size[1] != RawImgSize[1]:
				print('Error on Image Size != ', RawImgSize)
			else:
				# Label vector is generated from folder name. It add one label(folder name) to 'label'
				label.append(dirname)   
				for i in range(RawImgSize[0]):
					for j in range(RawImgSize[1]):
						pixel = float(img.getpixel((j, i)))
						arr.append(pixel)
						
	# 'arr' is 1D vector. reshape arr to #file * imageRow * imageCol * 1 numpy array.
	# Then combine with label by mnist default class 'DataSet'
	# return the MNIST-like dataset
	
	train_labels = np.array(label)
	train_images = np.array(arr).reshape((len(label),RawImgSize[0], RawImgSize[1],1))
	dtype=dtypes.float32
	reshape=True
	seed=None
	options = dict(dtype=dtype, reshape=reshape, seed=seed)
	mnData = mnist.DataSet(train_images, train_labels, **options)
	return mnData
	