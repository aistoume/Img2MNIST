import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import random
from tensorflow.examples.tutorials.mnist import input_data
from numpy.random import choice, permutation
from itertools import combinations


flags = tf.app.flags
FLAGS = flags.FLAGS


class BatchGenerator():
	def __init__(self, images, labels):
		np.random.seed(0)
		random.seed(0)
		self.labels = labels
		print(images.shape)
		self.images = images.reshape((10, 512, 512, 1))   # 10 is the # of images. You should change it manually.
		self.tot = len(labels)
		self.i = 0
		self.num_idx = dict()
		for idx, num in enumerate(self.labels):
			print(num)
			if num in self.num_idx:
				self.num_idx[num].append(idx)
			else:
				self.num_idx[num] = [idx]				
		self.to_img = lambda x: self.images[x]

	def next_batch(self, batch_size):
		left = []
		right = []
		sim = []
		# genuine
		for i in range(2):
			n = 1
			l = choice(self.num_idx[i], n*2, replace=False).tolist()
			left.append(self.to_img(l.pop()))
			right.append(self.to_img(l.pop()))
			sim.append([0])
			
		#impostor
		for i,j in combinations(range(2), 2):
			left.append(self.to_img(choice(self.num_idx[i])))
			right.append(self.to_img(choice(self.num_idx[j])))
			sim.append([1])
		return np.array(left), np.array(right), np.array(sim)
		

def get_mnist():
	mnist = input_data.read_data_sets("MNIST_data/")
	return mnist
