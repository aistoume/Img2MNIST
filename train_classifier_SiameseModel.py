from __future__ import division, print_function, absolute_import

#import tflearn
import numpy as np
from ReadJPGToNPArray import ConvertImg
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
from dataset import BatchGenerator, get_mnist
import tensorflow as tf
from model import *


flags.DEFINE_integer('batch_size', 2, 'Batch size.')
flags.DEFINE_integer('train_iter', 20, 'Total training iter')
flags.DEFINE_integer('step', 5, 'Save after ... iteration')

# Raw image path and convert image to MNIST-like dataset
train_directory = 'RawImage/train'
validation_directory = 'RawImage/validation'
test_directory = 'RawImage/test'

train = ConvertImg(train_directory)
validation = ConvertImg(validation_directory)
test = ConvertImg(test_directory)
MNData = base.Datasets(train=train, validation=validation, test=test)

# Train Model

gen = BatchGenerator(MNData.train.images, MNData.train.labels)
test_im = np.array([im.reshape((512,512,1)) for im in MNData.test.images])
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']


left = tf.placeholder(tf.float32, [None, 512, 512, 1], name='left')
right = tf.placeholder(tf.float32, [None, 512, 512, 1], name='right')
with tf.name_scope("similarity"):
	label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
	label = tf.to_float(label)
margin = 0.2

left_output = mynet(left, reuse=False)

right_output = mynet(right, reuse=True)

loss = contrastive_loss(left_output, right_output, label, margin)

global_step = tf.Variable(0, trainable=False)


# starter_learning_rate = 0.0001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
# tf.scalar_summary('lr', learning_rate)
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)


saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	#setup tensorboard	
	tf.summary.scalar('step', global_step)
	tf.summary.scalar('loss', loss)
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('train.log', sess.graph)

	#train iter
	for i in range(FLAGS.train_iter):
		# get left and right images and symbol. b_sim = 0 means images are the same; 1 means different
		# you have to change the files number in 'dataset.py'

		b_l, b_r, b_sim = gen.next_batch(FLAGS.batch_size)

		_, l, summary_str = sess.run([train_step, loss, merged], 
			feed_dict={left:b_l, right:b_r, label: b_sim})
		
		writer.add_summary(summary_str, i)
		print("\r#%d - Loss"%i, l)

		
		if (i + 1) % FLAGS.step == 0:
			#generate test
			feat = sess.run(left_output, feed_dict={left:test_im[:8]})
			
			labels = MNData.test.labels
			# plot result
			f = plt.figure(figsize=(16,9))
			
			# range of j is the same as total classes of your label
			# label = *,0 means images are the same, *,1 means different
			
			for j in range(2):   
				plt.plot(feat[labels==j, 0].flatten(), feat[labels==j, 1].flatten(),
					'*', c=c[j],alpha=0.8)
			plt.legend(['0', '1'])
			plt.savefig('img/%d.png' % (i + 1))

	saver.save(sess, "model/model.ckpt")




