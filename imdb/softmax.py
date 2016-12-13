from __future__ import absolute_import
from __future__ import division
from TFDataSet import *

import argparse

def train_soft(data_set):

	import tensorflow as tf
	sess = tf.InteractiveSession()

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/data',
	                  help='Directory for storing data')
	FLAGS = parser.parse_args()

	#mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	###Create the model
	# Placeholders
	x = tf.placeholder(tf.float32, [None, 300])
	y_ = tf.placeholder(tf.float32, [None, 2])

	# Variables and regression model
	W = tf.Variable(tf.zeros([300, 2]))
	b = tf.Variable(tf.zeros([2]))
	y = tf.matmul(x, W) + b

	# Define loss and optimizer
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	# Train
	tf.initialize_all_variables().run()

	for _ in range(1000):
		batch_xs, batch_ys = data_set.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	#Prediction
	prediction = tf.argmax(y, 1)
	return sess.run(prediction, feed_dict={x: data_set.test._vals})