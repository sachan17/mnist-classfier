# -*- coding: utf-8 -*-
# @Author: Prateek Sachan
# @Date:   2017-02-21 21:12:01
# @Last Modified by:   Prateek Sachan
# @Last Modified time: 2017-02-21 21:28:30

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST-data/", one_hot = True)

image_size = 28
classes = 10

train_data, train_labels = mnist.train.images, mnist.train.labels
test_data, test_labels = mnist.test.images, mnist.test.labels

valid_data = train_data[50000:55000, :]
valid_labels = train_labels[50000:55000]

train_data = train_data[:50000, :]
train_labels = train_labels[:50000]

#-----------1 Hidden layer---------
# using relu for activation
# sgd implementation for fast result

hidden_layer_size = 1024

graph = tf.Graph()
with graph.as_default():
	# placeholders
	tf_train_data = tf.placeholder(tf.float32, shape = (None, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape = (None, classes))
	tf_valid_data = tf.constant(valid_data)
	tf_test_data = tf.constant(test_data)

	#variables
	weigths = [
				tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layer_size])),
				tf.Variable(tf.truncated_normal([hidden_layer_size, classes]))
			]
	biases = [
				tf.Variable(tf.zeros([hidden_layer_size])),
				tf.Variable(tf.zeros([classes]))
			]  

	#training computations
	hidden_layer = tf.matmul(tf_train_data, weigths[0]) + biases[0]
	hidden_layer = tf.nn.relu(hidden_layer)

	output = tf.matmul(hidden_layer, weigths[1]) + biases[1]

	#loss
	beta = [5e-4, 5e-4]
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = output) 
	loss = tf.reduce_mean(cross_entropy)

	# optimizer
	learning_rate = 0.1
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	train_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_train_data, weigths[0]) + biases[0]), weigths[1]) + biases[1])
	valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_data, weigths[0]) + biases[0]), weigths[1]) + biases[1])
	test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_data, weigths[0]) + biases[0]), weigths[1]) + biases[1])

def accuracy(prediction, labels):
	return (100 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1)) / prediction.shape[0])

num_steps = 2000
batch_size = 128

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_data[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size)]

		feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels}
		_, l, prediction = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

		if step % 500 == 0:
			print('----Result at Mini step {}----'.format(step))
			print('Loss : {}'.format(l))
			print('Training Accuracy : {}'.format(accuracy(prediction, batch_labels)))
			print('Valid Accuracy : {}'.format(accuracy(valid_prediction.eval(), valid_labels)))
			print()

	print('Final Test Accuracy : {}'.format(accuracy(test_prediction.eval(), test_labels)))