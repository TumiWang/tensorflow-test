#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import input_data

import numpy as np
import tensorflow as tf

# init train constant
learning_rate = 0.001
num_steps = 2000
batch_size =128
num_input = 784
num_classes = 10
dropout = 0.25

# conv neural net core
def conv_neural_net(x_dict, n_classes, dropout, is_training):
	with tf.variable_scope('ConvNeuralNet', reuse=False):
		x = x_dict['images']
		x = tf.reshape(x, shape=[-1, 28, 28, 1])
		
		conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
		conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

		conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
		conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

		fcl = tf.contrib.layers.flatten(conv2)
		fcl = tf.layers.dense(fcl, 1024)
		fcl = tf.layers.dropout(fcl, rate=dropout, training=is_training)

		out = tf.layers.dense(fcl, n_classes)
	return out

def model_fn(features, labels, mode):
	if mode == tf.estimator.ModeKeys.TRAIN:
		logits = conv_neural_net(features, num_classes, dropout, is_training=True)
		loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)
	logits = conv_neural_net(features, num_classes, dropout, is_training=False)
	pred_classes = tf.argmax(logits, axis=1)
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
	assert mode == tf.estimator.ModeKeys.EVAL
	acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
	loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, eval_metric_ops={'accuracy': acc_op})

# train model function
def train_model(model, mnist):
	input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.train.images}, y=mnist.train.labels, batch_size=batch_size, num_epochs=None, shuffle=True)
	model.train(input_fn, steps=num_steps)

# evaluate model function
def eval_model(model, mnist):
	input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.test.images}, y=mnist.test.labels, batch_size=batch_size, shuffle=False)
	model.evaluate(input_fn)

def _main():
	# fetch and load data
	mnist = input_data.read_data_sets('data/', one_hot=False)
	# enable the logging info
	tf.logging.set_verbosity(tf.logging.INFO)
	# create model
	model_dir = 'model/conv_mnist'
	model = tf.estimator.Estimator(model_fn, model_dir=model_dir)
	# train model
	if not os.path.exists(model_dir):
		train_model(model, mnist)
	if not os.path.exists(model_dir):
		print('model need to train')
		return
	# eval model
	eval_model(model, mnist)

if __name__ == '__main__':
	_main()

