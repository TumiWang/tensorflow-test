#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import input_data

import numpy as np
import tensorflow as tf

# init train constant
learning_rate = 0.1
num_steps = 10000
batch_size = 128
display_step = 100
n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784
num_classes = 10

# neural net core
def neural_net(x_dict):
	x = x_dict['images']
	layer_1 = tf.layers.dense(x, n_hidden_1)
	layer_2 = tf.layers.dense(layer_1, n_hidden_2)
	out_layer = tf.layers.dense(layer_2, num_classes)
	return out_layer

# neural net model function
def model_fn(features, labels, mode):
	logits = neural_net(features)

	pred_classes = tf.argmax(logits, axis=1)
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

	pred_probas = tf.nn.softmax(logits)
	loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
	acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
	return tf.estimator.EstimatorSpec(mode=mode,loss=loss_op,train_op=train_op,eval_metric_ops={'accuracy': acc_op})

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
	model_dir = 'model/mnist_esti'
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
