#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import os
import input_data
import mnist_estimator
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def test(model, mnist):
	n_images = 4
	index = random.randint(0, mnist.test.labels.size - n_images)
	test_images = mnist.test.images[index : index + n_images]
	input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': test_images}, shuffle=False)
	preds = list(model.predict(input_fn))
	for i in range(n_images):
		print('Model prediction: %d' % preds[i])
		plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
		plt.show()

def _main():
	# enable the logging info
	tf.logging.set_verbosity(tf.logging.INFO)
	mnist = input_data.read_data_sets('data/', one_hot=False)
	# create model
	model_dir = 'model/mnist_esti'
	model = tf.estimator.Estimator(mnist_estimator.model_fn, model_dir=model_dir)
	# model has been trained
	if not os.path.exists(model_dir):
		print('model need to train')
		return
	# predict model
	test(model, mnist)

if __name__ == '__main__':
	_main()
