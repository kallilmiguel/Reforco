#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 16:49:22 2019

@author: kallil
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

print("No of images in the training set {}".format(mnist.train.images.shape))
print("No of labels in the training set {}".format(mnist.train.labels.shape))

print("No of images in the test set {}".format(mnist.test.images.shape))
print("No of labels in the test set {}".format(mnist.test.labels.shape))

img1 = mnist.train.images[41].reshape(28,28)
plt.imshow(img1, cmap='Greys')

x = tf.placeholder(tf.float32, [None, 784])

y = tf.placeholder(tf.float32, [None, 10])

# Initializing our parameters
learning_rate = 0.1
epochs = 10
batch_size = 100

# Define weight and biases between input and hidden layers

w_xh = tf.Variable(tf.random_normal([784,300], stddev=0.03), name='w_xh')
b_h = tf.Variable(tf.random_normal([300]), name='b_h')

# Define weight and bias between hidden and output layers

w_hy = tf.Variable(tf.random_normal([300,10], stddev=0.03), name='w_hy')
b_y = tf.Variable(tf.random_normal([10]), name='b_y')

# Perform the forward propagation
z1 = tf.add(tf.matmul(x, w_xh), b_h)
a1 = tf.nn.relu(z1)
z2 = tf.add(tf.matmul(a1, w_hy), b_y)
yhat = tf.nn.softmax(z2)

# Define cost function as a cross-entropy loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(yhat),reduction_indices=[1]))

# Use the built in function for the Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Calculate the acurracy to evaluate our training model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize TensorFlow Variables
init_op = tf.global_variables_initializer()

# Start the tensorflow session and training model
with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels)/batch_size)
    for epoch in range(epochs):
        avg_cost=0
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y:batch_y})
            avg_cost += c/total_batch
        print("Epoch: ", (epoch+1), "cost = ""{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))