#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 22:16:19 2019

@author: kallil
"""

# Import required libraries


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
from tensorflow.examples.tutorials.mnist import input_data
fashion_mnist = input_data.read_data_sets('data/fashion/', one_hot=True)

# Check what we have in our data
print("No of images in the training set {}".format(fashion_mnist.train.images.shape))
print("No of labels in the training set {}".format(fashion_mnist.train.labels.shape))

print("No of images in the test set {}".format(fashion_mnist.test.images.shape))
print("No of labels in the test set {}".format(fashion_mnist.test.labels.shape))


# Label all product categories
labels = {
        0: 'T-Shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
        }

# Sample 1

# Get 28x28 image
sample_1 = fashion_mnist.train.images[41].reshape(28,28)
# Get corresponding integer label from one-hot encoded data
sample_label_1 = np.where(fashion_mnist.train.labels[41] == 1)[0][0]
# Plot sample
print("y = {label_index} ({label})".format(label_index=sample_label_1, label=labels[sample_label_1]))
plt.imshow(sample_1, cmap='Greys')

# Sample 2

# Get 28x28 image
sample_2 = fashion_mnist.train.images[19].reshape(28,28)
# Get corresponding integer label from one-hot encoded data
sample_label_2 = np.where(fashion_mnist.train.labels[19] == 1)[0][0]
# Plot sample
print("y = {label_index} ({label})".format(label_index=sample_label_2, label=labels[sample_label_2]))
plt.imshow(sample_2, cmap='Greys')

# Build a CNN that classifies all of these images into their respective categories
# Define placeholders for input images and output labels, as our
# output image is size 784, we define a placeholder for input x as follows
x = tf.placeholder(tf.float32, [None, 784])

# Reshape the input to the format [p,q,r,s] where q and r are the actial size of an input image, which is 28 x 28
# ans s is the channel number. As we have only greyscale images, the value of s is 1. P implies thenumber of
# training sammples, that is, the batch size. We can set that as -1 and it will be dynamically changed during 
# training
x_shaped = tf.reshape(x, [-1,28,28,1])

# As we have 10 different labels, we define placeholders for the output as follows
y = tf.placeholder(tf.float32, [None, 10])

# Now we define a function which performs he convolutional operation, that is the multiplication
# between the input matrix x by the filter w with a stride of 1 and same padding
# We set strides [1,1,1,1]. The first and last relate to the fact we don't want to move
# between training samples and different channels, and the other two refers that we move
# the filter by 1 pixel in both height and weight
def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

# We define a function to perform the pooling operation. We define a pooling with stride of 2 and
# 'SAME' padding. ksize implies our pooling window shape:
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Define weights and biases. With two conv layers and a fully connected layer
# The weights are the filters in convolutional layers
# Use a 5x5 filter and set filter size to 32. Since we are using greyscale images, 
# our input channel number will be 1, so our weight matrix will be [5,5,1,32]
w_c1 = tf.Variable(tf.random_normal([5,5,1,32]))

# As the second conv layer takes the input from the first conv layer which has 32 as its channel 
# output, the number of input channels to the next layer becomes 32.
w_c2 = tf.Variable(tf.random_normal([5,5,32,64]))

# Next we initialize the bias
b_c1 = tf.Variable(tf.random_normal([32]))
b_c2 = tf.Variable(tf.random_normal([64]))

# Next we perform operations in the first conv layer, that is, the convolution operation on the
# input x with ReLU activations followed by a max pooling
conv1 = tf.nn.relu(conv2d(x_shaped,w_c1) + b_c1)
conv1 = maxpool2d(conv1)

# Now, the result of the first conv layer will be passed to the next conv layer where we perform
# the convolutional layer operation on the result of a first conv layer with ReLU activations,
# followed by max pooling
conv2 = tf.nn.relu(conv2d(conv1, w_c2) + b_c2)
conv2 = maxpool2d(conv1)

# After the two conv layers, our input img will be downsampled from 28x28 to 7x7. We need to flatten 
# this output before feeding to the fully connected layer. Then, the result of the second conv layer will be fed
# into the fully connected layer and we multiply this with weights, add bias and apply ReLU
x_flattened = tf.reshape(conv2, [100, 7*7*64])
w_fc = tf.Variable(tf.random_normal([7*7*64, 1024]))
b_fc = tf.Variable(tf.random_normal([1024]))
fc = tf.nn.relu(tf.matmul(x_flattened, w_fc) + b_fc)

# Now we need to define the weights and bias for the output layer, which is [number of neurons in the
# current layer, number of neurons in the next layer]
w_out = tf.Variable(tf.random_normal([1024,10]))
b_out = tf.Variable(tf.random_normal([10]))

# We can get the output by multiplying the result of a fully connected layer with the weight matrix and add bias. 
# We will get the probabilities of the output using the softmax activation function
output = tf.matmul(fc, w_out) + b_out
yhat = tf.nn.softmax(output)

# We will minimize our loss function using a new type of optimizer called the Adam optimizer
# instead of using the gradient descent optimizer
learning_rate = 0.01
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Next we will calculate the accuracy as follows
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Andd define the hyperparameters
epochs = 10
batch_size = 100

# Now we will start the TensorFlow session and build the model


init_op = tf.global_variables_initializer()

with tf.Session() as sess:
   sess.run(init_op)
   total_batch = int(len(fashion_mnist.train.labels) / batch_size)
    # For each epcoh
   for epoch in range(epochs):
        
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = fashion_mnist.train.next_batch(batch_size=batch_size)
            
            _, c = sess.run([optimizer, cross_entropy], 
                         feed_dict={x: batch_x, y: batch_y})
            
            avg_cost += c / total_batch
            
        print("Epoch:", (epoch + 1), "cost =""{:.3f}".format(avg_cost))
        
   print(sess.run(accuracy, feed_dict={x: fashion_mnist.test.images, y: fashion_mnist.test.labels}))

