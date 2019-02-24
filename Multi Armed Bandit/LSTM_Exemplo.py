#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 18:44:41 2019

@author: kallil
"""

import locale
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np


locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


# read our file containing song lyrics
with open("ZaynLyrics.txt", "r") as f:
    data = f.read()
    data = data.replace('\n', ' ')
    data = data.lower()
    
# let us see what we got in our data
data[:50]

#then we store all the characters we got in our data into a varibale all_chars
all_chars = list(set(data))

#we store the number of unique characters in unique_chars
unique_chars = len(all_chars)

# Also we store the total number of characters in total_char
total_chars = len(data)

#now we create a mapping between each character to their index
#char_to_ix will have a character to index mapping
#ix_to_char willl have an index to character mapping

char_to_ix = {ch:i for i,ch in enumerate(all_chars)}
ix_to_char = {i:ch for i,ch in enumerate(all_chars)}

# Next, we define a function called generate_batch, which will generate input and target values. Target
# values are jus the i times shift of input value

def generate_batch(seq_length, i):
    inputs = [char_to_ix[ch] for ch in data[i:i+seq_length]]
    targets = [char_to_ix[ch] for ch in data[i+1:i+seq_length+1]]
    inputs = np.array(inputs).reshape(seq_length,1)
    targets = np.array(targets).reshape(seq_length,1)
    return inputs, targets

# Now we initialize the sequence length, learning rate and number of nodes which is number of neurons
seq_length = 25
learning_rate = 0.1
num_nodes = 300

# Now let us build our LSTM RNN. Tensorflow provides us a function called BasicLSTMCell() for building the LSTM cell and we
# need to specify the number of units we want in the LSTM cell and type of activation we wish to use

# So we create an LSTM cell and then build the RNN with that cell using tf.nn.dynamic_rnn() function which will return the 
# output and the state value

def build_rnn(x):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units = num_nodes, activation = tf.nn.relu)
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
    return outputs, states


# Now we create a placeholder for our input X and the target Y
X = tf.placeholder(tf.float32,[None,1])
Y = tf.placeholder(tf.float32, [None,1])

# Convert X and Y to integer type
X = tf.cast(X, tf.int32)
Y = tf.cast(Y,tf.int32)

# We also create a one hot representations for X and Y
X_onehot = tf.one_hot(X,unique_chars)
Y_onehot = tf.one_hot(Y,unique_chars)

# Get the output and ststes from the RNN by calling build_rnn function
outputs, states = build_rnn(X_onehot)

# Transpose the output
outputs = tf.transpose(outputs, perm=[1,0,2])

# Initialize weigts and bias value
W = tf.Variable(tf.random_normal((num_nodes,unique_chars), stddev=0.001))
B = tf.Variable(tf.zeros((1,unique_chars)))

# Now we calculate output(Ys) by multiplying output with weigths and adding bias
Ys = tf.matmul(outputs[0],W) + B

# Apply softmax activation on the output (Ys) and get the probabillities
prediction = tf.nn.softmax(Ys)

# Now we calculate the cross entropy loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=Ys))

#Our objective is to minimize the loss so we backpropagate the network and perform gradient descent
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Now we define the function called predict which results the indices of next char according to our model
def predict(seed, i):
    x = np.zeros((1,1))
    x[0][0] = seed
    indices = []
    for t in range(i):
        p=sess.run(prediction,{X:x})
        index = np.random.choice(range(unique_chars), p=p.ravel())
        x[0][0] = index
        indices.append(index)
    return indices

# Now we set our batch size, no of batches and number of epochs also the shift value for generating batch
batch_size = 100
total_batch = int(total_chars//batch_size)
epoch=1000
shift=0

#Finally, we can now start the tensorflow session and start building the model
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoch):
        print("Epoch {}:".format(epoch))
        if shift + batch_size+1 >= len(data):
            shift = 0
        
        # get the input and target for each batch by generate_batch function which shifts the input by shift value
        #and from target
        for i in range(total_batch):
            inputs, targets = generate_batch(batch_size,shift)
            shift += batch_size
            
            #calculate loss
            if(i%100==0):
                loss = sess.run(cross_entropy, feed_dict={X:inputs, Y:targets})
                
                # We get index of next predicted character by the predict function
                index = predict(inputs[0],200)
                
                # Pass the index to our ix_to_char dictionary and get the char
                txt = ''.join(ix_to_char[ix] for ix in index)
                print('Iteration %i'%(i))
                print('\n %s \n' %(txt, ))
                
            sess.run(optimiser, feed_dict={X:inputs, Y:targets})
