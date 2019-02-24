#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 22:41:42 2019

@author: kallil
"""

import sys 
import numpy as np #matrix and vector computation package
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#Set seaborn plotting style
sns.set_style('darkgrid')
# Set the seed for reproducability
np.random.seed(seed=13)

# Define the vector of input samples as x, with 20 values
# Sampled from a uniform distribution between 0 and 1
x = np.random.uniform(0, 1, 20)

# Generate the target values t from x with small gaussian noise
# So the estimation won't be perfect
# Define a function f that represents the line that generates t without noise
def f(x):
    return x*2

# Create the targets t with some gaussian noise
noise_variance = 0.2 #variance of the gaussian noise
# Gaussian noise error for each sample in x
noise = np.random.randn(x.shape[0])*noise_variance
# Create targets t
t = f(x) + noise

# Plot the target t versus the input x
plt.figure(figsize=(5,3))
plt.plot(x,t,'o', label='$t$')
# Plot the initial line
plt.plot([0, 1], [f(0),f(1)], 'b--', label='$f(x)$')
plt.xlabel('$x$', fontsize=12)
plt.ylabel('$t$', fontsize=12)
plt.axis((0,1,0,2))
plt.title('inputs (x) vs targets (t)')
plt.legend(loc=2)
plt.show()

# Define our neural network model
def nn(x,w):
    """ Output function y = x * w"""
    return x * w

# Define our loss function
def loss(y,t):
    """MSE loss function"""
    return np.mean((t-y)**2)

# Plot the loss vs the given weigth w
    
# Vector of weigths for which we want to plot the loss
ws = np.linspace(0,4, num=100) # weigth values
# loss for each weigth in ws
loss_ws = np.vectorize(lambda w: loss(nn(x,w) , t))(ws)

#plot
plt.figure(figsize=(5, 3))
plt.plot(ws, loss_ws, 'r--', label='loss')
plt.xlabel('$w$', fontsize=12)
plt.ylabel('$\\xi$', fontsize=12)
plt.title('loss function with respect to $w$')
plt.xlim(0, 4)
plt.legend()
plt.show()

# Implement the gradient descent function
def gradient(w, x, t):
    """ Gradient function. (Remember that y = nn(x,w) = x * w)"""
    return 2 * x * (nn(x, w) - t)

def delta_w(w_k, x, t, learning_rate):
    """ Update function delta w"""
    return learning_rate * np.mean(gradient(w_k, x, t))


# Initial weigth parameter
w = np.random.rand()
# Set the learning rate
learning_rate = 0.9

# Perform the gradient descent updates, and print the weigths and loss
nb_of_iterations = 4 #number of gradient descent updates
w_loss = [(w, loss(nn(x,w), t))] #Keep track of weigth and loss values
for i in range(nb_of_iterations):
    dw = delta_w(w, x, t, learning_rate) # Delta w update
    w = w - dw
    w_loss.append((w, loss(nn(x,w), t))) # Save weigth and loss
    
# Print the final w, and loss
for i in range(0, len(w_loss)):
    print(f'w({i}): {w_loss[i][0]:.4f} \t loss: {w_loss[i][1]:.4f}')
    
    

# Visualise the gradient descent updates
plt.figure(figsize=(6, 4))
plt.plot(ws, loss_ws, 'r--', label='loss')  # Loss curve
# Plot the updates
for i in range(0, len(w_loss)-1):
    w1, c1 = w_loss[i]
    w2, c2 = w_loss[i+1]
    plt.plot(w1, c1, 'bo')
    plt.plot([w1, w2],[c1, c2], 'b-')
    plt.text(w1, c1+0.05, f'${i}$')
plt.plot(w2, c2, 'bo', label='$w(k)$')
plt.text(w2, c2+0.05, f'${i+1}$')
# Show figure
plt.xlabel('$w$', fontsize=12)
plt.ylabel('$\\xi$', fontsize=12)
plt.title('Gradient descent updates plotted on loss function')
plt.xlim(0, 4)
plt.legend(loc=1)
plt.show()