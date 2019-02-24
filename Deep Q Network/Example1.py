#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:13:00 2019

@author: kallil
"""

# Import all the necessary libraries
import numpy as np
import gym
import tensorflow as tf
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from collections import deque, Counter
import random
from datetime import datetime

env = gym.make('MsPacman-v0')
n_outputs = env.action_space.n

# Now we define a preprocess_observation function for preprocessing our
# input game screen. We reduce the image size and convert the image to grayscale
color = np.array([210, 164, 74]).mean()

def preprocess_observation(obs):
    
    # Crop and resize the image
    img = obs[1:176:2, ::2]
    
    # Convert the image to greyscale
    img = img.mean(axis=2)
    
    # Improve image contrast
    img[img==color] = 0
    
    # Next we normalize the image from -1 to +1
    img = (img-128) / 128 -1
    
    return img.reshape(88,80,1)

# Define a function for building our Q network. The input will be the game state x
# We build a Q network with three conv. layers with the same padding and a fully connected layer
tf.reset_default_graph()

def q_network(X, name_scope):
    # Initialize layers
    initializer = tf.contrib.layers.variance_scaling_initializer()
    
    with tf.variable_scope(name_scope) as scope:
        
        # Initialize the convolutional layers
        layer_1 = conv2d(X, num_outputs=32, kernel_size=(8,8), stride=4, padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer_1', layer_1)
        layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4,4), stride=2, padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer_2', layer_2)
        layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3,3), stride=1, padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer_3', layer_3)
        
        #Flatten the result of layer_3 before feeding to the fully connected layer
        flat = flatten(layer_3)
        
        fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)
        tf.summary.histogram('fc', fc)
        output = fully_connected(fc, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)
        tf.summary.histogram('output', output)
        
        # Vars will store the parameters of the network such as weights
        vars = {v.name[len(scope.name)]: v for v in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
        
        return vars, output
    
# Define an epsilon greedy function. In this case we use a decaying epsilon-greedy policy where the value of epsilon will be 
# decaying over time as we don't want to explore forever. So, over time, our policy will be exploiting only good actions
epsilon = 0.5
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 500000
def epsilon_greedy(action, step):
    p = np.random.random(1).squeeze()
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return action
    
#Now we initialize our experience replay buffer of length 20000, which holds the experience
# We store all the agent's experiences in the experience replay buffer and we sample this mini
# batch of experiences for training the network:
buffer_len = 20000
exp_buffer = deque(maxlen=buffer_len)

def sample_memories(batch_size):
    perm_batch =  np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4]

# Next we define all our hyperparameters
num_episodes = 800
batch_size = 48
input_shape = (None, 88, 80, 1)
learning_rate = 0.001
X_shape = (None, 88, 80, 1)
discount_factor = 0.97

global_step = 0
copy_steps = 100
steps_train = 4
start_steps = 2000
logdir = 'logs'

# Now we define the placeholder for our input, such as the game state
X = tf.placeholder(tf.float32, shape=X_shape)

# We define a boolean called in training mode to toggle the training
in_training_mode = tf.placeholder(tf.bool)

# We build our Q network, which takes the input X and generates Q values for all
# the actions in he state
mainQ, mainQ_outpputs = q_network(X, 'mainQ')

# Similarly, we build our target Q network
targetQ, targetQ_outputs = q_network(X, 'targetQ')

# Define the placeholder for our action values
X_action = tf.placeholder(tf.int32, shape=(None,))
Q_action = tf.reduce_sum(targetQ_outputs * tf.one_hot(X_action, n_outputs), axis=-1, keep_dims=True)

# Copy the main Q network parameters to the target Q network
copy_op = [tf.assign(main_name, targetQ[var_name]) for var_name, main_name in mainQ.items()]
copy_target_to_main = tf.group(*copy_op)

# Define a placeholder for our output, such as action
y = tf.placeholder(tf.float32, shape=(None, 1))

# Now we calculate the loss, which is the difference between the actual value and 
# predicted value
loss = tf.reduce_mean(tf.square(y - Q_action))

# We use AdamOptimizer for minimizing the loss
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# Set up the log files for visualization in TensorBoard
loss_summary = tf.summary.scalar('LOSS', loss)
merge_summary = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Next we start the TensorFlow session and run the model

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    # for each episode
    for i in range(num_episodes):
        done = False
        obs = env.reset()
        epoch = 0
        episodic_reward = 0
        actions_counter = Counter()
        episodic_loss = []
        
        # While the state is not the terminal state
        while not done:
            
            env.render()
            # get the preprocessed game screen
            obs = preprocess_observation(obs)
            
            #feed the game screen and get the Q values for each action
            actions = mainQ_outpputs.eval(feed_dict = {X:[obs], in_training_mode:False})
            
            # Get the action
            action = np.argmax(actions, axis=-1)
            actions_counter[str(action)] += 1
            
            # Select the action using epsilon greedy policy
            action = epsilon_greedy(action, global_step)
            # Now perform the action and move to the next state, 
            # next_obs, receive reward
            next_obs, reward, done, _ = env.step(action)
            
            # Store this transition as an experience in the replay buffer
            exp_buffer.append([obs, action, preprocess_observation(next_obs), reward, done])
            # After certain steps, we train our Q network with samples from the experiene replay buffer
            if global_step % steps_train == 0 and global_step > start_steps:
                # Sample experience
                o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)
                
                # States
                o_obs = [x for x in o_obs]
                
                # Next states
                o_next_obs = [x for x in o_next_obs]
                
                # Next actions 
                next_act = mainQ_outpputs.eval(feed_dict={X:o_next_obs, in_training_mode:False})
                
                # Reward
                y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1-o_done)
                
                # merge all summaries and write to the file
                mrg_summary = merge_summary.eval(feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:False})
                file_writer.add_summary(mrg_summary, global_step)
                
                # now we train the network and calculate loss
                train_loss, _ = sess.run([loss, training_op], feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:True})
                episodic_loss.append(train_loss)
            
            # After some interval we copy our main Q network weights to target Q network
            if ((global_step+1) % copy_steps == 0 and global_step > start_steps):
                copy_target_to_main.run()
            obs = next_obs
            epoch += 1
            global_step += 1
            episodic_reward += reward
        print('Epoch', epoch, 'Reward', episodic_reward,)