#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:11:22 2019

@author: kallil
"""

import gym_bandits
import gym
import numpy as np
import math

env = gym.make('BanditTenArmedGaussian-v0')

# Reset environment
env.reset()

# number of rounds (iterations)
num_rounds = 20000

# Count of number of times an arm was pulled
count = np.zeros(10)

# Sum of rewards of each arm
sum_rewards = np.zeros(10)

# Q value which is the average reward
Q = np.zeros(10)

def softmax(tau):
    total = sum([math.exp(val/tau) for val in Q])
    probs = [math.exp(val/tau)/total for val in Q]
    threshold = np.random.random()
    cumulative_prob = 0.0
    for i in range(len(probs)):
        cumulative_prob += probs[i]
        if(cumulative_prob > threshold):
            return i
    return np.argmax(probs)

# Start pulling the arm
for i in range(num_rounds):
    # Select the arm using softmax
    arm = softmax(0.5)
    # Get the reward
    observation, reward, done, info = env.step(arm)
    # Update the count of that arm
    count[arm] += 1
    # Sum the rewards obtained from the arm
    sum_rewards[arm] += reward
    # Calculate Q value which is  the average rewards of the arm
    Q[arm] = sum_rewards[arm]/count[arm]

print('The optimal arm is {}'.format(np.argmax(Q)))
    