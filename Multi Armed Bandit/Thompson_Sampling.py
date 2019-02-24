#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:45:49 2019

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

# Initialize alpha and beta values
alpha = np.ones(10)
beta = np.ones(10)

def thompson_sampling(alpha, beta):
    samples = [np.random.beta(alpha[i]+1, beta[i]+1) for i in range(10)]
    
    return np.argmax(samples)

#Start pulling arms
for i in range(num_rounds):
    #Select the arm using UCB
    arm = thompson_sampling(alpha, beta)
    # Get the reward
    observation, reward, done, info = env.step(arm)
    # Update the count of that arm
    count[arm] += 1
    # Sum the rewards obtained from the arm
    sum_rewards[arm] += reward
    # Calculate Q value which is the average rewards of the arm
    Q[arm] = sum_rewards[arm]/count[arm]
    
    # If it is a positive reward, increment alpha
    if reward > 0:
        alpha[arm]+=1
    # else increment beta
    else:
        beta[arm] += 1

print('The optimal arm is {}'.format(np.argmax(Q)))