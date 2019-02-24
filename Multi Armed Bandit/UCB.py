#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:57:03 2019

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

def UCB(iters):
    ucb = np.zeros(10)
    # Explore all the arms
    if iters < 10:
        return i
    else:
        for arm in range (10):
            # Calculate upper bound
            upper_bound = math.sqrt((2*math.log(sum(count)))/count[arm])
            # Add upper bound to the Q value
            ucb[arm] = Q[arm] + upper_bound
        # return the arm which has a maximum value
        return np.argmax(ucb)

#Start pulling arms
for i in range(num_rounds):
    #Select the arm using UCB
    arm = UCB(i)
    # Get the reward
    observation, reward, done, info = env.step(arm)
    # Update the count of that arm
    count[arm] += 1
    # Sum the rewards obtained from the arm
    sum_rewards[arm] += reward
    # Calculate Q value which is the average rewards of the arm
    Q[arm] = sum_rewards[arm]/count[arm]

print('The optimal arm is {}'.format(np.argmax(Q)))