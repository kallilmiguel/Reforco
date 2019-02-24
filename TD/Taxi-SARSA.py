#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 16:12:11 2019

@author: kallil
"""

import gym
import random

env = gym.make("Taxi-v2")

env.render()

alpha = 0.85
gamma = 0.90
epsilon = 0.8
Q = {}

for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s,a)] = 0.0
        
def update_q_table(prev_state, action, reward, nextstate, alpha, gamma):
    qa = max([Q[(nextstate,a)] for a in range(env.action_space.n)]) 
    Q[(prev_state, action)] += alpha * (reward + gamma * qa - Q[(prev_state, action)])
    
def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda x: Q[(state,x)])
    
for i in range(4000):
    # We store cumulative reward of each episodes in r
    r=0
    # Then for every iterations, we initialize the state
    state = env.reset()
    # Then we pick up the action using epsilon greedy policy
    action = epsilon_greedy_policy(state, epsilon)
    while True:
        # Then we perform the action in the state and move the next state
        nextstate, reward, done, _ = env.step(action)
        # Then we pick up the next action using epsilon greedy policy
        nextaction = epsilon_greedy_policy(nextstate, action)
        # We calculate Q value of the previous state using our update rule
        Q[(state, action)] += alpha * (reward + gamma * Q[(nextstate, nextaction)] - Q[(state, action)])
        # Finally we update our state and action with next action and next state
        action = nextaction        
        state = nextstate
        r+= reward
        # We will break the loop, if we are in the terminal state
        if done:
            break
    
    print("Total reward: ", r)
    
env.close()