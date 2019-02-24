#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 23:44:26 2019

@author: kallil
"""

import gym

env = gym.make('BipedalWalker-v2')

for episode in range (100):
    observation = env.reset()
    for i in range(10000):
        env.render()
        action = env.action_space.sample()
        observation,reward,done,info = env.step(action)
        if done:
            print ("{} timesteps taken for the Episode".format(i+1))
            break
        