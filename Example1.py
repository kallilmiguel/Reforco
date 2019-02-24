#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 23:11:56 2019

@author: kallil
"""

import gym 

env = gym.make('CartPole-v0')

env.reset()

for _ in range (1000):
    env.render()
    env.step(env.action_space.sample())