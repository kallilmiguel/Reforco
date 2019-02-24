#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 23:16:43 2019

@author: kallil
"""

import gym 

env = gym.make('CarRacing-v0')

env.reset()

for _ in range (1000):
    env.render()
    env.step(env.action_space.sample())
     