#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 23:50:59 2019

@author: kallil
"""

import gym
import universe
import random

env = gym.make('CarRacing-v0')
env.configure(remotes=1)
observation_n = env.reset()