#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 00:29:12 2019

@author: kallil
"""

import numpy as np

teucu = np.random.random((4,5))
teucu2 = np.random.random((4,5))
teubulisco = np.hstack((teucu,teucu2))
meuchapeu = np.clip(teubulisco, -.6,.6)