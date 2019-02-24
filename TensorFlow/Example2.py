#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:37:40 2019

@author: kallil
"""

import tensorflow as tf

a = tf.multiply(2,3)

with tf.Session() as sess:
    #run the session
    print(sess.run(a))