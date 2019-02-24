#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:44:41 2019

@author: kallil
"""

import tensorflow as tf

tf.reset_default_graph()

#a = tf.constant(5)
#b = tf.constant(4)
#c = tf.multiply(a,b)
#d = tf.constant(2)
#e = tf.constant(3)
#f = tf.multiply(d,e)
#g = tf.add(c,f)

##Scoping
with tf.name_scope("Computation"):
    with tf.name_scope("Part1"):
        a = tf.constant(5)
        b = tf.constant(4)
        c = tf.multiply(a,b)
    with tf.name_scope("Part2"):
        d = tf.constant(2)
        e = tf.constant(3)
        f = tf.multiply(d,e)
        
with tf.name_scope("Result"):
    g = tf.add(c,f)
    
with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    print(sess.run(g))
    writer.close()
    
