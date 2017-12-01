# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:32:05 2017

@author: Kyle
"""

import tensorflow as tf

a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(x)
writer.close()

