#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = 'maxim'

import tensorflow as tf

# Creates a graph.
with tf.device('/gpu:0'):
 a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
 b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
 c = tf.add(a, b)

# Creates a session with log_device_placement set to True.
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
  # Runs the op.
  print(sess.run(c))
