#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = 'maxim'

import tensorflow as tf

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.constant(99, name='z')         # not present in tensorboard!

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  writer = tf.summary.FileWriter('/tmp/inline', sess.graph)
  for _ in range(100):
    sess.run(tf.add(x, y)) # someone decides to be clever to save one line of code
  writer.close()

  # will print 100 add ops
  print(tf.get_default_graph().as_graph_def())
