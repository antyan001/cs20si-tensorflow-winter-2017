#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = 'maxim'

import numpy as np
import tensorflow as tf


# Hyper-parameters
HIDDEN_SIZE = 200
BATCH_SIZE = 2
WINDOW_SIZE = 10

# Data
vocab = "abcdef"

def vocab_encode(text):
  return [vocab.index(x) + 1 for x in text if x in vocab]

def vocab_decode(array):
  return ''.join([vocab[x - 1] for x in array])

def read_data(filename, window=WINDOW_SIZE, overlap=WINDOW_SIZE // 2):
  for text in open(filename):
    text = vocab_encode(text)
    for start in range(0, len(text) - window, overlap):
      chunk = text[start: start + window]
      chunk += [0] * (window - len(chunk))
      yield chunk

def read_batch(stream, batch_size=BATCH_SIZE):
  batch = []
  for element in stream:
    batch.append(element)
    if len(batch) == batch_size:
      yield batch
      batch = []
  yield batch


seq = tf.placeholder(tf.int32, [None, None])
temp = tf.placeholder(tf.float32)

seq_one_hot = tf.one_hot(seq - 1, len(vocab))   # Note -1!
cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
in_state = tf.placeholder_with_default(cell.zero_state(tf.shape(seq_one_hot)[0], tf.float32),
                                       shape=[None, HIDDEN_SIZE])
# The details of this expression here:
# https://danijar.com/variable-sequence-lengths-in-tensorflow/
sign = tf.sign(seq_one_hot)
used = tf.reduce_max(sign, 2)
length = tf.reduce_sum(used, 1)
length_simplified = tf.reduce_sum(tf.reduce_max(seq_one_hot, 2), 1)


def process_batch(batch):
  seq_val, sign_val, reduce_max_val, length_val, length_val2 = sess.run([seq_one_hot, sign, used, length, length_simplified], {seq: batch})
  print(batch)
  print(seq_val)
  print(sign_val)
  print(reduce_max_val)
  print(length_val)
  print(length_val2)
  print()
  print()
  print()


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  process_batch([[1, 1], [1, 0], [0, 0]])
  process_batch([[1, 2, 3], [1, 2, 4]])
  process_batch([[1, 2, 0], [0, 0, 0]])
  process_batch([[1, 2, 4, 5, 0, 0, 0]])
  process_batch([[1, 2, 4, 5]])

  # for batch in read_batch(read_data('data/arxiv_abstracts.txt')):
  #   process_batch(batch)
