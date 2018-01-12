""" A clean, no_frills character-level generative language model.
Created by Danijar Hafner (danijar.com), edited by Chip Huyen
for the class CS 20SI: "TensorFlow for Deep Learning Research"
http://web.stanford.edu/class/cs20si/lectures/slides_11.pdf

Based on Andrej Karpathy's blog: 
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""
from __future__ import print_function

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append('..')

import tensorflow as tf

import utils


# Hyper-parameters
HIDDEN_SIZE = 200
BATCH_SIZE = 128
WINDOW_SIZE = 50
SKIP_STEP = 100
TEMPRATURE = 0.7
LR = 0.003
LEN_GENERATED = 400


# Data
vocab = " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ" + \
        "\\^_abcdefghijklmnopqrstuvwxyz{|}"

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


# Model
seq = tf.placeholder(tf.int32, [None, None])
temp = tf.placeholder(tf.float32)

# Correct one_hot encoding with -1
# See https://github.com/chiphuyen/stanford-tensorflow-tutorials/pull/80
seq_one_hot = tf.one_hot(seq - 1, len(vocab))

cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
zero_state = cell.zero_state(batch_size=tf.shape(seq_one_hot)[0], dtype=tf.float32)
in_state = tf.placeholder_with_default(input=zero_state, shape=[None, HIDDEN_SIZE])

# This line to calculate the real length of seq
# all seq are padded to be of the same length which is NUM_STEPS
# The details of this expression here:
# https://danijar.com/variable-sequence-lengths-in-tensorflow/
# and in the `playground` file
#
# Original expression:
# length = tf.reduce_sum(tf.reduce_max(tf.sign(seq_one_hot), 2), 1)
length = tf.reduce_sum(tf.reduce_max(seq_one_hot, 2), 1)
output, out_state = tf.nn.dynamic_rnn(cell, seq_one_hot, length, in_state)

# fully_connected is syntactic sugar for tf.matmul(w, output) + b
# it will create w and b for us
logits = tf.contrib.layers.fully_connected(output, len(vocab), None)

# Predict the *next* char in the sequence:
# logits[:, :-1]       -> for each batch (axis=0) take all but the last item
# seq_one_hot[:, 1:]   -> for each batch (axis=0) take all but the first item
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, :-1], labels=seq_one_hot[:, 1:]))
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
optimizer = tf.train.AdamOptimizer(LR).minimize(loss, global_step=global_step)

# sample the next character from Maxwell-Boltzmann Distribution with temperature temp
# it works equally well without tf.exp
sample = tf.multinomial(tf.exp(logits[:, -1] / temp), 1)[:, 0]

def online_inference(sess, seed='T'):
  sentence = seed
  state = None
  for _ in range(LEN_GENERATED):
    batch = [vocab_encode(sentence[-1])]
    feed = {seq: batch, temp: TEMPRATURE}
    if state is not None:
      feed.update({in_state: state})
    index, state = sess.run([sample, out_state], feed)
    sentence += vocab_decode(index + 1)     # account for -1 in one_hot encoding
  print(sentence)
  print()


# Training
utils.make_dir('checkpoints')
utils.make_dir('checkpoints/arxiv')
saver = tf.train.Saver(max_to_keep=2)
with tf.Session() as sess:
  writer = tf.summary.FileWriter('graphs/gist', sess.graph)
  sess.run(tf.global_variables_initializer())

  ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/arxiv/checkpoint'))
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

  for epoch in range(10):
    for batch in read_batch(read_data('data/arxiv_abstracts.txt')):
      batch_loss, _, iteration = sess.run([loss, optimizer, global_step], feed_dict={seq: batch})
      if (iteration + 1) % SKIP_STEP == 0:
        print('Iter=%d Loss=%.3f' % (iteration + 1, batch_loss))
        online_inference(sess)
        saver.save(sess, 'checkpoints/arxiv/char-rnn', iteration)
