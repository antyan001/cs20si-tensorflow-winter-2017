""" Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from 
the number of fire in the city of Chicago
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

import utils

DATA_FILE = 'data/fire_theft.xls'

# Phase 1: Assemble the graph
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
x = tf.placeholder(dtype=tf.float32, shape=(), name='x')
y = tf.placeholder(dtype=tf.float32, shape=(), name='y')

# Step 3: create weight and bias, initialized to 0
w = tf.Variable(initial_value=0.0, name='w')
b = tf.Variable(initial_value=0.0, name='b')

# Step 4: predict Y (number of theft) from the number of fire
# name your variable Y_predicted
y_predicted = x * w + b

# Step 5: use the square error as the loss function
#loss = tf.nn.l2_loss(y_predicted - y, name='loss')
loss = utils.huber_loss(y, y_predicted)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# Phase 2: Train our model
with tf.Session() as sess:
  # Step 7: initialize the necessary variables, in this case, w and b
  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter('/tmp/linear-reg', sess.graph)

# Step 8: train the model
  for i in range(100):  # run 100 epochs
    total_loss = 0
    for train_x, train_y in data:
      # Session runs optimizer to minimize loss and fetch the value of loss. Name the received value as l
      _, l = sess.run([optimizer, loss], feed_dict={x: train_x, y: train_y})

      total_loss += l
    print("Epoch {0}: {1}".format(i, total_loss / n_samples))

  writer.close()

  w_value, b_value = sess.run([w, b])

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()
