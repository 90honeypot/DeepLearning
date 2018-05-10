# Lab 9-2
# Tensorboard for XOR NN
# 1. Visualize your TF graph
# 2. Plot quantitative metrics
# 3. Show additional data

import numpy as np
import tensorflow as tf
import pprint as pp

# 1. From TF graph, decide which tensors you want to log
w2_hist = tf.summary.histogram("weights2", W2)	# Histogram (multi-dimensional tensors)
cost_summ = tf.summary.scalar("cost", cost)		# Scalar tensors

# 2. Merge all summaries
summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 3. Create writer and add graph
writer = tf.summary.FileWriter('./logs') # file location
writer.add_graph(sess.graph)

# 4. Run summary merge and add_summary
s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
writer.add_summary(s, global_step=global_step)
global_step += 1

# 5. Launch TensorBoard (terminal)
# tensorboard --logdir=./logs
# Tip!
# ssh -L local_port:127.0.0.1:remote_port username@server.com
# local>	$ ssh -L 7007:121.0.0.0:6006 hunkim@server.com
# server>	$ tensorboard --logdir=./logs
# You can navigate to http://127.0.0.1:7007



# Histogram (multi-dimensional tensors)
W2 = tf.Variable(tf.random_normal([2,1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

w2_hist = tf.summary.histogram("weights2", W2)
b2_hist = tf.summary.histogram("bias2", b2)
hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

# Add scope for better graph hierarchy
with tf.name_scope("layer1") as scope:
	W1 = tf.Variable(tf.random_normal([2,2]), name='weight1')
	b1 = tf.Variable(tf.random_normal([2]), name='bias1')
	layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

	w1_hist = tf.summary.histogram("weights1", W1)
	b1_hist = tf.summary.histogram("bias1", b1)
	layer1_hist = tf.summary.histogram("lyaer1", layer1)

with tf.name_scope("layer2") as scope:
	W2 = tf.Variable(tf.random_normal([2,1]), name='weight2')
	b2 = tf.Variable(tf.random_normal([1]), name='bias2')
	hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

	w2_hist = tf.summary.histogram("weights2", W2)
	b2_hist = tf.summary.histogram("bias2", b2)
	hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)	


# Multiple runs
# learning_rate=0.1 vs learning_rate=0.01
# logs/ directory1, directory2, directory3 ...
# tensorboard --logdir=./logs
 


























