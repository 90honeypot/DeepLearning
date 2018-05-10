# Lab 9
# NN for XOR

import numpy as np
import tensorflow as tf
import pprint as pp

# XOR data set
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

# # 1 layer (logistic regression)
# W = tf.Variable(tf.random_normal([2,1]), name='weight') # tf.random_normal([in, out])
# b = tf.Variable(tf.random_normal([1]), name='bias')     # tf.random_normal([out])
# # sigmoid = tf.div(1., 1. + tf.exp(tf.matmul(X,W)))
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# Neural Net (2 layers)
with tf.name_scope("layer1") as scope:
	W1 = tf.Variable(tf.random_normal([2,10]), name='weight1')
	b1 = tf.Variable(tf.random_normal([10]), name='bias1')
	layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

	w1_hist = tf.summary.histogram("weights1", W1)
	b1_hist = tf.summary.histogram("bias1", b1)
	layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope:
	W2 = tf.Variable(tf.random_normal([10,1]), name='weight2')
	b2 = tf.Variable(tf.random_normal([1]), name='bias2')
	hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

	w2_hist = tf.summary.histogram("weights2", W2)
	b2_hist = tf.summary.histogram("bias2", b2)
	hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)



# Wide NN for XOR : 
# W1 = tf.Variable(tf.random_normal([2,10]), name='weight1')
# b1 = tf.Variable(tf.random_normal([10]), name='bias1')
# layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
# W2 = tf.Variable(tf.random_normal([10,1]), name='weight2')
# b2 = tf.Variable(tf.random_normal([1]), name='bias2')
# hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# Deep NN for XOR : 
# W1 = tf.Variable(tf.random_normal([2,10]), name='weight1')
# b1 = tf.Variable(tf.random_normal([10]), name='bias1')
# layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
# W2 = tf.Variable(tf.random_normal([10,10]), name='weight2')
# b2 = tf.Variable(tf.random_normal([10]), name='bias2')
# layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
# W3 = tf.Variable(tf.random_normal([10,10]), name='weight3')
# b3 = tf.Variable(tf.random_normal([10]), name='bias3')
# layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
# W4 = tf.Variable(tf.random_normal([10,1]), name='weight4')
# b4 = tf.Variable(tf.random_normal([1]), name='bias4')
# hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

with tf.name_scope("cost") as scope:
	cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
	cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("train") as scope:
	train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

# Launch graph
with tf.Session() as sess:
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter("./logs")
	writer.add_graph(sess.graph)

	# Initialize TensorFlow variables
	sess.run(tf.global_variables_initializer())

	for step in range(10001):
		summary, _ = sess.run([merged_summary,train], feed_dict={X: x_data, Y: y_data})
		writer.add_summary(summary, global_step=step)

		if step % 100 == 0:
			print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2]))

	# Accuracy report
	h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
	print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
