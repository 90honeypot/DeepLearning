
# source ~/tensorflow/bin/activate
# source ~/tensorflow/bin/deactivate

# Lab - 8
# Tensor Manipulation
import pprint as pp
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

print("----- Array -----")
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim)               # rank
print(t.shape)              # shape : the number of elements
print(t[0], t[1], t[-1])    # -1 : the last one
print(t[2:5], t[4:-1])      # [A:B] : not include B
print(t[:2], t[3:])         # [A:] : from A to end


print("----- 2D Array -----")
t2 = np.array([[1., 2., 3.], [4., 5., 6.]])
pp.pprint(t2)
print(t2.ndim)
print(t2.shape)


print("----- Shape -----")
ten1 = tf.constant([1,2,3,4])       # rank : 1 / shape(4)
print(tf.shape(ten1).eval())
ten2 = tf.constant([[1,2,3,4],[1,2,3,4]])       # rank : 2 / shape(2,4)
print(tf.shape(ten2).eval())
ten3 = tf.constant([[[3,1,2],[1,3,4]],[[1,5,6],[3,7,8]]])       # rank : 3 / shape(2,2,3)
print(tf.shape(ten3).eval())


print("----- Axis -----")   # 가장 안쪽 = -1, 바깥쪽부터 0, 1, 2, ...
ta = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
print(ta.eval())
print(tf.shape(ta).eval())


print("----- Matmul -----") # Matmul VS. multiply
matrix1 = tf.constant([[1., 2.],[3.,4.]])
matrix2 = tf.constant([[1.], [2.]])
print("Matrix1", matrix1.shape)
print("Matrix2", matrix2.shape)
print(tf.matmul(matrix1, matrix2).eval())       # right
print((matrix1*matrix2).eval())     # wrong


print("----- Broadcasting -----") # To make it possible to calculate wrong shape
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant(1.)
print((matrix1+matrix2).eval())
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([3., 4.])
print((matrix1+matrix2).eval())
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([[3.],[4.]])
print((matrix1+matrix2).eval())


print("----- Reduce_mean, Reduce_sum -----")
print(tf.reduce_mean([1, 2], axis=0).eval())
print(tf.reduce_mean([1., 2.], axis=0).eval())
print("### mean")
x = [[1., 2.], [3., 4.]]
print(tf.reduce_mean(x).eval())
print(tf.reduce_mean(x, axis=0).eval())
print(tf.reduce_mean(x, axis=1).eval())
print(tf.reduce_mean(x, axis=-1).eval())
print("### sum")
print(tf.reduce_sum(x).eval())
print(tf.reduce_sum(x, axis=0).eval())
print(tf.reduce_sum(x, axis=1).eval())
print(tf.reduce_sum(x, axis=-1).eval())
print(tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval())


print("----- Argmax -----") # location of the maximum value
y = [[0,1,2],[2,1,0]]
print(tf.argmax(y, axis=0).eval())
print(tf.argmax(y, axis=1).eval())
print(tf.argmax(y, axis=-1).eval())


print("----- Reshape *** -----")
r = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,1,2]]])
print(r.shape)
print(tf.reshape(r, shape=[-1,3]).eval())
print(tf.reshape(r, shape=[-1,1,3]).eval())
print("### squeeze")
print(tf.squeeze([[0],[1],[2]]).eval())
print("### expand")
print(tf.expand_dims([0,1,2], 1).eval())


print("----- One hot -----")
temp =tf.one_hot([[0],[1],[2],[0]],depth=3).eval()
print(temp) # rank +1
print(tf.reshape(temp, shape=[-1,3]).eval())


print("----- Casting -----")
print(tf.cast([1.8, 2.2, 4.9], tf.int32).eval())
print(tf.cast([True, False, 1==1, 1==0], tf.int32).eval())


print("----- Stack -----")
x = [1,4]
y = [2,5]
z = [3,6]
print(tf.stack([x,y,z]).eval())
print(tf.stack([x,y,z], axis=1).eval())
print(tf.stack([x,y,z], axis=-1).eval())


print("----- One, Zeros_like -----")
k = [[0,1,2], [2,1,0]]
print(tf.ones_like(k).eval())
print(tf.zeros_like(k).eval())


print("----- Zip ------")
for x, y in zip([1,2,3],[4,5,6]):
    print(x, y)


sess.close()
