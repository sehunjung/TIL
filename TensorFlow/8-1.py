
# https://www.tensorflow.org/api_guides/python/array_ops
import tensorflow as tf
import numpy as np
import pprint
tf.set_random_seed(777)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# t = np.array([0., 1., 2., 3., 4., 5., 6.])
# pp.pprint(t)
# print(t.ndim) # rank
# print(t.shape) # shape
# print(t[0], t[1], t[-1])
# print(t[2:5], t[4:-1])
# print(t[:2], t[3:])

# t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
# pp.pprint(t)
# print(t.ndim) # rank
# print(t.shape) # shape


# 1,2 * 2, 1 => 1,1 
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
matrix3 = tf.matmul(matrix1, matrix2).eval()
print(matrix3)

tf.reshape(matrix3, shape=[-1,1])
print(matrix3)

