# file read & data slicing..

import tensorflow as tf
import numpy as np # file을 읽기 위해 필요...

tf.set_random_seed(777)  # for reproducibility

# 경로 설정해서 가져 옴.=> xy 데이터 모두가져옴.
#f_path  = "C:\==Task\Github\TIL\TensorFlow\"
xy = np.loadtxt('C:\==Task\Github\TIL\TensorFlow\data-01-test-score.csv', delimiter=',', dtype=np.float32)

# 가져온 데이터에서 x,y slicing
x_data = xy[:, 0:-1] # x축은 모두 y축은 제일 뒤만 빼고
y_data = xy[:, [-1]] # x축은 모두 y축은 제일 뒤 라인만

# Make sure the shape and data are OK
 

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Set up feed_dict variables inside the loop.
for step in range(2001):
   cost_val, hy_val, _ = sess.run(
       [cost, hypothesis, train], 
       feed_dict={X: x_data, Y: y_data})
   if step % 10 == 0:
       print(step, "Cost: ", cost_val, 
                  "\nPrediction:\n", hy_val)


# # Ask my score
# print("Your score will be ", sess.run(hypothesis, 
#            feed_dict={X: [[100, 70, 101]]}))

# print("Other scores will be ", sess.run(hypothesis, 
#            feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
