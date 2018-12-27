import tensorflow as tf

# 데이터를 행렬로 정의.
# 5,3 행렬
x_data = [[73., 80., 75.], [93., 88., 93.],
         [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
#5,1 행렬
y_data = [[152.], [185.], [180.], [196.], [142.]]

# placeholders for a tensor that will be always fed.
# 데이터를 받을 변수도 shape 이용해서 정의
# None은 정의하지 않음 => 무한대의 인스턴스 가능..
X = tf.placeholder(tf.float32, shape=[None, 3]) # 3열 
Y = tf.placeholder(tf.float32, shape=[None, 1]) # 결과 1열

# tf var도 shape를 유의해서 정의..
# 5,3 인풋 5,1 아웃이므로 w는 3,1 shape를 가짐.
# 기존의 w1, w2, w3가 행렬로 변해서 w[3,1]로 변환가능.. 
W = tf.Variable(tf.random_normal([3, 1]), name='weight')

#기울기는 변하지 않음.
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
# 이론식도 matrix 곱샘식으로 변경 matmul 사용
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
# hy -> cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize
# cost -> train
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
   cost_val, hy_val, _ = sess.run(
       [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
   if step % 10 == 0:
       print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

