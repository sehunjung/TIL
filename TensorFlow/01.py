#tensorflw 실습

#python -m pip install tensorflow

# 1일차...
import tensorflow as tf

#hello = tf.constant("Hello, TensorFlow!")
#sess = tf.Session()
#print(sess.run(hello))


# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0)
# node3 = tf.add(node1, node2)

# sess = tf.Session()

# #print("sess.run(node3):", sess.run(node3))


# a = tf.placeholder(tf.float32) #변수만 선언
# b = tf.placeholder(tf.float32)
# adder_node = a + b   # 프로우만 선언-operation

# print(sess.run(adder_node, feed_dict={a: 3, b:4.5})) # 출력시 변수 연결

# 수식정의 f(x) = Wx = b
# W, b처럼 가중치, 기울기 등 입력 데이터 외 tf에서 사용하는 변수
w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#데이터 입력 변수선언 -1차원
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

# 수식 선언...데이터의 흐름...
hy = w * x + b

# 최소값 구하는 식 정의 - "예상치-실제"제곱(square)을 구하고 평균(reduce_mean)구하기 
# 최소 평균 거리 구하는 식
cost = tf.reduce_mean(tf.square(hy - y))

# Minimize - 기본으로 사용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost) # cost 식의  최소화 모델 정의.. tf에서 미분(기울기)을 자동적용

# makeru graph
sess = tf.Session()

# Initializes global variables in the graph.
# tf variable 사용시 필수.
sess.run(tf.global_variables_initializer())

# # 반복 시작.
for step in range(2001): #0부터 2001번 반복
   cost_val, w_val, b_val, _ = sess.run([cost, w, b, train], 
    feed_dict={x: [1, 2, 3, 4, 5 ], y: [2.1, 3.1, 4.1, 5.1, 6.1]})
   if step % 20 == 0: # 20번에 한번씩 아래 실행
       print(step, cost_val, w_val, b_val)

print(sess.run(hy, feed_dict={x:[1.5, 3, 5]}))


