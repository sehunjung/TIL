
import tensorflow as tf
import matplotlib.pyplot as plt

# f(x) = W*X
# W = tf.Variable(tf.random_normal([1]), name='weight')

W = tf.placeholder(tf.float32)

# X = tf.placeholder(tf.float32)
X = [1, 2, 3]
# Y = tf.placeholder(tf.float32)
Y = [2, 4, 6]

# 가정... 이론...
hy = W * X

# cost loss function...
# 가정 - 실제 제곱 합산....
cost = tf.reduce_sum(tf.square(hy - Y))

# W의 최적화...
# 미분은 https://www.derivative-calculator.net/ 에 수식을 넣으면 계산해 줌
# (wx -y)^2 => 2w(wx-y) => 2(wx-y)x => (wx-y)x : 2배수로 나누어도 크게 차이 없으므로 간소화
# learning_rate = 0.1 #학습 상수
# gradient = tf.reduce_mean((W * X - Y) * X ) # W최소값 미분 공식 적용
# descent = W - learning_rate * gradient # 새로운 W에 학습 상수 적용
# update = W.assign(descent) # W에 새로 학습된 값을 업데이트해서 update 입력

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for step in range(21):
#    sess.run(update, feed_dict={X: [1,2,3], Y: [1,2,3]}) #업데이트 된 W로 학습
#    print(step, sess.run(cost, feed_dict={X: [1,2,3], Y: [1,2,3]}), sess.run(W))

#계산값 저장할 array 선언
W_val = []
cost_val = [] 
for i in range(-30, 50):
   feed_W = i * 0.1
   # cost 수식에 0.1씩 증가식키면서 학습..현재 cost, W 저장...
   curr_cost, curr_W = sess.run([cost, W], feed_dict={W:feed_W})
   W_val.append(curr_W)
   cost_val.append(curr_cost)
  # print(curr_W, curr_cost)

plt.plot(W_val, cost_val)
plt.show()




