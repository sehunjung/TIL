
import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
# h(x) = wx + b => matul => sigmoid => 가설 지정
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# 실제 가설 최소화 구조.. log식에 그대로 대입...
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                      tf.log(1 - hypothesis))

# train을 한줄로 표현
# cost -> train.
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#예상이 0.5 이상이면 참 => 1로 변환
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# 예상과 실제가 같으면 1, 아니면 0 로 변환해서 평균 구함.=> 정확도.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


# Launch graph
# sess 선언방식 바뀜.
with tf.Session() as sess:
   # Initialize TensorFlow variables -초기화
   sess.run(tf.global_variables_initializer())

   for step in range(10001):
       #입력 값으로 학습
       cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
    #    if step % 200 == 0:
    #        print(step, cost_val)


 # Accuracy report
   h, c, a = sess.run([hypothesis, predicted, accuracy],
                      feed_dict={X: x_data, Y: y_data})

   print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

