
import tensorflow as tf
#import numpy as np # file을 읽기 위해 필요...

# xy = np.loadtxt('C:\==Task\Github\TIL\TensorFlow\data-03-diabetes.csv', delimiter=',', dtype=np.float32)

# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]

filename_queue = tf.train.string_input_producer(
   ['C:\==Task\Github\TIL\TensorFlow\data-03-diabetes.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# data의 구조를 확인하고 type, shape를 맞춰야 함.
record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
# test, 결과 데이터를 일단 한번에 가져옴...
xy = tf.decode_csv(value, record_defaults=record_defaults)

# 배치로 읽어들인 데이터 분리...테스트, 결과.
train_x_batch, train_y_batch = \
   tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight') #shape 확인...
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])

    # x, y를 하나의 변수에 넣어서 feed 입력 가능.
    # feed = {X: x_data, Y: y_data}
    feed = {X: x_batch, Y: y_batch}
    for step in range(10001):
       sess.run(train, feed_dict=feed)
       if step % 200 == 0:
           print(step, sess.run(cost, feed_dict=feed))

    coord.request_stop()
    coord.join(threads)

   # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

