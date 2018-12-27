
import tensorflow as tf

filename_queue = tf.train.string_input_producer(
   #['C:\==Task\Github\TIL\TensorFlow\data-01-test-score.csv','C:\==Task\Github\TIL\TensorFlow\data-02-test-score.csv'], shuffle=False, name='filename_queue')
   ['C:\==Task\Github\TIL\TensorFlow\data-01-test-score.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)


# data의 구조를 확인하고 type, shape를 맞춰야 함.
record_defaults = [[0.], [0.], [0.], [0.]]
# test, 결과 데이터를 일단 한번에 가져옴...
xy = tf.decode_csv(value, record_defaults=record_defaults)

# 배치로 읽어들인 데이터 분리...테스트, 결과.
train_x_batch, train_y_batch = \
   tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3]) # shape에 유의..
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight') # x,y의 feature로 정의...
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis 수식 생성...매트릭스 계산.
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
# hy -> cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize  cost -> train
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    #batch에서 받아온 데이터 할당...
   x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
   cost_val, hy_val, _ = sess.run(
       [cost, hypothesis, train], 
       feed_dict={X: x_batch, Y: y_batch}) # 할당받은 데이터 학습...
   if step % 10 == 0:
       print(step, "Cost: ", cost_val, 
                   "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)
