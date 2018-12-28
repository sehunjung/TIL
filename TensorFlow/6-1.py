import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility

# 파일이 아닌 db나 json에서 받아 오게 하려면??
filename_queue = tf.train.string_input_producer(
   ['C:\==Task\Github\TIL\TensorFlow\data-04-zoo.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# data의 구조를 확인하고 type, shape를 맞춰야 함.
# 13개 구조를 한번에 정의 하지 못할까??
# record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], 
# [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
record_defaults = [[0.]] * 17
# test, 결과 데이터를 일단 한번에 가져옴...
xy = tf.decode_csv(value, record_defaults=record_defaults)

# 배치로 읽어들인 데이터 분리...테스트, 결과.
train_x_batch, train_y_batch = \
   tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

#결과의 범주 
nb_classes = 7  # 0 ~ 6

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 16]) # shape에 유의..

Y = tf.placeholder(tf.int32, shape=[None, 1]) #one hot 사용하려면 int32 type
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot으로 변환...shape가 늘어남...차원이 늘어남.. 
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # on hot으로 shape를 교정.

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight') # x,y의 feature로 정의...
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis 수식 생성.. logit과 hyothesis 분리
logits = tf.matmul(X, W) + b  # 다중 행렬 계산
hypothesis = tf.nn.softmax(logits) # 시그노이드, 소프트맥스 처리 =>확률로 나옴.

# Simplified cost/loss function
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 예측 모델
prediction = tf.argmax(hypothesis, 1) #예측을 모델에 대응 -> 0~6 사이 값을 리턴
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1)) # 예측/실제 평가 => 1,0 리턴
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 평가 평균 계산 => 정확도

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
    #학습
    sess.run(optimizer, feed_dict={X: x_batch, Y: y_batch})
    if step % 100 == 0:
        loss, acc = sess.run([cost, accuracy], feed_dict={
                            X: x_batch, Y: y_batch})

        #python 기본 문법 학습 필요
        print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
            step, loss, acc))

coord.request_stop()
coord.join(threads)

# Let's see if we can predict
pred = sess.run(prediction, feed_dict={X: x_batch})
# y_data: (N,1) = flatten => (N, ) matches pred.shape
#python 기본 문법 학습 필요
for p, y in zip(pred, y_batch.flatten()):
    #python 기본 문법 학습 필요
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

# # 윈도우나 web에 데이타 및 그래프 그릴려면??

