import numpy as np
import pandas as pd
import tensorflow as tf

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
# from keras.utils import np_utils


a = np.array(range(1,111))
b = np.array(range(1001, 1111))

window_size = 11

# def split_11(seq, window_size):  # 데이터를 11개씩 자르기용.    
#     aaa = []
#     for i in range(len(a)-window_size +1):                 # 열
#         subset = a[i:(i+window_size)]       # 0~5
#         aaa.append([item for item in subset])
#         # print(aaa)
#     print(type(aaa))    
#     return np.array(aaa)

def split_11(x, window_size):  # 데이터를 11개씩 자르기용.    
    aaa = []
    for i in range(len(x)-window_size +1):                 # 열
        subset = x[i:(i+window_size)]       # 0~5
        aaa.append([item for item in subset])
        # print(aaa)
    print(type(aaa))    
    return np.array(aaa)

dataset = split_11(a, window_size)     # window_size 만큼씩 잘라진다.
datasetb = split_11(b, window_size)     # window_size 만큼씩 잘라진다.
print("===========================")
print("DS A :", dataset)
print("DS A shape:",dataset.shape)    # (100, 11)

print("DS B:", datasetb)
print("DS B shape:", datasetb.shape)    # (100, 11)

print("===========================")
# train, label 분리 1~10 , 11 
x_train = dataset[:,0:10]
y_train = dataset[:,10]
print('x_train:', x_train[0], 'y_train :', y_train[0])

xb_train = datasetb[:,0:10]
yb_train = datasetb[:,10]
print('xb_train:', xb_train[0], 'yb_train :', yb_train[0])

print("===========================")
x_train = np.reshape(x_train, (len(a)-window_size+1, 10, 1))
print('len(a) :', len(a))
print('x_train.shape :', x_train.shape)    # (100, 10, 1)

xb_train = np.reshape(xb_train, (len(b)-window_size+1, 10, 1))
print('len(b) :', len(b))
print('xb_train.shape :', xb_train.shape)    # (100, 10, 1)


# 모델 구성하기
# model A
input1 = Input(shape=(10,1))

lstm1 = LSTM(32, input_shape=(10,1), return_sequences=True)(input1)
lstm1 = LSTM(10)(lstm1)
dense1 = Dense(5, activation='relu')(lstm1)

output1 = Dense(1)(dense1)

modelA = Model(inputs=input1, outputs=output1)
modelA.summary()

modelA.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
ta_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
modelA.fit(x_train, y_train, epochs=300, batch_size=1, verbose=2, callbacks = [ta_hist])


# model B
input2 = Input(shape=(10,1))

lstm2 = LSTM(32, input_shape=(10,1), return_sequences=True)(input2)
lstm2 = LSTM(10, return_sequences=True)(lstm2)
lstm2 = LSTM(10, return_sequences=True)(lstm2)
lstm2 = LSTM(10)(lstm2)
dense2 = Dense(5, activation='relu')(lstm2)

output2 = Dense(1)(dense2)

modelB = Model(inputs=input2, outputs=output2)
modelB.summary()

modelB.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
modelB.fit(xb_train, yb_train, epochs=500, batch_size=1, verbose=2, callbacks = [tb_hist])

# model  save
from keras.models import load_model
import json

#save model A json
# modelA.save_weights('lstm_a_W.h5')
# with open('lstm_a.json', 'w') as af:
#     af.write(modelA.to_json())

modelA.save('lstm_A.h5')

print("Saved modelA from disk")

#save model B json
# modelB.save_weights('lstm_b_W.h5')
# with open('lstm_b.json', 'w') as bf:
#     bf.write(modelB.to_json())
# json으로 모델 저장시 
modelB.save('lstm_B.h5')

print("Saved modelB from disk")