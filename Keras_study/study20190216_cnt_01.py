

import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Activation, Concatenate, Input
from keras.models import Sequential, Model
from keras.layers.merge import concatenate

x1 = np.array(range(1,11))
x10 = np.concatenate((x1,x1,x1,x1,x1,x1,x1,x1,x1,x1))
x2 = np.array(range(101,111))
x20 = np.concatenate((x2,x2,x2,x2,x2,x2,x2,x2,x2,x2))
x3 = np.array(range(1001, 1101))

y1 = np.array(range(1,11))
y10 = np.concatenate((y1,y1,y1,y1,y1,y1,y1,y1,y1,y1))
y2 = np.array(range(101,111))
y20 = np.concatenate((y2,y2,y2,y2,y2,y2,y2,y2,y2,y2))
y3 = np.array(range(1001, 1101))
x=np.array([range(1,11),range(101,111)])

# print('x1.shape :', x1.shape) 
# print('type(x1) :', type(x1))
# print(x1)


np.transpose(x10)
np.transpose(x20)
np.transpose(x3)
# x3.flatten()

np.transpose(y10)
np.transpose(y20)
np.transpose(y3)
# y3.flatten()

# print("x shape", x10.shape, x20.shape, x3.shape)
# print("x shape", x10, x20, x3)
# print("y shape", y10.shape, y20.shape, y3.shape)



x1_train = x10[:70]
x2_train = x20[:70]     
x3_train = x3[:70]  
y1_train = y10[:70]
y2_train = y20[:70]
y3_train = y3[:70]

x1_test = x10[70:]  
x2_test = x20[70:] 
x3_test = x3[70:] 
y1_test = y10[70:]  
y2_test = y20[70:]
y3_test = y3[70:]


# print('x1_train : ', x1_train)
# print('x1_test : ', x1_test)
# print('x1_train.shape :', x1_train.shape)
# print('x1_test.shape :', x1_test.shape)


# 모델 구성
# model 1
input1 = Input(shape=(1,))
dense1 = Dense(100, activation='relu')(input1)

# model 2
input2 = Input(shape=(1,))
dense2 = Dense(50, activation='relu')(input2)

# model 3
input3 = Input(shape=(1,))
dense3 = Dense(50, activation='relu')(input3)

# mere
# merge = Concatenate()([dense1, dense2])
merge1 = concatenate([dense1, dense2, dense3])

# output
output_1 = Dense(100)(merge1)
output1 = Dense(1)(output_1)

output_2 = Dense(100)(merge1)
output2 = Dense(1)(output_2)

output_3 = Dense(100)(merge1)
output3 = Dense(1)(output_3)

model = Model(inputs=[input1, input2, input3], outputs=[output1, output2, output3])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.summary()

# tensorboard log
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train, y3_train] , epochs=100, batch_size=1,
          validation_data=([x1_test, x2_test, x3_test], [y1_test, y2_test, y3_test]), 
          callbacks = [tb_hist]) # tensorboard log 


#mac os keras.json 
# cmd + shift + g => ~/.keras

#DIR
# /Users/sehun.jung/Downloads/TIL/Keras_study/graph
# tensorboard --logdir=/Users/sehun.jung/Downloads/TIL/Keras_study/graph
# http://localhost:6006

## loss는 어떻게???
# a, b = model.evaluate([x1_test, x2_test, x3_test],[y1_test, y2_test, y3_test], batch_size=1)
# print("loss: ", a,"mse:", b)     

y_predict = model.predict([x1_test, x2_test, x3_test])   # list
print("y_predic", y_predict)        

# y_predict = y_predict.value()
y_predict = np.array(y_predict)

# print(type(y_predict))
# print(y_predict.shape)

# for 1 line...
y_predict = y_predict.flatten()
# print(y_predict)        

# print(type(y_predict))
# print(y_predict.shape)

# print('끗')

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score((np.array([y1_test, y2_test, y3_test])).flatten(), y_predict)
# r2_y_predict = r2_score(y1_test + y2_test, y_predict)
print("R2 : ", r2_y_predict)    

# RMSE 구하기
from sklearn.metrics import mean_squared_error, mean_absolute_error
def RMSE(y12_test, y_predict):
    return np.sqrt(mean_squared_error(y12_test, y_predict))
print("RMSE : ", RMSE((np.array([y1_test, y2_test, y3_test])).flatten(), y_predict))



# model save
from keras.models import load_model
import json

#save model json
model.save_weights('cnt_01_W.h5')
with open('cnt_01.json', 'w') as f:
    f.write(model.to_json())

#hdf5 binary format
# model.save('cnt_01.h5')
# model.save_weights('cnt_01_W.h5')

print("Saved model from disk")

# loss = 0.0597
# R2 = 0.999
# RMSE : 0.17