

import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Concatenate, Input
from keras.models import Sequential, Model
from keras.layers.merge import concatenate

x1 = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([101,102,103,104,105,106,107,108,109,110])
y1 = np.array([1,2,3,4,5,6,7,8,9,10])
y2 = np.array([101,102,103,104,105,106,107,108,109,110])
# x=np.array([range(1,11),range(101,111)])

print('x1.shape :', x1.shape) 
print('type(x1) :', type(x1))
print(x1)

x1_train = x1[:7]
x2_train = x2[:7]       
y1_train = y1[:7]
y2_train = y2[:7]
x1_test = x1[7:]  
x2_test = x2[7:]  
y1_test =y1[7:]  
y2_test =y2[7:]

print('x1_train : ', x1_train)
print('x1_test : ', x1_test)
print('x1_train.shape :', x1_train.shape)
print('x1_test.shape :', x1_test.shape)


# 모델 구성
# model 1
input1 = Input(shape=(1,))
dense1 = Dense(100, activation='relu')(input1)
# model 2
input2 = Input(shape=(1,))
dense2 = Dense(50, activation='relu')(input2)
# merge
# merge = Concatenate()([dense1, dense2])
merge1 = concatenate([dense1, dense2])

output_1 = Dense(100)(merge1)
output1 = Dense(1)(output_1)
output_2 = Dense(100)(merge1)
output2 = Dense(1)(output_2)


model = Model(inputs=[input1, input2], outputs=[output1, output2])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.summary()


model.fit([x1_train, x2_train], [y1_train, y2_train] , epochs=100, batch_size=1,
          validation_data=([x1_test, x2_test], [y1_test, y2_test]))

# a, b = model.evaluate([x1_test, x2_test],[y1_test, y2_test], batch_size=1)
# print(a, b)     

y_predict = model.predict([x1_test, x2_test])   # list
print(y_predict)        

# y_predict = y_predict.value()
y_predict = np.array(y_predict)

print(type(y_predict))
print(y_predict.shape)


y_predict = y_predict.flatten()
print(y_predict)        

print(type(y_predict))
print(y_predict.shape)

print('끗')

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score((np.array([y1_test, y2_test])).flatten(), y_predict)
# r2_y_predict = r2_score(y1_test + y2_test, y_predict)
print("R2 : ", r2_y_predict)    

# RMSE 구하기
from sklearn.metrics import mean_squared_error, mean_absolute_error
def RMSE(y12_test, y_predict):
    return np.sqrt(mean_squared_error(y12_test, y_predict))
print("RMSE : ", RMSE((np.array([y1_test, y2_test])).flatten(), y_predict))


# loss = 0.0597
# R2 = 0.999
# RMSE : 0.17