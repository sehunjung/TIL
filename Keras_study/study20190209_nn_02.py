
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
import numpy as np
import pandas as pd

x1 = np.array(range(1,11))
np.transpose(x1)
y1 = np.array(range(1,11))
np.transpose(y1)

x2 = np.array(range(101,111))
np.transpose(x2)
y2 = np.array(range(101,111))
np.transpose(y2)
# np.array(range(1,11))

# z =  np.hstack([x1, x2])

# print("x shape :", x1.shape)
# print("x type :" , type(x1))
# print("x dim :" , z.ndim)
# print(z, x1)

x1_train = x1[:7]
y1_train = y1[:7]
x1_test = x1[7:]
y1_test =y1[7:]

x2_train = x2[:7]
y2_train = y2[:7]
x2_test = x2[7:]
y2_test = y2[7:]

# input layer
visible1 = Input(shape=(1,))
visible2 = Input(shape=(1,))

# first 
Den1_10 = Dense(10, activation='relu')(visible1)
Den1_1 = Dense(100)(Den1_10)
Den1_1 = Dense(100)(Den1_1)
Den1_1 = Dense(100)(Den1_1)


# second 
Den2_10 = Dense(10, activation='relu')(visible2)
Den2_1 = Dense(100)(Den2_10)
Den2_1 = Dense(100)(Den2_1)
Den2_1 = Dense(100)(Den2_1)


# merge 
merge = concatenate([Den1_1, Den2_1])

# two output
output1 = Dense(1)(merge)
output2 = Dense(1)(merge)

model = Model(inputs=[visible1,visible2], outputs=[output1, output2])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# summarize layers
print(model.summary())

model.fit([x1_train, x2_train], [y1_train, y2_train], 
            epochs=200, batch_size=1,
            validation_data=([x1_test, x2_test], [y1_test, y2_test]))

# model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test), verbose=0)


# a1, b1 = model.evaluate(x1_test, y1_test, batch_size=1)
# print("A1 Loss :", a1, "MSE : ", b1)


# a2, b2 = model.evaluate(x2_test, y2_test, batch_size=1)
# print("A1 Loss :", a2, "MSE : ", b2)


y_predict = model.predict([x1_test, x2_test])
print("Prediction :" ,y_predict)
# 두개의 피쳐를 하나로 펼쳐서 넣어야 R2 가 나옴
y_predict = np.array(y_predict)
print("y_predict : ", y_predict.shape)
y_predict = y_predict.flatten()


# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(np.array([x1_test, y1_test]).flatten(), y_predict)
print("R2 : ", r2_y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error, mean_absolute_error
def RMSE(y12_test, y_predict):
    return np.sqrt(mean_squared_error(y12_test, y_predict))
print("RMSE : ", RMSE((np.array([y1_test, y2_test])).flatten(), y_predict))
