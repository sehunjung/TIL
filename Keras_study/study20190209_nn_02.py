
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
y1 = np.array(range(1,11))

x2 = np.array(range(101,111))
y2 = np.array(range(101,111))
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
Den1_1 = Dense(1)(Den1_10)

# second 
Den2_10 = Dense(10, activation='relu')(visible2)
Den2_1 = Dense(1)(Den2_10)

# merge 
merge = concatenate([Den1_1, Den2_1])
merge = Dense(1)(merge) #마지막 결과 1개로 
# final layer
# output = Dense(1, activation='relu')(merge)

model = Model(inputs=[visible1,visible2], outputs=merge)

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# summarize layers
print(model.summary())

model.fit([x1_train, x2_train], y1_train, 
            epochs=200, batch_size=1,
            validation_data=([x1_test, x2_test], y1_test), verbose=0)

# model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test), verbose=0)


'''
a1, b1 = model.evaluate(x1_test, y1_test, batch_size=1)
print("Loss :", a1, "MSE : ", b1)

a2, b2 = model.evaluate(x2_test, y2_test, batch_size=1)
print("Loss :", a2, "MSE : ", b2)


y1_predict, y2_predict = model.predict([x1_test, x2_test])
print("Prediction :" ,y1_predict, y2_predict)


# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
'''