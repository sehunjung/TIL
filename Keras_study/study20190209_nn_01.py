
import numpy as np
import pandas as pd
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate

x1 = np.array(range(1,11))
y1 = np.array(range(1,11))

x2 = np.array(range(101,111))
y2 = np.array(range(101,111))
# np.array(range(1,11))

# print("x shape :", x.shape)
# print("x type :" , type(x))
# print("x dim :" , x.ndim)

x1_train = x1[:7]
y1_train = y1[:7]
x1_test = x1[7:]
y1_test =y1[7:]

x2_train = x2[:7]
y2_train = y2[:7]
x2_test = x2[7:]
y2_test =y2[7:]

print(x2_train)
print(x2_test)


# input layer
visible1 = Input(shape=(1,))
visible2 = Input(shape=(1,))


# first with 2 input
Inp = concatenate([visible1, visible2])
Den1_10 = Dense(10, activation='relu')(Inp)
Den1_1 = Dense(1)(Den1_10)

model = Model(inputs=[visible1,visible2], outputs=Den1_1)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# summarize layers
print(model.summary())


model.fit([x1_train, x2_train], y1_train, 
        epochs=200, 
        batch_size=1,
        validation_data=([x1_test, x2_test], y1_test), verbose=0)
# model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test), verbose=0)

'''
a, b = model.evaluate(x_test, y_test, batch_size=1)
print("Loss :", a, "MSE : ", b)

y_predict = model.predict(x_test)
print("Prediction :" ,y_predict)


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