
import numpy as np
import pandas as pd
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate

x1 = np.array(range(1,11))
np.transpose(x1)
# 배열을 피쳐 형식으로 세로로 변환..
y1 = np.array(range(1,11))
np.transpose(y1)

x2 = np.array(range(101,111))
np.transpose(x2)
y2 = np.array(range(101,111))
np.transpose(y2)
# np.array(range(1,11))


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
# 1열로 받아 들일....(1,) => 1열, x행을 의미함...
visible1 = Input(shape=(1,))
visible2 = Input(shape=(1,))

# first with 2 input
Inp = concatenate([visible1, visible2])

#model
Den1_10 = Dense(10, activation='relu')(Inp)
Den1_1 = Dense(100)(Den1_10)
Den1_1 = Dense(100)(Den1_1)
Den1_1 = Dense(100)(Den1_1)
Den1_1 = Dense(2)(Den1_1)

output1 = Dense(1)(Den1_1)
output2 = Dense(1)(Den1_1)

model = Model(inputs=[visible1,visible2], outputs=[output1, output2])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# summarize layers
print(model.summary())


model.fit([x1_train, x2_train], [y1_train, y2_train], 
        epochs=200, 
        batch_size=1,
        validation_data=([x1_test, x2_test], [y1_test, y2_test]), verbose=0)
# model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test), verbose=0)


## loss를 여러개 받는 방법은???
# a = []
# a, b = model.evaluate([[x1_test, x2_test], [y1_test, y2_test], batch_size=1)
# print("Loss :", a, "MSE : ", b)

y_predict = model.predict([x1_test, x2_test])
print("Prediction :" ,y_predict)


# 두개의 피쳐를 하나로 펼쳐서 넣어야 R2 가 나옴
y_predict = np.array(y_predict)
print("y_predict : ", y_predict.shape)
y_predict = y_predict.flatten()

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
