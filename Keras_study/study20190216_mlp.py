import numpy as np
import pandas as pd
from keras.layers import Dense,Activation,Flatten
from keras.layers.core import Flatten
from keras.models import Sequential

x = np.array([[1,2,3,4,5,6,7,8,9,10], [101,102,103,104,105,106,107,108,109,110]])
y = np.array([[1,2,3,4,5,6,7,8,9,10], [101,102,103,104,105,106,107,108,109,110]])
# x = np.array([range(1,11),range(101,111)])

print('x.shape :', x.shape) # (2,10)
print('type(x) :', type(x))
print(x)


x=np.transpose(x)   # (10,2)
y=np.transpose(y)

# x = np.reshape(10,2)
# y = np.reshape(10,2)

# x_train = x[:,:7]
# y_train = y[:,:7]
# x_test = x[:,7:]
# y_test =y[:,7:]

x_train = x[:7,]    # (7,2)
y_train = y[:7,]
x_test = x[7:,]     # (3,2)
y_test =y[7:,]

print('x_train : ', x_train)
print('x_test : ', x_test)
print('x_train.shape :', x_train.shape)
print('x_test.shape :', x_test.shape)



# 모델 구성

model = Sequential()
model.add(Dense(100, input_dim = 2, activation='relu'))
# model.add(Dense(100000, input_shape=(2, ), activation='relu'))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
# model.add(Flatten())
model.add(Dense(2))


model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.summary()


model.fit(x_train, y_train, epochs=500, batch_size=1, validation_data=(x_test, y_test))


a, b = model.evaluate(x_test, y_test, batch_size=1)  # a[1]
print(a, b)     

y_predict = model.predict(x_test)
print(y_predict)


# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)    

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))


# loss = 0.00045
# R2 = 0.999
# RMSE : 0.021