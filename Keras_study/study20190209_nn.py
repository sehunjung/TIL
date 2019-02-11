
import numpy as np
import pandas as pd
from keras.layers import Dense,Activation
from keras.models import Sequential

# numpy array는  리스트를 받아들인다, 스칼라 애러남..
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# np.array(range(1,11))

# 모델의 인풋/아웃풋 shape, dimension이 중요하다.
print("x shape:", x.shape)
print("x type", type(x))

# 트레인/테스트 7:3
x_train = x[:7]
y_train = y[:7]
x_test = x[7:]
y_test =y[7:]


print("x_trin:", x_train, x_train.shape)
print("x_test:", x_test, x_test.shape)


# 모델 구성

model = Sequential()
# 1개의 인풋 10개의 아웃풋...
# 첫째 층은 인풋이 중요 - shape, dim 모두가능...
# model.add(Dense(10, input_dim = 1, activation='relu'))
model.add(Dense(10, input_shape=(1,), activation='relu'))
# 마지막 층에서 1개로 결과 도출..
model.add(Dense(1))

# 옵티마이저 정리 필요..
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])

# 모델의 파라미터 및 shape, dim 을 꼭 확인 
# 예를 들어 10개의 노드가 있지만 실제로는 bias가 있어 11개의 노드가 됨(y=wx+b) 
# 각노드의 파라미터 계산시 노드갯수 + 1 을 해서 계산 해야 함.
model.summary()

# fit 함수의 기본 배치는 32...
# model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test))
model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test), verbose=0)


# 학습된 모델을 평가...loss/mse,acc 를 반환
a, b = model.evaluate(x_test, y_test, batch_size=1)
print("Loss :", a, "MSE : ", b)

# 예측.....
y_predict = model.predict(x_test)
print("Prediction :" ,y_predict)


# R2 구하기
# 의미파악 필요 - 높을수록 정확함.
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)


# RMSE 구하기
# 실제와 예측의 차이 - 작을수록 정확함.
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))