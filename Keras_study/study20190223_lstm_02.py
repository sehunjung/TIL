import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Concatenate, Input
from keras.models import Sequential, Model
from keras.layers.merge import concatenate


# for load model....weights
from keras.models import load_model
import json
from keras.models import model_from_json

# with open('lstm_a.json', 'r') as af:
#     modelA = model_from_json(af.read())
# modelA.load_weights('lstm_a_W.h5')
modelA = load_model('lstm_A.h5')
print("Loaded modelA from disk")

# with open('lstm_b.json', 'r') as bf:
#     modelB = model_from_json(bf.read())
# modelB.load_weights('lstm_b_W.h5')
modelB = load_model('lstm_B.h5')
print("Loaded modelB from disk")


# model check.
modelA.summary()
modelB.summary()


# modelA.load_weights('lstm_a_W.h5')
# modelB.load_weights('lstm_b_W.h5')

# models = [modelA, modelB]

# model_input = Input((4,10,1))

# def ensemble(models, model_input):

#     outputs = [model(model_input) for model in models]
#     y = Average()(outputs)
#     model = Model(inputs = model_input, outputs = y, name='ensemble')
#     return model

# ensemble_model = ensemble(models,model_input)
# ensemble_model.summary()


# test data
ax_test = np.array([[range(1,11)],[range(11,21)],[range(21,31)],[range(31,41)]])
print("ax_test.shape :", ax_test.shape)
ax_test = np.reshape(ax_test, (4, 10, 1))
print("ax_test :", ax_test)
ay_test = np.array([11, 21, 31, 41])

bx_test = np.array([[range(1001,1011)],[range(1011,1021)],[range(1021,1031)],[range(1031,1041)]])
print("bx_test.shape :", bx_test.shape)
bx_test = np.reshape(bx_test, (4, 10, 1))
print("bx_test :", bx_test)
by_test = np.array([1011, 1021, 1031, 1041])

print("ax_test :", ax_test)
print("ay_test :", ay_test)
print("===========================")
print("bx_test :", bx_test)
print("by_test :", by_test)


#predict
loss_a, acc_a = modelA.evaluate(ax_test, ay_test)
a_predict = modelA.predict(ax_test)

print('loss_a : ', loss_a)
print('acc_a : ', acc_a)
print("===========================")

loss_b, acc_b = modelA.evaluate(bx_test, by_test)
b_predict = modelB.predict(bx_test)

print('loss_b : ', loss_b)
print('acc_b : ', acc_b)

# predict sum

total_predict = a_predict + 0.25 * b_predict

print('a_predic : ', a_predict)
print('b_predic : ', b_predict)
print("===========================")
print('total_predic : ', total_predict)


# R2 구하기
from sklearn.metrics import r2_score
r2_total_predict = r2_score(ay_test, total_predict)
print("A R2 : ", r2_total_predict)    

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(ay_test, total_predict))
print("A MSE : ", RMSE(ay_test, total_predict))

