import numpy as np
import pandas as pd
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

np.transpose(x10)
np.transpose(x20)
np.transpose(x3)
# x3.flatten()

np.transpose(y10)
np.transpose(y20)
np.transpose(y3)
# y3.flatten()


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

# for load model....weights
from keras.models import load_model
import json
from keras.models import model_from_json


with open('cnt_01.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('cnt_01_W.h5')

#hdf5 binary format
# model = load_model('cnt_01.h5')
# model.load_weights('cnt_01_W.h5')

print("Loaded model from disk")

# model check.
model.summary()

# predic
y_predict = model.predict([x1_test, x2_test, x3_test])
print("y_predic", y_predict)        


y_predict = np.array(y_predict)
# for 1 line...
y_predict = y_predict.flatten()

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score((np.array([y1_test, y2_test, y3_test])).flatten(), y_predict)
print("R2 : ", r2_y_predict)    

# RMSE 구하기
from sklearn.metrics import mean_squared_error, mean_absolute_error
def RMSE(y12_test, y_predict):
    return np.sqrt(mean_squared_error(y12_test, y_predict))
print("RMSE : ", RMSE((np.array([y1_test, y2_test, y3_test])).flatten(), y_predict))