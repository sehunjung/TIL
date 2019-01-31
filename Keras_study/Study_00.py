from keras import models
from keras import layers

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns
from keras.layers.core import Dense, Dropout

plt.style.use('seaborn')
sns.set(font_scale=2.5)

import warnings
warnings.filterwarnings('ignore')

# type 변경을 몰라서 입력을 float으로  입력
# x_train = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])
# x_target = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])
x_train = np.array([1,2,3,4,5,6,7,8,9]).astype('float32')
x_target = np.array([1,2,3,4,5,6,7,8,9]).astype('float32')

mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std

x_target -= mean
x_target /= std

# x_train -= np.mean(x_train)
# x_train /= np.std(x_train)

# print(x_target.shape)
# print(x_target)

k = 4
num_val_samples = len(x_train) // k
for i in range(k):
    print('processing fold', i)
    val_data = x_train[i * num_val_samples: (i+1)*num_val_samples]
    val_target = x_target[i * num_val_samples: (i+1)*num_val_samples]

    partial_train_data = np.concatenate(
        [x_train[:i * num_val_samples],
        x_train[(i+1) * num_val_samples:]], axis = 0)

    partial_train_target = np.concatenate(
        [x_target[:i * num_val_samples],
        x_target[(i+1) * num_val_samples:]],axis = 0) 

# print(val_data.shape)
# print(val_data)
# print(val_target.shape)
# print(val_target)

# print(partial_train_data.shape)
# print(partial_train_data)
# print(partial_train_target.shape)
# print(partial_train_target)


model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(9,)))
model.add(layers.Dense(64, activation='relu', input_shape=(1,)))
model.add(Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])


num_epochs = 500
all_histories = []
history = model.fit(partial_train_data,
                    partial_train_target,
                    validation_data=(val_data, val_target),
                    epochs=num_epochs,
                    batch_size=1,
                    verbose=0)


hists = history.history['val_mean_absolute_error']
all_histories.append(hists)

average_ame_history = [np.mean([x[i] for x  in all_histories]) for i in range(num_epochs)]

# 두 그래프 하나로 그리기 찾아야 함....
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 16))

# graph 1
plt.subplot(121) # 1,2 의 첫번째
plt.plot(range(1, len(average_ame_history) + 1), average_ame_history)
plt.xlabel('Epochs') 
plt.ylabel('Validation MAE')

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smoothed_mae_history = smooth_curve(average_ame_history)

# graph 2
plt.subplot(122) #1,2 의 두번째
plt.plot(range(1, len(smoothed_mae_history)+1 ), smoothed_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()