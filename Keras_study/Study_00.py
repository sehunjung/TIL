from keras import models
from keras import layers

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2.5)

import warnings
warnings.filterwarnings('ignore')


x_train = [1,2,3,4,5,6,7,8,9]
x_label = [1,2,3,4,5,6,7,8,9]

one_hot_x_train = to_categorical(x_train)
one_hot_x_label = to_categorical(x_label)


x_val = one_hot_x_train[:3]
partial_x_train = one_hot_x_train[3:]

y_val = one_hot_x_label[:3]
partial_x_label = one_hot_x_label[3:]

from keras.layers.core import Dense, Dropout

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(10,)))
model.add(Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(10, activation='relu'))
model.add(Dropout(0.2))

model.add(layers.Dense(10, activation='softmax'))

model.summary()


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(partial_x_train,
                    partial_x_label,
                    epochs=500,
                    batch_size=128,
                    validation_data=(x_val, y_val),
                    verbose=0)


hists = [history]
hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists])
hist_df.index = np.arange(1, len(hist_df)+1)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(hist_df.val_acc, lw=5, label= 'V_Acc')
ax[0].plot(hist_df.acc, lw=5, label='T_Acc')
ax[0].set_ylabel('Acc')
ax[0].set_xlabel('Epoch')
ax[0].grid()
ax[0].legend(loc=0)


ax[1].plot(hist_df.val_loss, lw=5, label= 'V_Loss')
ax[1].plot(hist_df.loss, lw=5, label='T_loss')
ax[1].set_ylabel('loss')
ax[1].set_xlabel('Epoch')
ax[1].grid()
ax[1].legend(loc=0)
plt.show()