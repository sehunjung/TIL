
from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

import numpy as np

def vectorize_squences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))
  for i , squence in enumerate(sequences):
    results[i, squence] = 1
  return results

x_train = vectorize_squences(train_data)
x_test = vectorize_squences(test_data)

def to_one_hot(labels, dimension=46):
  results = np.zeros((len(labels), dimension))
  for i, label in enumerate(labels):
    results[i, label] = 1.
  return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                   verbose=0)

results = model.evaluate(x_test, one_hot_test_labels)
print("results")


predictions = model.predict(x_test)

arg_pre = []
for num in range(len(predictions)):
  arg_pre.append(np.argmax(predictions[num]))
print(arg_pre)