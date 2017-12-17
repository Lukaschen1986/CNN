# -*- coding: utf-8 -*-
from __future__ import absolute_import # 模块绝对引入
from __future__ import division # 精确除法
from __future__ import print_function
import os
os.getcwd()
#os.chdir("D:/my_project/Python_Project/test/NN/cifar-10-python")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from scipy.stats import itemfreq
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import cnn_layers_tf as clt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras import backend as K
K.image_data_format()
K.set_image_data_format('channels_first')
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.models import load_model

def load_CIFAR_batch(path, data_format):
    """ load single batch of cifar """
    with open(path, "rb") as f:
        dicts = pickle.load(f, encoding="latin1")
        X = dicts["data"]
        y = dicts["labels"]
        label = dicts["filenames"]
        if data_format == "channels_first":
            X = X.reshape(10000, 3, 32, 32)
        elif data_format == "channels_last":
            X = X.reshape(10000, 32, 32, 3)
        else:
            raise ValueError("data_format must in ('channels_first', channels_last)")
        y = np.array(y)
        return X, y, label
    
x_1, y_1, label_1 = load_CIFAR_batch("./cifar-10-python/data_batch_1", "channels_first")
x_2, y_2, label_2 = load_CIFAR_batch("./cifar-10-python/data_batch_2", "channels_first")
x_3, y_3, label_3 = load_CIFAR_batch("./cifar-10-python/data_batch_3", "channels_first")
x_4, y_4, label_4 = load_CIFAR_batch("./cifar-10-python/data_batch_4", "channels_first")
x_5, y_5, label_5 = load_CIFAR_batch("./cifar-10-python/data_batch_5", "channels_first")
x_test, y_test, ls = load_CIFAR_batch("./cifar-10-python/test_batch", "channels_first")
x = np.concatenate((x_1, x_2, x_3, x_4, x_5, x_test), axis=0)
y = np.concatenate((y_1, y_2, y_3, y_4, y_5, y_test))
x, y = shuffle(x, y, random_state=0) # array([6, 9, 8, ..., 5, 4, 8])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
itemfreq(y_test)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
itemfreq(y_valid)

# 把数据变成float32更精确
val_max = np.max(x_train)
x_train = (x_train/val_max).astype("float32")
x_valid = (x_valid/val_max).astype("float32")
x_test = (x_test/val_max).astype("float32")

y_train_ot = clt.one_hot(y_train)
y_valid_ot = clt.one_hot(y_valid)
y_test_ot = clt.one_hot(y_test)

out_num = len(set(y_train))
input_shape = x_train.shape[1:]
keep_prob = 1.0
batch_size = 128
x_train.shape[0] // batch_size # batch_sample

model = Sequential()

model.add(Conv2D(input_shape=input_shape, # 当使用该层作为第一层时，应提供input_shape参数
                 filters=32, 
                 kernel_size=(5,5), 
                 strides=(1,1),
                 padding="same", # 补0策略，为“valid”, “same” 
                 activation="relu",
                 use_bias=True,
                 bias_initializer=initializers.zeros(),
                 kernel_regularizer=l2(0.0001),
                 kernel_initializer=initializers.random_normal(0.0, 0.0001)))
#model.add(BatchNormalization(axis=1, center=True, beta_initializer='zeros', 
#                             scale=True, gamma_initializer='ones', 
#                             epsilon=10**-8, momentum=0.9))
#model.add(Activation("relu"))
model.add(Dropout(rate=(1-keep_prob)))
model.add(MaxPooling2D(pool_size=(3,3), strides=2)) # pool_size=(3,3)

model.add(Conv2D(filters=32, 
                 kernel_size=(5,5), 
                 strides=(1,1),
                 padding="same", # 补0策略，为“valid”, “same” 
                 activation="relu",
                 use_bias=True,
                 bias_initializer=initializers.zeros(),
                 kernel_regularizer=l2(0.0001),
                 kernel_initializer=initializers.random_normal(0.0, 0.01)))
model.add(Dropout(rate=(1-keep_prob)))
model.add(AveragePooling2D(pool_size=(3,3), strides=2)) # pool_size=(3,3)

model.add(Conv2D(filters=64, 
                 kernel_size=(5,5), 
                 strides=(1,1),
                 padding="same", # 补0策略，为“valid”, “same” 
                 activation="relu",
                 use_bias=True,
                 bias_initializer=initializers.zeros(),
                 kernel_regularizer=l2(0.0001),
                 kernel_initializer=initializers.random_normal(0.0, 0.01)))
model.add(Dropout(rate=(1-keep_prob)))
model.add(AveragePooling2D(pool_size=(3,3), strides=2)) # pool_size=(3,3)

model.add(Flatten())

model.add(Dense(units=64, 
                activation="relu",
                use_bias=True,
                kernel_initializer=initializers.random_normal(0.0, 0.1),
                bias_initializer=initializers.zeros(),
                kernel_regularizer=l2(0.0001)))

model.add(Dense(units=out_num, 
                activation="softmax",
                use_bias=True,
                kernel_initializer=initializers.random_normal(0.0, 0.1),
                bias_initializer=initializers.zeros(),
                kernel_regularizer=l2(0.0001)))

model.summary()
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

model.compile(loss="categorical_crossentropy", 
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=0.04),
              metrics=["accuracy"])
#alpha = learning_init * decay_rate**(global_step/decay_steps)

## fit
model.fit(x_train, y_train_ot, batch_size=batch_size, epochs=10, validation_data=(x_valid,y_valid_ot), verbose=1) #validation_data=(X,Y); validation_split=0.2
## pred
loss_train, accu_train = model.evaluate(x_train, y_train_ot, verbose=1)
loss_valid, accu_valid = model.evaluate(x_valid, y_valid_ot, verbose=1)
loss_test, accu_test = model.evaluate(x_test, y_test_ot, verbose=1)

output = model.predict(x_test, batch_size=batch_size, verbose=1)
y_pred = np.argmax(output, axis=1)
sum(y_pred == y_test) / len(y_test) # 0.9249
pd.crosstab(y_test, y_pred, margins=True)
