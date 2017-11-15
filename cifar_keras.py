# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
os.getcwd()
import numpy as np
import pandas as pd
from scipy.stats import itemfreq
import pickle

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
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.models import load_model

from sklearn import datasets
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

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

    
X_1, y_1, label_1 = load_CIFAR_batch("D:\\my_project\\Python_Project\\test\\NN\\cifar-10-python\\data_batch_1", "channels_first")
X_2, y_2, label_2 = load_CIFAR_batch("D:\\my_project\\Python_Project\\test\\NN\\cifar-10-python\\data_batch_2", "channels_first")
X_3, y_3, label_3 = load_CIFAR_batch("D:\\my_project\\Python_Project\\test\\NN\\cifar-10-python\\data_batch_3", "channels_first")
X_4, y_4, label_4 = load_CIFAR_batch("D:\\my_project\\Python_Project\\test\\NN\\cifar-10-python\\data_batch_4", "channels_first")
X_5, y_5, label_5 = load_CIFAR_batch("D:\\my_project\\Python_Project\\test\\NN\\cifar-10-python\\data_batch_5", "channels_first")
X_test, y_test, ls = load_CIFAR_batch("D:\\my_project\\Python_Project\\test\\NN\\cifar-10-python\\test_batch", "channels_first")

X = np.concatenate((X_1, X_2, X_3, X_4, X_5), axis=0)
y = np.concatenate((y_1, y_2, y_3, y_4, y_5))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.02)
itemfreq(y_valid)

# 把数据变成float32更精确
val_max = np.max(X_train)
X_train = (X_train/val_max).astype("float32")
X_valid = (X_valid/val_max).astype("float32")
X_test = (X_test/val_max).astype("float32")

batch_size = 512
outNum = len(np.unique(y_train)); outNum
epochs = 10
keep_prob = 0.8
input_shape = X_train.shape[1:]

# 把类别0-9变成2进制，方便训练
y_train_onehot = to_categorical(y_train, outNum)
y_valid_onehot = to_categorical(y_valid, outNum)
y_test_onehot = to_categorical(y_test, outNum)

# model
model = Sequential()
## 1
model.add(Conv2D(input_shape=input_shape, # 当使用该层作为第一层时，应提供input_shape参数
                 nb_filter=32, 
                 kernel_size=(3,3), 
                 strides=(1,1),
                 padding="same", # 补0策略，为“valid”, “same” 
                 activation=None,
                 use_bias=True,
                 kernel_initializer="random_uniform", # random_normal
                 bias_initializer="zeros",
                 kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization(center=True, beta_initializer='zeros', 
                             scale=True, gamma_initializer='ones', 
                             epsilon=10**-8, momentum=0.5))
model.add(Activation("relu"))
model.add(Dropout(rate=(1-keep_prob)))
model.add(MaxPooling2D(pool_size=(2,2), border_mode="same"))
## 2
model.add(Conv2D(nb_filter=32, 
                 kernel_size=(3,3), 
                 strides=(1,1),
                 padding="same",
                 activation=None,
                 use_bias=True,
                 kernel_initializer="random_uniform", 
                 bias_initializer="zeros",
                 kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization(center=True, beta_initializer='zeros', 
                             scale=True, gamma_initializer='ones', 
                             epsilon=10**-8, momentum=0.5))
model.add(Activation("relu"))
model.add(Dropout(rate=(1-keep_prob)))
model.add(MaxPooling2D(pool_size=(2,2), border_mode="same"))
## 3
model.add(Conv2D(nb_filter=64, 
                 kernel_size=(3,3), 
                 strides=(1,1),
                 padding="same",
                 activation=None,
                 use_bias=True,
                 kernel_initializer="random_uniform", 
                 bias_initializer="zeros",
                 kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization(center=True, beta_initializer='zeros', 
                             scale=True, gamma_initializer='ones', 
                             epsilon=10**-8, momentum=0.5))
model.add(Activation("relu"))
model.add(Dropout(rate=(1-keep_prob)))
model.add(MaxPooling2D(pool_size=(2,2), border_mode="same"))
## 4
model.add(Conv2D(nb_filter=64, 
                 kernel_size=(3,3), 
                 strides=(1,1),
                 padding="same",
                 activation=None,
                 use_bias=True,
                 kernel_initializer="random_uniform", 
                 bias_initializer="zeros",
                 kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization(center=True, beta_initializer='zeros', 
                             scale=True, gamma_initializer='ones', 
                             epsilon=10**-8, momentum=0.5))
model.add(Activation("relu"))
model.add(Dropout(rate=(1-keep_prob)))
## 5
model.add(Flatten())
## 6
model.add(Dense(units=128, 
                activation=None,
                use_bias=True,
                kernel_initializer="random_uniform", # random_normal
                bias_initializer="zeros",
                kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization(center=True, beta_initializer='zeros', 
                             scale=True, gamma_initializer='ones', 
                             epsilon=10**-8, momentum=0.5))
model.add(Activation("relu"))
model.add(Dropout(rate=(1-keep_prob)))
## 7
model.add(Dense(units=outNum, activation='softmax'))
## summary
model.summary()
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
## compile
model.compile(loss="categorical_crossentropy", 
              optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=0.0),
              metrics=["accuracy"])
## fit
model.fit(X_train, y_train_onehot, batch_size=batch_size, epochs=10, validation_data=(X_valid,y_valid_onehot), verbose=1) #validation_data=(X,Y); validation_split=0.2
## pred
loss_train, accu_train = model.evaluate(X_train, y_train_onehot, verbose=1)
loss_valid, accu_valid = model.evaluate(X_valid, y_valid_onehot, verbose=1)
loss_test, accu_test = model.evaluate(X_test, y_test_onehot, verbose=1)

output = model.predict(X_test, batch_size=batch_size, verbose=1)
y_pred = np.argmax(output, axis=1)
sum(y_pred == y_test) / len(y_test) # 0.9249
pd.crosstab(y_test, y_pred, margins=True)
