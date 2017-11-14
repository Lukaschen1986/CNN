# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
os.getcwd()
import numpy as np
import pandas as pd

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras import backend as K
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

# 训练集，测试集收集非常方便
(X_train, y_train), (X_predict, y_predict) = mnist.load_data()
print(X_train.shape) # (60000, 28, 28)

# 输入的图片是28*28像素的灰度图
img_rows, img_cols = X_train.shape[1], X_train.shape[2]
# keras输入数据有两种格式，一种是通道数放在前面，一种是通道数放在后面，其实就是格式差别而已
if K.image_data_format() == 'channels_first':
    X_train_reshape = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_predict_reshape = X_predict.reshape(X_predict.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train_reshape = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_predict_reshape = X_predict.reshape(X_predict.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 把数据变成float32更精确
val_max = np.max(X_train_reshape)
X_train_reshape = (X_train_reshape/val_max).astype("float32")
X_predict_reshape = (X_predict_reshape/val_max).astype("float32")
print(X_train_reshape.shape)
print(X_predict_reshape.shape)

# batch_size 太小会导致训练慢，过拟合等问题，太大会导致欠拟合。所以要适当选择
batch_size = 128
outNum = 10
epochs = 10
keep_prob = 0.8

# 把类别0-9变成2进制，方便训练
#one_hot = lambda y: np.eye(len(set(y)))[y]
y_train = to_categorical(y_train, outNum)
y_predict = to_categorical(y_predict, outNum)

# model
model = Sequential()
## 1
model.add(Conv2D(input_shape=input_shape, # 当使用该层作为第一层时，应提供input_shape参数
                 data_format=K.image_data_format(),
                 nb_filter=64, 
                 kernel_size=(3,3), 
                 strides=(1,1),
                 padding="same", # 补0策略，为“valid”, “same” 
                 activation=None,
                 use_bias=True,
                 kernel_initializer="random_uniform", # random_normal
                 bias_initializer="zeros",
                 kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization(center=True, beta_initializer='zeros', scale=True, gamma_initializer='ones'))
model.add(Activation("relu"))
model.add(Dropout(rate=(1-keep_prob)))
model.add(MaxPooling2D(pool_size=(2,2), border_mode="same"))
## 2
model.add(Conv2D(nb_filter=64, 
                 kernel_size=(3,3), 
                 strides=(1,1),
                 padding="same",
                 activation=None,
                 use_bias=True,
                 kernel_initializer="random_uniform", 
                 bias_initializer="zeros",
                 kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization(center=True, beta_initializer='zeros', scale=True, gamma_initializer='ones'))
model.add(Activation("relu"))
model.add(Dropout(rate=(1-keep_prob)))
model.add(MaxPooling2D(pool_size=(2,2), border_mode="same"))
## 3
model.add(Conv2D(nb_filter=32, 
                 kernel_size=(3,3), 
                 strides=(1,1),
                 padding="same",
                 activation=None,
                 use_bias=True,
                 kernel_initializer="random_uniform", 
                 bias_initializer="zeros",
                 kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization(center=True, beta_initializer='zeros', scale=True, gamma_initializer='ones'))
model.add(Activation("relu"))
model.add(Dropout(rate=(1-keep_prob)))
model.add(MaxPooling2D(pool_size=(2,2), border_mode="same"))
## 4
model.add(Flatten())
## 5
model.add(Dense(units=128, 
                activation=None,
                use_bias=True,
                kernel_initializer="random_uniform", # random_normal
                bias_initializer="zeros",
                kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization(center=True, beta_initializer='zeros', scale=True, gamma_initializer='ones'))
model.add(Activation("relu"))
model.add(Dropout(rate=(1-keep_prob)))
## 6
model.add(Dense(units=128, 
                activation=None,
                use_bias=True,
                kernel_initializer="random_uniform", # random_normal
                bias_initializer="zeros",
                kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization(center=True, beta_initializer='zeros', scale=True, gamma_initializer='ones'))
model.add(Activation("relu"))
model.add(Dropout(rate=(1-keep_prob)))
## 7
model.add(Dense(units=outNum, activation='softmax'))
## summary
model.summary()
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True, rankdir='TB')
## compile
model.compile(loss="categorical_crossentropy", 
              optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=0.0),
              metrics=["accuracy"])
## fit
model.fit(X_train_reshape, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
## pred
y_pred = model.predict_classes(X_predict_reshape)
#sum(y_pred == y_predict) / len(y_predict) # 0.98619

#score = model.evaluate(X_predict, y_predict, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
#y_pred = model.predict_classes(X_predict_reshape)

model.save(filepath="mnist-mpl.h5", overwrite=True, include_optimizer=True)
model = load_model(filepath="mnist-mpl.h5", custom_objects=None, compile=True)
