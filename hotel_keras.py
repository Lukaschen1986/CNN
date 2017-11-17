# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
os.getcwd()
os.chdir("D:\\my_project\\Python_Project\\test\\NN\\bedroom\\标间")
import numpy as np
import pandas as pd
from scipy.stats import itemfreq
import pickle

import cv2 as cv
from PIL import Image

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
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from sklearn import datasets
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# resize, flip, toarray
def img_resize(filename, height, width):
    pic = image.load_img(filename)
    pic_array = image.img_to_array(pic, "channels_last").astype("uint8")
    pic_resize = cv.resize(pic_array, dsize=(height,width))
    pic_update = Image.fromarray(pic_resize)
    return pic_update.save(".\\" + filename)

def img_flip(filename, flipCode):
    pic = image.load_img(filename)
    pic_array = image.img_to_array(pic, "channels_last").astype("uint8")
    pic_flip = cv.flip(pic_array, flipCode) # flipCode>0 水平； flipCode=0 垂直； flipCode<0 水平+垂直
    pic_update = Image.fromarray(pic_flip)
    return pic_update.save(".\\flip_" + filename)

def img_batch_update(path, height, width, flipCode):
    os.chdir(path)
    for filename in os.listdir():
        img_resize(filename, height, width)
        img_flip(filename, flipCode)
img_batch_update(path="D:\\my_project\\Python_Project\\test\\NN\\bedroom\\大床", height=120, width=120, flipCode=1)

def img_batch_toarray(path, channel, height, width, data_format, y_label):
    assert isinstance(y_label, int), "y_label must astype int"
    os.chdir(path)
    X = np.zeros(channel*height*width, dtype="float32").reshape(channel,height,width)
    X = np.expand_dims(X, axis=0)
    for filename in os.listdir():
        pic = image.load_img(filename) # keras load style
        pic_array = image.img_to_array(pic, data_format).astype("float32")
        pic_array = np.expand_dims(pic_array, axis=0)
        X = np.concatenate((X, pic_array), axis=0)
    X = X[1:] # 删除第一个0数据
    y = np.tile(y_label, len(X)).astype("int32")
    data = {"target":X, "label":y}
    return data
data_1 = img_batch_toarray(path="D:\\my_project\\Python_Project\\test\\NN\\bedroom\\标间", 
                           channel=3, height=120, width=120, 
                           data_format="channels_first", y_label=0)
data_2 = img_batch_toarray(path="D:\\my_project\\Python_Project\\test\\NN\\bedroom\\大床", 
                           channel=3, height=120, width=120, 
                           data_format="channels_first", y_label=1)    
    
# permutation
X_1 = data_1["target"]
y_1 = data_1["label"]
X_2 = data_2["target"]
y_2 = data_2["label"]
X_1 = np.random.permutation(X_1) # 440 (330, 55, 55)
X_2 = np.random.permutation(X_2) # 440 (330, 55, 55)
#X = np.concatenate((X_1, X_2), axis=0)
#y = np.concatenate((y_1, y_2))
#np.random.seed(1)
#np.random.permutation(X)
#np.random.permutation(y)

X_1_train = X_1[0:330]
X_1_valid = X_1[330:385]
X_1_test = X_1[385:440]
y_1_train = y_1[0:330]
y_1_valid = y_1[330:385]
y_1_test = y_1[385:440]

X_2_train = X_2[0:330]
X_2_valid = X_2[330:385]
X_2_test = X_2[385:440]
y_2_train = y_2[0:330]
y_2_valid = y_2[330:385]
y_2_test = y_2[385:440]

X_train = np.concatenate((X_1_train, X_2_train), axis=0)
X_valid = np.concatenate((X_1_valid, X_2_valid), axis=0)
X_test = np.concatenate((X_1_test, X_2_test), axis=0)
y_train = np.concatenate((y_1_train, y_2_train))
y_valid = np.concatenate((y_1_valid, y_2_valid))
y_test = np.concatenate((y_1_test, y_2_test))

# keras
val_max = np.max(X_train)
X_train = (X_train/val_max).astype("float32")
X_valid = (X_valid/val_max).astype("float32")
X_test = (X_test/val_max).astype("float32")

batch_size = 128
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
                 strides=1,
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
#model.add(Dropout(rate=(1-keep_prob)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
## 2
model.add(Conv2D(nb_filter=32, 
                 kernel_size=(3,3), 
                 strides=1,
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
#model.add(Dropout(rate=(1-keep_prob)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
## 3
model.add(Conv2D(nb_filter=64, 
                 kernel_size=(3,3), 
                 strides=1,
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
#model.add(Dropout(rate=(1-keep_prob)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
## 4
model.add(Conv2D(nb_filter=64, 
                 kernel_size=(3,3), 
                 strides=1,
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
#model.add(Dropout(rate=(1-keep_prob)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
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
#model.add(Dropout(rate=(1-keep_prob)))
## 7
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
#model.add(Dropout(rate=(1-keep_prob)))
## 8
model.add(Dense(units=outNum, activation='softmax'))
## summary
model.summary()
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
## compile
model.compile(loss="categorical_crossentropy", 
              optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=0.0),
              metrics=["accuracy"])
## fit
model.fit(X_train, y_train_onehot, batch_size=batch_size, epochs=10, validation_data=(X_valid,y_valid_onehot), verbose=1)
## pred
loss_train, accu_train = model.evaluate(X_train, y_train_onehot, verbose=1) # 0.6553, 0.7136
loss_valid, accu_valid = model.evaluate(X_valid, y_valid_onehot, verbose=1) # 0.7792, 0.5999
loss_test, accu_test = model.evaluate(X_test, y_test_onehot, verbose=1) # 0.7772, 0.5909

output = model.predict(X_test, batch_size=batch_size, verbose=1)
y_pred = np.argmax(output, axis=1)
sum(y_pred == y_test) / len(y_test) # 0.9249
pd.crosstab(y_test, y_pred, margins=True)

model.save(filepath="cifar.h5", overwrite=True, include_optimizer=True)
model = load_model(filepath="cifar.h5", custom_objects=None, compile=True)
