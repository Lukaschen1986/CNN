# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
os.getcwd()
import numpy as np
#import pandas as pd
from scipy.stats import itemfreq
import pickle
#import copy
#import cv2 as cv
#from PIL import Image
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
#from keras.layers.normalization import BatchNormalization
#from keras.layers.core import Activation
#from keras import backend as K
#K.image_data_format()
#K.set_image_data_format('channels_first')
#from keras.utils.np_utils import to_categorical
#from keras.regularizers import l1, l2
#from keras.optimizers import SGD, Adam
#from keras.utils import plot_model
#from keras.models import load_model
#from keras.applications.vgg16 import VGG16
#from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
#from sklearn import datasets
#from sklearn import linear_model
from sklearn.cross_validation import train_test_split
#from sklearn import svm
#from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
#import matplotlib.pyplot as plt
import cnn_layers as cl

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

x = X_train

loss_res = []
filters1 = 32
w1, b1 = cl.filter_init(x, F=filters1, HH=5, WW=5)
stride = 1; padding = cl.get_padding(x, w1, stride)
conv_param1 = {"S":stride, "P":padding}
pool_param = {"S":2, "HP":2, "WP":2}
mode = "train"
momentum = 0.9
gamma1 = np.ones((1, filters1), dtype="float32")
beta1 = np.zeros((1, filters1), dtype="float32")
running_mean1 = running_var1 = np.zeros((1, filters1), dtype="float32")
bn_param1 = {"mode":mode, "momentum":momentum, "running_mean":running_mean1, "running_var":running_var1}


# Conv_1
out1, cache1 = cl.conv_bn_relu_pool_forward(x, w1, b1, gamma1, beta1, conv_param1, bn_param1, pool_param)

w2, b2 = cl.affine_init(out1, units=128)
keep_prob = 0.8
dropout_param = {"mode":mode, "keep_prob": keep_prob}
out2, cache2 = cl.affine_relu_forward(out1, w2, b2, dropout_param)

loss, dout2 = cl.softmax_loss(out2, y)
loss_res.append(loss)

dout1, dw2, db2 = cl.affine_relu_backward(dout2, cache2)
dx, dw1, db1, dgamma1, dbeta1 = cl.conv_bn_relu_pool_backward(dout1, cache1)

w2, w2_config = cl.Adam(w2, dw2)
b2, b2_config = cl.Adam(b2, db2)
w1, w1_config = cl.Adam(w1, dw1)
b1, b1_config = cl.Adam(b1, db1)
gamma1, gamma1_config = cl.Adam(gamma1, dgamma1)
beta1, beta1_config = cl.Adam(beta1, dbeta1)

w2, w2_config = cl.Adam(w2, dw2, w2_config)
b2, b2_config = cl.Adam(b2, db2, b2_config)
w1, w1_config = cl.Adam(w1, dw1, w1_config)
b1, b1_config = cl.Adam(b1, db1, b1_config)
gamma1, gamma1_config = cl.Adam(gamma1, dgamma1, gamma1_config)
beta1, beta1_config = cl.Adam(beta1, dbeta1, beta1_config)

