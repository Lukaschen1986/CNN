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
    
X_1, y_1, label_1 = load_CIFAR_batch("D:\\file\\Py_project\\CNN\\cifar-10-python\\data_batch_1", "channels_first")
#X_2, y_2, label_2 = load_CIFAR_batch("D:\\file\\Py_project\\CNN\\cifar-10-python\\data_batch_2", "channels_first")
#X_3, y_3, label_3 = load_CIFAR_batch("D:\\file\\Py_project\\CNN\\cifar-10-python\\data_batch_3", "channels_first")
#X_4, y_4, label_4 = load_CIFAR_batch("D:\\file\\Py_project\\CNN\\cifar-10-python\\data_batch_4", "channels_first")
#X_5, y_5, label_5 = load_CIFAR_batch("D:\\file\\Py_project\\CNN\\cifar-10-python\\data_batch_5", "channels_first")
#X_test, y_test, ls = load_CIFAR_batch("D:\\file\\Py_project\\CNN\\cifar-10-python\\test_batch", "channels_first")
#
#X = np.concatenate((X_1, X_2, X_3, X_4, X_5), axis=0)
#y = np.concatenate((y_1, y_2, y_3, y_4, y_5))
#
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
#itemfreq(y_valid)
#
## 把数据变成float32更精确
#val_max = np.max(X_train)
#X_train = (X_train/val_max).astype("float32")
#X_valid = (X_valid/val_max).astype("float32")
#X_test = (X_test/val_max).astype("float32")    
#
#x = X_train

x = X_1[0:1000]
y = y_1[0:1000]

loss_res = []
mode = "train"
momentum = 0.9
pool_param = {"S":2, "HP":2, "WP":2}
keep_prob = 1.0
dropout_param = {"mode":mode, "keep_prob": keep_prob}

# epoch1
# 1
filters1 = 32
w1, b1 = cl.filter_init(x, F=filters1, HH=5, WW=5)
stride = 1; padding = cl.get_padding(x, w1, stride)
conv_param1 = {"S":stride, "P":padding}
gamma1 = np.ones((1, filters1), dtype="float32")
beta1 = np.zeros((1, filters1), dtype="float32")
running_mean1 = running_var1 = np.zeros((1, filters1), dtype="float32")
bn_param1 = {"mode":mode, "momentum":momentum, "running_mean":running_mean1, "running_var":running_var1}

out1, cache1 = cl.conv_bn_relu_pool_forward(x, w1, b1, gamma1, beta1, conv_param1, bn_param1, pool_param)

# 2
filters2 = 64
w2, b2 = cl.filter_init(out1, F=filters2, HH=3, WW=3)
stride = 1; padding = cl.get_padding(out1, w2, stride)
conv_param2 = {"S":stride, "P":padding}
gamma2 = np.ones((1, filters2), dtype="float32")
beta2 = np.zeros((1, filters2), dtype="float32")
running_mean2 = running_var2 = np.zeros((1, filters2), dtype="float32")
bn_param2 = {"mode":mode, "momentum":momentum, "running_mean":running_mean2, "running_var":running_var2}

out2, cache2 = cl.conv_bn_relu_pool_forward(out1, w2, b2, gamma2, beta2, conv_param2, bn_param2, pool_param)

# 3
filters3 = 128
w3, b3 = cl.filter_init(out2, F=filters3, HH=3, WW=3)
stride = 1; padding = cl.get_padding(out2, w3, stride)
conv_param3 = {"S":stride, "P":padding}
gamma3 = np.ones((1, filters3), dtype="float32")
beta3 = np.zeros((1, filters3), dtype="float32")
running_mean3 = running_var3 = np.zeros((1, filters3), dtype="float32")
bn_param3 = {"mode":mode, "momentum":momentum, "running_mean":running_mean3, "running_var":running_var3}

out3, cache3 = cl.conv_bn_relu_pool_forward(out2, w3, b3, gamma3, beta3, conv_param3, bn_param3, pool_param)

# 4
out3 = cl.flatten(out3)
w4, b4 = cl.affine_init(out3, units=128)
out4, cache4 = cl.affine_relu_forward(out3, w4, b4, dropout_param)

# 5
w5, b5 = cl.affine_init(out4, units=10)
out5, cache5 = cl.affine_relu_forward(out4, w5, b5, dropout_param)

# dout5
loss, dout5 = cl.softmax_loss(out5, y)
loss_res.append(loss)
print(loss_res)

# dout4
dout4, dw5, db5 = cl.affine_relu_backward(dout5, cache5)

# dout3
dout3, dw4, db4 = cl.affine_relu_backward(dout4, cache4)

# dout2
dout2, dw3, db3, dgamma3, dbeta3 = cl.conv_bn_relu_pool_backward(dout3, cache3)

# dout1
dout1, dw2, db2, dgamma2, dbeta2 = cl.conv_bn_relu_pool_backward(dout2, cache2)

# dout0
dout0, dw1, db1, dgamma1, dbeta1 = cl.conv_bn_relu_pool_backward(dout1, cache1)

# opti
w5, w5_config = cl.Adam(w5, dw5); b5, b5_config = cl.Adam(b5, db5)

w4, w4_config = cl.Adam(w4, dw4); b4, b4_config = cl.Adam(b4, db4)

w3, w3_config = cl.Adam(w3, dw3); b3, b3_config = cl.Adam(b3, db3)
gamma3, gamma3_config = cl.Adam(gamma3, dgamma3); beta3, beta3_config = cl.Adam(beta3, dbeta3)

w2, w2_config = cl.Adam(w2, dw2); b2, b2_config = cl.Adam(b2, db2)
gamma2, gamma2_config = cl.Adam(gamma2, dgamma2); beta2, beta2_config = cl.Adam(beta2, dbeta2)

w1, w1_config = cl.Adam(w1, dw1); b1, b1_config = cl.Adam(b1, db1)
gamma1, gamma1_config = cl.Adam(gamma1, dgamma1); beta1, beta1_config = cl.Adam(beta1, dbeta1)


# epoch2
out1, cache1 = cl.conv_bn_relu_pool_forward(x, w1, b1, gamma1, beta1, conv_param1, bn_param1, pool_param)
out2, cache2 = cl.conv_bn_relu_pool_forward(out1, w2, b2, gamma2, beta2, conv_param2, bn_param2, pool_param)
out3, cache3 = cl.conv_bn_relu_pool_forward(out2, w3, b3, gamma3, beta3, conv_param3, bn_param3, pool_param)
out3 = cl.flatten(out3)
out4, cache4 = cl.affine_relu_forward(out3, w4, b4, dropout_param)
out5, cache5 = cl.affine_relu_forward(out4, w5, b5, dropout_param)

loss, dout5 = cl.softmax_loss(out5, y)
loss_res.append(loss)
print(loss_res)

dout4, dw5, db5 = cl.affine_relu_backward(dout5, cache5)
dout3, dw4, db4 = cl.affine_relu_backward(dout4, cache4)
dout2, dw3, db3, dgamma3, dbeta3 = cl.conv_bn_relu_pool_backward(dout3, cache3)
dout1, dw2, db2, dgamma2, dbeta2 = cl.conv_bn_relu_pool_backward(dout2, cache2)
_, dw1, db1, dgamma1, dbeta1 = cl.conv_bn_relu_pool_backward(dout1, cache1)

w5, w5_config = cl.Adam(w5, dw5, w5_config); b5, b5_config = cl.Adam(b5, db5, b5_config)
w4, w4_config = cl.Adam(w4, dw4, w4_config); b4, b4_config = cl.Adam(b4, db4, b4_config)
w3, w3_config = cl.Adam(w3, dw3, w3_config); b3, b3_config = cl.Adam(b3, db3, b3_config)
gamma3, gamma3_config = cl.Adam(gamma3, dgamma3, gamma3_config); beta3, beta3_config = cl.Adam(beta3, dbeta3, beta3_config)
w2, w2_config = cl.Adam(w2, dw2, w2_config); b2, b2_config = cl.Adam(b2, db2, b2_config)
gamma2, gamma2_config = cl.Adam(gamma2, dgamma2, gamma2_config); beta2, beta2_config = cl.Adam(beta2, dbeta2, beta2_config)
w1, w1_config = cl.Adam(w1, dw1, w1_config); b1, b1_config = cl.Adam(b1, db1, b1_config)
gamma1, gamma1_config = cl.Adam(gamma1, dgamma1, gamma1_config); beta1, beta1_config = cl.Adam(beta1, dbeta1, beta1_config)

