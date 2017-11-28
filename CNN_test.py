# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
os.getcwd()
import numpy as np
import pandas as pd
from scipy.stats import itemfreq
import pickle
import copy

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

data = img_batch_toarray(path="D:\\my_project\\Python_Project\\test\\NN\\pic", 
                         channel=3, height=224, width=224, 
                         data_format="channels_first", y_label=0)
x = data["target"]
y = data["label"]
val_max = np.max(x)
x = (x/val_max).astype("float32")

# conv_1
w1, b1 = filter_init(x, F=4, HH=5, WW=5)
#w1_flip = filter_flip(w1)
stride = 1
padding = get_padding(x, w1, stride)
conv_param1 = {"S":stride, "P":padding}
conv_out1, conv_cache1 = conv_forward(x, w1, b1, conv_param1)
conv_out1.shape

# batchnorm_forward_1
_, C, _, _, = conv_out1.shape
#gamma1 = np.ones((1, C), dtype="float32")
#beta1 = np.zeros((1, C), dtype="float32")
#running_mean1 = running_var1 = np.zeros((1, C), dtype="float32")
bn_param1 = {"mode":"train", "momentum":0.9, "running_mean":running_mean1, "running_var":running_var1}
bn_out1, bn_cache1 = batchnorm_forward(conv_out1, gamma1, beta1, bn_param1)
bn_out1.shape

# active_1
relu_out1, relu_cache1 = relu_forward(bn_out1)

# dropout_1
dropout_param1 = {"keep_prob":0.8, "mode":"train"}
drop_out1, drop_cache1 = dropout_forward(relu_out1, dropout_param1)

# maxPooling_1
pool_param1 = {"S":2, "HP":2, "WP":2}
max_out1, max_cache1 = maxPooling_forward(drop_out1, pool_param1)

# conv_2
w2, b2 = filter_init(max_out1, F=6, HH=3, WW=3)
#w2_flip = filter_flip(w2)
stride = 1
padding = get_padding(max_out1, w2, stride)
conv_param2 = {"S":stride, "P":padding}
conv_out2, conv_cache2 = conv_forward(max_out1, w2, b2, conv_param2)
conv_out2.shape

# batchnorm_forward_2
_, C, _, _, = conv_out2.shape
#gamma2 = np.ones((1, C), dtype="float32")
#beta2 = np.zeros((1, C), dtype="float32")
#running_mean2 = running_var2 = np.zeros((1, C), dtype="float32")
bn_param2 = {"mode":"train", "momentum":0.9, "running_mean":running_mean2, "running_var":running_var2}
bn_out2, bn_cache2 = batchnorm_forward(conv_out2, gamma2, beta2, bn_param2)
bn_out2.shape # (2, 6, 112, 112)

# active_2
relu_out2, relu_cache2 = relu_forward(bn_out2) # relu_cache2: bn_out2
relu_cache2.shape

# dropout_2
dropout_param2 = dropout_param1
drop_out2, drop_cache2 = dropout_forward(relu_out2, dropout_param2)

# maxPooling_2
pool_param2 = {"S":2, "HP":2, "WP":2}
max_out2, max_cache2 = maxPooling_forward(drop_out2, pool_param2)

# affine
flatten_out = flatten(max_out2)
#w3, b3 = affine_init(flatten_out, units=512)
z3, z_cache3 = affine_forward(flatten_out, w3, b3) # z_cache3: a2, w3, b3
a3, a_cache3 = relu_forward(z3) # a_cache3: z3

#w4, b4 = affine_init(a3, units=128)
z4, z_cache4 = affine_forward(a3, w4, b4) # z_cache4: a3, w4, b4
a4, a_cache4 = relu_forward(z4) # a_cache4: z4

#w5, b5 = affine_init(a4, units=10)
z5, z_cache5 = affine_forward(a4, w5, b5) # z_cache5: a4, w5, b5
z5.shape

# softmax
#l2_param = {"w3":affine_cache3[1],
#            "w4":affine_cache4[1],
#            "w5":affine_cache5[1],
#            "lam": 10**-4}
loss, dout5, dw5, db5 = softmax_loss(z5, y, z_cache5)
dout4, dw4, db4 = affine_backward(dout5, z_cache5, a_cache4, z_cache4, relu_backward)
dout3, dw3, db3 = affine_backward(dout4, z_cache4, a_cache3, z_cache3, relu_backward)

dout_max_out2 = maxPooling_backward(max_out2, max_cache2)
#dout_relu_out2 = relu_backward(dout_max_out2)
dout_bn_out2, dgamma2, dbeta2 = batchnorm_backward(dout_max_out2, bn_cache2)
dout_conv_out2, dw2, db2 = conv_backward(dout_bn_out2, conv_cache2, relu_cache2, relu_backward)

dout_max_out1 = maxPooling_backward(dout_conv_out2, max_cache1)
#dout_relu_out1 = relu_backward(dout_max_out1)
dout_bn_out1, dgamma1, dbeta1 = batchnorm_backward(dout_max_out1, bn_cache1)
dout_conv_out1, dw1, db1 = conv_backward(dout_bn_out1, conv_cache1, relu_cache1, relu_backward)

