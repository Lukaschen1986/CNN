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
img_batch_update(path="D:\\my_project\\Python_Project\\test\\NN\\bedroom\\model", height=224, width=224, flipCode=1)

def img_batch_toarray(path, channel, height, width, data_format, y_label):
    assert isinstance(y_label, int), "y_label must astype int"
    os.chdir(path)
    X = np.zeros((channel, height, width), dtype="float32")
    X = np.expand_dims(X, axis=0)
    for filename in os.listdir():
        pic = image.load_img(filename) # keras load style
        pic_array = image.img_to_array(pic, data_format).astype("float32") # data_format="channels_first"
        pic_array = np.expand_dims(pic_array, axis=0)
        X = np.concatenate((X, pic_array), axis=0)
    X = X[1:] # 删除第一个0数据
    y = np.tile(y_label, len(X)).astype("int32")
    data = {"target":X, "label":y}
    return data
data = img_batch_toarray(path="D:\\my_project\\Python_Project\\test\\NN\\bedroom\\model", 
                         channel=3, height=224, width=224, 
                         data_format="channels_first", y_label=0)

X = data["target"]
N, C, H, W = X.shape

padding_func = lambda H,HH,S: int(np.floor((S*(H-1)+HH-H) / 2))

# keras
val_max = np.max(X)
X = (X/val_max).astype("float32")

def filter_init(kernel_initializer, F, C, HH, WW):
    w = np.zeros((F, C, HH, WW), dtype="float32")
    for i in range(F):
        if kernel_initializer == "random_uniform":
            w[i] = np.random.rand(C, HH, WW)
        elif kernel_initializer == "random_normal":
            w[i] = np.random.randn(C, HH, WW)
        else:
            raise ValueError("kernel_initializer must in ('random_uniform','random_normal')")
    b = np.zeros((F))
    return w, b
w, b = filter_init(kernel_initializer="random_normal", F=10, C=C, HH=3, WW=3)

F, _, HH, WW = w.shape
S = 1
P = padding_func(H, HH, S)

X_pad = np.zeros((N, C, H+P*2, W+P*2), dtype="float32")
X_pad[:, :, P:H+P, P:W+P] = X

out = np.zeros((N, F, H, W), dtype="float32")

def filter_flip(w):
    F, C, _, _ = w.shape
    w_flip = copy.deepcopy(w)
    for f in range(F):
        for c in range(C):
            w_flip[f,c,:,:] = np.flip(np.flip(w[f,c,:,:], axis=1), axis=0)
    return w_flip
w_flip = filter_flip(w)

for f in range(F):
    for i in range(H):
        for j in range(W):
            # f = 0; i = 223; j = 0
            out[:,f,i,j] = np.sum(X_pad[:, :, i*S:i*S+HH, j*S:j*S+WW] * w_flip[f, :, :, :], axis=(1,2,3))
    out[:,f,:,:] += b[f]
