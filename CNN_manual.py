# -*- coding: utf-8 -*-
# http://blog.csdn.net/l_b_yuan/article/details/64927643
# http://m.blog.csdn.net/dajiabudongdao/article/details/77263608
# https://wenku.baidu.com/view/deb7be2b6137ee06eef918a0.html
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
#img_batch_update(path="D:\\my_project\\Python_Project\\test\\NN\\pic", height=224, width=224, flipCode=1)

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
data = img_batch_toarray(path="D:\\my_project\\Python_Project\\test\\NN\\pic", 
                         channel=3, height=224, width=224, 
                         data_format="channels_first", y_label=0)

X = data["target"]
y = data["label"]
val_max = np.max(X)
X = (X/val_max).astype("float32")
#N, C, H, W = X.shape

# conv_1
#def filter_init(X, F, HH, WW, kernel_initializer):
#    _, C, _, _ = X.shape
#    w = np.zeros((F, C, HH, WW), dtype="float32")
#    for f in range(F):
#        if kernel_initializer == "random_uniform":
#            w[f,:,:,:] = np.random.rand(C, HH, WW) * 0.01
#        elif kernel_initializer == "random_normal":
#            w[f,:,:,:] = np.random.randn(C, HH, WW) * 0.01
#        else:
#            raise ValueError("kernel_initializer must in ('random_uniform','random_normal')")
#    b = np.zeros((F))
#    return w, b

def filter_init(X, F, HH, WW):
    _, C, _, _ = X.shape
    w = np.zeros((F, C, HH, WW), dtype="float32")
    for f in range(F):
        w[f,:,:,:] = np.random.randn(C, HH, WW) * 0.01
    b = np.zeros((F))
    return w, b
w1, b1 = filter_init(X=X, F=4, HH=3, WW=3)

def filter_flip(w):
    F, C, _, _ = w.shape
    w_flip = copy.deepcopy(w)
    for f in range(F):
        for c in range(C):
            w_flip[f,c,:,:] = np.flip(np.flip(w[f,c,:,:], axis=1), axis=0)
    return w_flip
w_flip_1 = filter_flip(w1)

stride = 1
def get_padding(X, w_flip, S):
    _, _, H, _ = X.shape
    _, _, HH, _ = w_flip.shape
    P = int(np.floor((S*(H-1)+HH-H) / 2))
    return P
padding = get_padding(X=X, w_flip=w_flip_1, S=stride)
conv_param = {"S":stride, "P":padding}

def conv_forward(X, w_flip, b, conv_param):
    out = None
    N, C, H, W = X.shape
    F, _, HH, WW = w_flip.shape
    S = conv_param["S"]
    P = conv_param["P"]
    X_pad = np.zeros((N, C, H+P*2, W+P*2), dtype="float32")
    X_pad[:, :, P:H+P, P:W+P] = X
    out = np.zeros((N, F, H, W), dtype="float32")
    for f in range(F):
        for i in range(H):
            for j in range(W):
                # f = 0; i = 0; j = 0
                out[:,f,i,j] = np.sum(X_pad[:, :, i*S:i*S+HH, j*S:j*S+WW] * w_flip[f, :, :, :], axis=(1,2,3))
        out[:,f,:,:] += b[f]
    cache = (X, w_flip, b, conv_param)
    return out, cache
conv_out_1, conv_cache_1 = conv_forward(X, w_flip_1, b1, conv_param)

# active_1
def relu_forward(conv_out):
    out = np.maximum(0, conv_out)
    return out
active_out_1 = relu_forward(conv_out_1)

# max_1
pool_param = {"S":2, "HP":2, "WP":2}
def maxPooling_forward(active_out, pool_param):
    out = None
    N, C, H, W = active_out.shape
    S = pool_param["S"]; HP = pool_param["HP"]; WP = pool_param["WP"]
    HP2 = int(np.floor((H-HP)/S+1))
    WP2 = int(np.floor((W-WP)/S+1))
    out = np.zeros((N, C, HP2, WP2))
    for i in range(HP2):
        for j in range(WP2):
            # i = 0; j = 0
            out[:,:,i,j] = np.max(active_out[:, :, i*S:i*S+HP, j*S:j*S+WP].reshape(N, C, -1), axis=2)
#            out[:,:,i,j] = np.max(active_out[:, :, i*S:i*S+HP, j*S:j*S+WP].reshape(N, C, -1), axis=(2,3))
    cache = (active_out, pool_param)
    return out, cache
max_out_1, max_cache_1 = maxPooling_forward(active_out_1, pool_param)

# conv_2
w2, b2 = filter_init(X=max_out_1, F=6, HH=3, WW=3)
w_flip_2 = filter_flip(w2)            

stride = 1
padding = get_padding(max_out_1, w_flip_2, stride)
conv_param = {"S":stride, "P":padding}
conv_out_2, conv_cache_2 = conv_forward(max_out_1, w_flip_2, b2, conv_param)

# active_2
active_out_2 = relu_forward(conv_out_2)

# max_2
pool_param = {"S":2, "HP":2, "WP":2}
max_out_2, max_cache_2 = maxPooling_forward(active_out_2, pool_param)

# flatten
def flatten(max_out):
    N, C, H, W = max_out.shape
    max_out_re = max_out.reshape(N, C*H*W)
    return max_out_re
max_out_flatten = flatten(max_out_2)

# dense
def dense_init(max_out_flatten, units):
    N, P = max_out_flatten.shape
    w = np.random.randn(P, units) * np.sqrt(2.0/P)
    b = np.zeros((1, units))
    return w, b
w3, b3 = dense_init(max_out_flatten, 512)

z3 = max_out_flatten.dot(w3) + b3
a3 = relu_forward(z3)

w4, b4 = dense_init(a3, 128)
z4 = a3.dot(w4) + b4

def softmax(z):
    res = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return res
output = softmax(z4)
N, P = output.shape

lam = 0.0001
Loss = -np.sum(np.log(output[range(N), y]))/N
Loss += lam/(2*N)*np.sum(w1**2) + lam/(2*N)*np.sum(w2**2) # 4.8524

# 反向逐层求导
def relu_dv(z):
    res = np.where(z < 0, 0, 1)
    return res

delta4 = output
delta4[range(N), y] -= 1
dw4 = a3.T.dot(delta4) + lam/N*w4
db4 = np.sum(delta4, axis=0, keepdims=True)

delta3 = delta4.dot(w4.T) * relu_dv(z3)
dw3 = max_out_flatten.T.dot(delta3) + lam/N*w3
db3 = np.sum(delta3, axis=0, keepdims=True)

delta2 = delta3.dot(w3.T) * active_dv(z2)
dw2 = a1.T.dot(delta2) + lam/n_batch*w2
db2 = np.sum(delta2, axis=0, keepdims=True)

X_shape = (2, 3, 112, 112)
w_shape = (2, 3, 3, 3)
x = np.ones(X_shape)
w = np.ones(w_shape)
b = np.array([1,2])
dout = np.ones((X_shape[0], w_shape[0], 112, 112))

max_out_2, max_cache_2
max_out_2.shape # (2, 6, 56, 56)
w2.shape # (6, 4, 3, 3)

def maxPooling_backward(delta, cache):
    
def maxPooling_backward(max_cache):
    active_out, pool_param = max_cache_2
    N,C,H,W = active_out.shape # (2, 6, 112, 112)
    HP, WP, S = pool_param["HP"], pool_param["WP"], pool_param["S"]
    H1 = int(np.floor((H-HP)/S + 1))
    W1 = int(np.floor((W-WP)/S + 1))
    
    active_out_dv = np.zeros_like(active_out)
    for i in range(H1):
        for j in range(W1):
            # i = 0; j = 0
            window = active_out[:, :, i*S:i*S+HP, j*S:j*S+WP]
            
            x_pooling = active_out[:, :, i*S:i*S+HP, j*S:j*S+WP]
            maxi = np.max(x_pooling)
            x_mask = x_pooling == maxi       
            active_out_dv[:, :, i*S:i*S+HP, j*S:j*S+WP] += 
            
            
            
            dx[nprime, cprime, k * S:k * S + Hp, l * S:l *S + Wp] += dout[nprime, cprime, k, l] * x_mask
            
            np.argmax(x_pooling.reshape(N, C, -1), axis=2)
            np.argmax(x_pooling, axis=(1,2))
            
            
            
            max_idx = np.argmax(active_out[:, :, i*S:i*S+HP, j*S:j*S+WP].reshape(N, C, -1), axis=2)
            max_cols = np.remainder(max_idx, WP) + j # remainder函数逐个返回两个数组中元素相除后的余数
            max_rows = max_idx / WP + i
            for n in range(N):
                for c in range(C):
                    # n = 0; c = 0
                    active_out_dv[n, c, max_rows[n,c], max_cols[n,c]]
            
            
            
    out[:,:,i,j] = np.max(active_out[:, :, i*S:i*S+HP, j*S:j*S+WP].reshape(N, C, -1), axis=2)
    for i,j in enumerate(range(0, H, S)):
        print(i,j)
    
    N,_,H,W = max_out_2.shape
    _,C,_,_ = w2.shape
    
    x = np.ones(max_out_2.shape)
    w = np.ones(w2.shape)
    delta = np.ones
















            
            conv_out[:, :, i*S:i*S+HP, j*S:j*S+WP]
            np.max(conv_out[:, :, i*S:i*S+HP, j*S:j*S+WP], axis=(1))

conv_out[:, :, i*S:i*S+HP, j*S:j*S+WP].reshape(N, C, -1)
np.max(conv_out[:, :, i*S:i*S+HP, j*S:j*S+WP].reshape(N, C, -1), axis=2)
out[:,:,i,j]

i=1; j=1
conv_out[:, :, 0:2, 0:2]
conv_out[:, :, i*S:i*S+HP, j*S:j*S+HP]


X_shape = (2, 3, 112, 112)
w_shape = (2, 3, 3, 3)
x = np.ones(X_shape)
w = np.ones(w_shape)
b = np.array([1,2])
dout = np.ones((X_shape[0], w_shape[0], 112, 112))


def conv_backward(dout, cache):
  dx, dw, db = None, None, None

  N, F, H1, W1 = dout.shape
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  HH = w.shape[2]
  WW = w.shape[3]
  S = conv_param["stride"]
  P = conv_param["padding"]
  
  dx, dw, db = np.zeros_like(X), np.zeros_like(w), np.zeros_like(b)
  x_pad = np.pad(x, [(0,0), (0,0), (P,P), (P,P)], 'constant')
  dx_pad = np.pad(dx, [(0,0), (0,0), (P,P), (P,P)], 'constant')
  db = np.sum(dout, axis=(0,2,3))

  for n in range(N):
    for i in range(H1):
      for j in range(W1):
        # Window we want to apply the respective f th filter over (C, HH, WW)
        x_window = x_pad[n, :, i * S : i * S + HH, j * S : j * S + WW]

        for f in range(F):
          dw[f] += x_window * dout[n, f, i, j] #F,C,HH,WW
          #C,HH,WW
          dx_pad[n, :, i * S : i * S + HH, j * S : j * S + WW] += w[f] * dout[n, f, i, j]

  dx = dx_pad[:, :, P:P+H, P:P+W]

  return dx, dw, db
