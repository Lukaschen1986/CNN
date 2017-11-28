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
#img_batch_update(path="D:\\my_project\\Python_Project\\test\\NN\\pic", height=224, width=224, flipCode=1)

def img_batch_toarray(path, channel, height, width, data_format, y_label):
    assert isinstance(y_label, int), "y_label must astype int"
    os.chdir(path)
    x = np.zeros((channel, height, width), dtype="float32")
    x = np.expand_dims(x, axis=0)
    for filename in os.listdir():
        pic = image.load_img(filename) # keras load style
        pic_array = image.img_to_array(pic, data_format).astype("float32") # data_format="channels_first"
        pic_array = np.expand_dims(pic_array, axis=0)
        x = np.concatenate((x, pic_array), axis=0)
    x = x[1:] # 删除第一个0数据
    y = np.tile(y_label, len(x)).astype("int32")
    data = {"target":x, "label":y}
    return data

def filter_init(x, F, HH, WW):
    _, C, _, _ = x.shape
    w = np.zeros((F, C, HH, WW), dtype="float32")
    for f in range(F):
        w[f,:,:,:] = np.random.randn(C, HH, WW) * 0.01
    b = np.zeros((F))
    return w, b

def filter_flip(w):
    F, C, _, _ = w.shape
    w_flip = copy.deepcopy(w)
    for f in range(F):
        for c in range(C):
            w_flip[f,c,:,:] = np.flip(np.flip(w[f,c,:,:], axis=1), axis=0)
    return w_flip

def get_padding(x, w, S):
    _, _, H, _ = x.shape
    _, _, HH, _ = w.shape
    P = int(np.floor((S*(H-1)+HH-H) / 2))
    return P


def conv_forward(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    S, P = conv_param["S"], conv_param["P"]
#    x_pad = np.zeros((N, C, H+P*2, W+P*2), dtype="float32")
#    x_pad[:, :, P:H+P, P:W+P] = x
    H_new = int(np.floor((H + 2*P - HH) / S + 1))
    W_new = int(np.floor((W + 2*P - WW) / S + 1))
    
    x_pad = np.pad(x, pad_width=((0,0), (0,0), (P,P), (P,P)), mode="constant", constant_values=0)
    out = np.zeros((N, F, H_new, W_new), dtype="float32")
    for f in range(F):
        for i in range(H_new):
            for j in range(W_new):
                # f = 0; i = 0; j = 0
                pre_out = x_pad[:, :, i*S:i*S+HH, j*S:j*S+WW] * w[f,:,:,:]
                out[:,f,i,j] = np.sum(pre_out, axis=(1,2,3))
        out[:,f,:,:] += b[f]
    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward(dout, cache):
    # dout = dout_bn_out2; z_cache = conv_cache2; a_cache = relu_cache2
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    S, P = conv_param["S"], conv_param["P"]
    H_out = int(np.floor((H + 2*P - HH) / S + 1))
    W_out = int(np.floor((W + 2*P - WW) / S + 1))
    
    x_pad = np.pad(x, pad_width=((0,0), (0,0), (P,P), (P,P)), mode="constant", constant_values=0)
    dx = np.zeros_like(x, dtype="float32") # 构造一个和卷积前一样大小的零矩阵
    dx_pad = np.zeros_like(x_pad, dtype="float32") # 构造一个和x_pad一样大小的零矩阵
    dw = np.zeros_like(w)
    db = np.sum(dout, axis = (0,2,3)) # db2 = np.sum(delta2, axis=0, keepdims=True)
    for i in range(H_out):
        for j in range(W_out):
            # i = 0; j = 0
            x_pad_mask = x_pad[:, :, i*S:i*S+HH, j*S:j*S+WW] # 提取filter对应的原矩阵数值
            for f in range(F):
                dw[f,:,:,:] += np.sum(x_pad_mask * dout[:,f,i,j][:,None,None,None], axis=0) 
                # dw=矩阵原值乘以全局梯度值,参考 dw2 = a1.T.dot(delta2)
            for n in range(N):
                dx_pad[n, :, i*S:i*S+HH, j*S:j*S+WW] += np.sum(dout[n,:,i,j][:,None,None,None] * w, axis=0)
                # 参考 delta1 = delta2.dot(w2.T) * active_dv(z1)
    dx = dx_pad[:, :, P:P+H, P:P+W] # 将dx_pad剔除padding的值传给dx得到最终dx
    return dx, dw, db
    
    
def batchnorm_forward(x, gamma, beta, bn_param):
    # x = conv_out1; momentum = 0.9; gamma = gamma1; beta = beta1
    mode = bn_param["mode"]
    momentum = bn_param["momentum"]
    running_mean = bn_param["running_mean"]
    running_var = bn_param["running_var"]
    N, D = x.shape
    # 分train和test执行bn
    if mode == "train":
        sample_mean = np.mean(x, axis=0, keepdims=True)
        sample_var = np.var(x, axis=0, keepdims=True)
        x_scale = (x - sample_mean) / (np.sqrt(sample_var + 10**-8))
        out = gamma * x_scale + beta
        cache = (gamma, x, sample_mean, sample_var, x_scale)
        running_mean = momentum*running_mean + (1-momentum)*sample_mean
        running_var = momentum*running_var + (1-momentum)*sample_var
    elif mode == "test":
        x_scale = (x - running_mean) / (np.sqrt(running_var + 10**-8))
        out = gamma * x_scale + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    return out, cache

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  N, C, H, W = x.shape
  temp_output, cache = batchnorm_forward(x.reshape(N*H*W, C), gamma, beta, bn_param) # 以通道数为列变量进行标准化
  out = temp_output.reshape(N, C, H, W)
  return out, cache

def batchnorm_backward_alt(dout, cache):
    # dout = dout_max_out2; cache = bn_cache2
    gamma, x, sample_mean, sample_var, x_scale = cache
    N = x.shape[0]
#    N, C, H, W = dout.shape
#    dout_reshape = dout.reshape(N*H*W, C)
#    x_reshape = x.reshape(N*H*W, C)
    # 根据公式推导
    dx_normalized = gamma * dout
    dsample_var = np.sum(dx_normalized * (x-sample_mean) * (-1.0/2) * (sample_var+10**-8)**(-3.0/2), axis=0, keepdims=True)
    dsample_mean = np.sum(dx_normalized * (-1.0) / (np.sqrt(sample_var+10**-8)), axis=0, keepdims=True) + (1.0/N) * dsample_var * np.sum(-2*(x-sample_mean), axis = 0, keepdims=True)
    dx = dx_normalized * 1.0/(np.sqrt(sample_var+10**-8)) + dsample_var*2.0*(x-sample_mean)/N + 1.0/N*dsample_mean
    dgamma = np.sum(dout*x_scale, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)
    return dx, dgamma, dbeta

def spatial_batchnorm_backward(dout, cache):
  N, C, H, W = dout.shape
#  dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
  dout_flat = dout.reshape(N*H*W, C)
  dx_flat, dgamma, dbeta = batchnorm_backward_alt(dout_flat, cache)
  dx = dx_flat.reshape(N, C, H, W)
  return dx, dgamma, dbeta
#def spatial_batchnorm_backward(dout, cache):
#  N,C,H,W = dout.shape
#  dx_temp, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(0,3,2,1).reshape((N*H*W,C)),cache)
#  dx = dx_temp.reshape(N,W,H,C).transpose(0,3,2,1)
#  return dx, dgamma, dbeta

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout):
    dx = np.where(dout < 0, 0, 1)
    return dx   

def dropout_forward(x, dropout_param):
    # x = relu_out1
    keep_prob, mode = dropout_param["keep_prob"], dropout_param["mode"]
    if mode == "train":
        mask = np.random.rand(*x.shape) < keep_prob / keep_prob
        out = x * mask
    elif mode == "test":
        out = x
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    cache = (dropout_param, mask)
    return out, cache

def maxPooling_forward(x, pool_param):
    # x = relu_out1
    N, C, H, W = x.shape
    S, HP, WP = pool_param["S"], pool_param["HP"], pool_param["WP"]
    H_out = int(np.floor((H-HP)/S + 1))
    W_out = int(np.floor((W-WP)/S + 1))
    out = np.zeros((N, C, H_out, W_out), dtype="float32")
    
    for i in range(H_out):
        for j in range(W_out):
            pre_out = x[:, :, i*S:i*S+HP, j*S:j*S+WP] # 标记filter每一步返回的值
            out[:,:,i,j] = np.max(pre_out, axis=(2,3)) # 取每一步filter的最大值
    
    cache = (x, pool_param)
    return out, cache

def maxPooling_backward(dout, cache):
    # dout = max_out2; cache = max_cache2
    x, pool_param = cache
    N, C, H, W = x.shape
    S, HP, WP = pool_param["S"], pool_param["HP"], pool_param["WP"]
    H_out = int(np.floor((H-HP)/S + 1))
    W_out = int(np.floor((W-WP)/S + 1))
    dx = np.zeros_like(x, dtype="float32") # 构造一个和池化前一样大小的零矩阵
    
    for i in range(H_out):
        for j in range(W_out):
            # i = 0; j = 0
            pre_dx = x[:, :, i*S:i*S+HP, j*S:j*S+WP] # 标记filter每一步返回的值
            pre_dx_mask = np.max(pre_dx, axis=(2,3)) # 取每一步filter的最大值
            mask_loc = pre_dx == pre_dx_mask[:,:,None,None] # 关键步骤：标记最大值位于filter的位置, None的作用是填补1，确保前后维度一致
            dx[:, :, i*S:i*S+HP, j*S:j*S+WP] += dout[:,:,i,j][:,:,None,None] * mask_loc # 用池化后的输出值乘以原位置坐标,把最大值塞进原始位置
    return dx


def flatten(x):
    N, C, H, W = x.shape
    out = x.reshape(N, C*H*W)
    return out

def affine_init(x, units):
    N, P = x.shape
    w = np.random.randn(P, units).astype("float32") * np.sqrt(2.0/P)
    b = np.zeros((1, units), dtype="float32")
    return w, b

def affine_forward(x, w, b):
    out = x.dot(w) + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, z_cache_latter, a_cache, z_cache_pre, active_backward):
    # dout = dout5; z_cache_latter = z_cache5; a_cache = a_cache4; z_cache_pre = z_cache4
    _, w, _ = z_cache_latter
    a, _, _ = z_cache_pre
    z = a_cache
    dx = dout.dot(w.T) * active_backward(z) # delta4 = delta5.dot(w5.T) * active_dv(z4)
    dw = a.T.dot(dx) # dw4 = a3.T.dot(delta4)
    db = np.sum(dx, axis=0, keepdims=True) # db4 = np.sum(delta4, axis=0, keepdims=True)
    return dx, dw, db

#dout.dot(w.T).shape
#active_backward(z).shape

def softmax_loss(x, y, z_cache):
    a = z_cache[0]
    # loss
    probs = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    N, _ = x.shape
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    # dout
    dout = probs # delta5 = output
    dout[np.arange(N), y] -= 1 # delta5[range(n_batch), y_batch] -= 1
    dw = a.T.dot(dout) # dw5 = a4.T.dot(delta5) + lam/n_batch*w5
    db = np.sum(dout, axis=0, keepdims=True) # db5 = np.sum(delta5, axis=0, keepdims=True)
    return loss, dout, dw, db

