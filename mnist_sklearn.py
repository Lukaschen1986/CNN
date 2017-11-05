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
from keras import backend as K
from keras.utils.np_utils import to_categorical

from sklearn import datasets
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# 训练集，测试集收集非常方便
(X_train, y_train), (X_predict, y_predict) = mnist.load_data()
X_train.shape # (60000, 28, 28)
# 输入的图片是28*28像素的灰度图
img_rows, img_cols = X_train.shape[1], X_train.shape[2]
# 数据展平
X_train_flatten = X_train.reshape(X_train.shape[0], img_rows*img_cols)
X_predict_flatten = X_predict.reshape(X_predict.shape[0], img_rows*img_cols)
val_max = np.max(X_train_flatten)
X_train_flatten = (X_train_flatten/val_max).astype("float32")
X_predict_flatten = (X_predict_flatten/val_max).astype("float32")
print(X_train_flatten.shape)
print(X_predict_flatten.shape)

n = len(X_train_flatten)
p = X_train_flatten.shape[1]
outNum = len(set(y_train))
hls = np.round(np.sqrt(p*outNum)).astype("int32")
batch_size = np.ceil(n/p/10).astype("int32") # (n/batch_size)/p = 10

# svm
clf = svm.SVC()
t0 = pd.Timestamp.now()
clf.fit(X_train_flatten, y_train)
t1 = pd.Timestamp.now(); t1-t0 # 00:10:16.450259
#clf.support_vectors_
#len(clf.support_) / len(y) # 0.3271
y_pred = clf.predict(X_predict_flatten)
sum(y_pred == y_predict) / len(y_predict) # 0.9446

# LR
clf = LogisticRegression(penalty="l2", 
                         tol=0.0001, 
                         solver="lbfgs", 
                         max_iter=100, 
                         multi_class="multinomial")
t0 = pd.Timestamp.now()
clf.fit(X_train_flatten, y_train)
t1 = pd.Timestamp.now(); t1-t0 # 00:00:48.229758
y_pred = clf.predict(X_predict_flatten)
sum(y_pred == y_predict) / len(y_predict) # 0.9263

# MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(hls,hls,hls,hls,hls), 
                    activation="tanh", 
                    solver="adam", 
                    alpha=0.0001, 
                    batch_size=batch_size, 
                    learning_rate_init=0.001, 
                    max_iter=50, 
                    tol=0.0001, 
                    beta_1=0.9, 
                    beta_2=0.999, 
                    epsilon=10**-8, 
                    verbose=True)
t0 = pd.Timestamp.now()
clf.fit(X_train_flatten, y_train)
t1 = pd.Timestamp.now(); t1-t0 # 00:01:40.426413
y_pred = clf.predict(X_predict_flatten)
sum(y_pred == y_predict) / len(y_predict) # 0.9762
