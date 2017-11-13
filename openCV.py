# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
os.getcwd()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.utils import plot_model
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import cv2 as cv # pip install --upgrade opencv_python

# load_img
img = cv.imread(".\\elephant.jpg")
cv.imshow("Image", img); cv.waitKey(0)
print(img.shape) # (1080, 1920, 3)
#plt.imshow(img)

X1 = image.img_to_array(img, data_format=K.image_data_format())
print(X1.shape)
X1 = np.expand_dims(X1, axis=0)
X1 = preprocess_input(X1, mode="tf")

img = image.load_img(path=".\\elephant.jpg")
plt.imshow(img)
X2 = image.img_to_array(img, data_format=K.image_data_format())
print(X2.shape)
X2 = np.expand_dims(X2, axis=0)
X2 = preprocess_input(X2, mode="tf")
X3 = np.concatenate((X1, X2), axis=0)

# 创建图片
emptyImage = np.zeros(img.size, np.uint8)

# 复制图像
img2 = img.copy()
plt.imshow(img2)

# 保存图像
cv.imwrite(".\\DSC07961_2.jpg", img2, [int(cv.IMWRITE_JPEG_QUALITY), 0])
cv.imwrite(".\\DSC07961_2.jpg", img2, [int(cv.IMWRITE_JPEG_QUALITY), 100])

# 分离通道
R, G, B = img[:,:,2], img[:,:,1], img[:,:,0] # B, G, R = cv.split(img)
cv.namedWindow("Image") 
cv.imshow("Blue", R)
cv.imshow("Red", G)
cv.imshow("Green", B)
cv.waitKey(0)
cv.destroyAllWindows()

# 合并通道
img_stack = np.dstack([B, G, R])
plt.imshow(img)

kernel = cv.getStructuringElement(cv.MORPH_RECT,(3, 3))
# 腐蚀
eroded = cv.erode(img, kernel)
cv.imshow("Image", eroded); cv.waitKey(0)
# 膨胀
dilated = cv.dilate(img,kernel) 
cv.imshow("Image", dilated); cv.waitKey(0)

# 检测边缘
result = cv.absdiff(eroded, dilated) # 腐蚀, 膨胀相减，得到
cv.imshow("Image", result); cv.waitKey(0) # 检测边缘灰度图
retval, result = cv.threshold(result, 40, 255, cv.THRESH_BINARY) # 二值化
cv.imshow("Image", result); cv.waitKey(0)
result = cv.bitwise_not(result) # 反色
cv.imshow("Image", result); cv.waitKey(0)

# 用低通滤波来平滑图像
dst = cv.blur(img, (5,5))
cv.imshow("Image", dst); cv.waitKey(0)

# 高斯模糊
gaussianResult = cv.GaussianBlur(img, (5,5), 1.5)
cv.imshow("Image", gaussianResult); cv.waitKey(0)

# 使用中值滤波消除噪点
result = cv.medianBlur(img, 5)  # 函数返回处理结果，第一个参数是待处理图像，第二个参数是孔径的尺寸，一个大于1的奇数
cv.imshow("Image", result); cv.waitKey(0)

# sobel
x = cv.Sobel(img, ddepth=cv.CV_16S, dx=1, dy=0)
y = cv.Sobel(img, ddepth=cv.CV_16S, dx=0, dy=1)
absX = cv.convertScaleAbs(x) # 转回uint8  
absY = cv.convertScaleAbs(y)
dst = cv.addWeighted(src1=absX, alpha=0.5, src2=absY, beta=0.5, gamma=0) # alpha是第一幅图片中元素的权重，beta是第二个的权重，gamma是加到最后结果上的一个值
cv.imshow("Image", dst); cv.waitKey(0)
