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
print(img.shape) # (224, 224, 3)
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
emptyImage = np.zeros(img.shape, np.uint8)

# 复制图像
img2 = img.copy()
plt.imshow(img2)

# 保存图像
cv.imwrite(".\\DSC07961_2.jpg", img2, [int(cv.IMWRITE_JPEG_QUALITY), 0])
cv.imwrite(".\\DSC07961_2.jpg", img2, [int(cv.IMWRITE_JPEG_QUALITY), 100])

# 通道分离
R, G, B = img[:,:,2], img[:,:,1], img[:,:,0] # B, G, R = cv.split(img)
cv.namedWindow("Image") 
cv.imshow("Blue", R)
cv.imshow("Red", G)
cv.imshow("Green", B)
cv.waitKey(0)
cv.destroyAllWindows()

# 通道合并
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

# sobel算子:图像中的边缘区域，像素值会发生“跳跃”，对这些像素求导，在其一阶导数在边缘位置为极值，这就是Sobel算子使用的原理——极值处就是边缘
x = cv.Sobel(img, ddepth=cv.CV_16S, dx=1, dy=0)
y = cv.Sobel(img, ddepth=cv.CV_16S, dx=0, dy=1)
absX = cv.convertScaleAbs(x) # 转回uint8  
absY = cv.convertScaleAbs(y)
dst = cv.addWeighted(src1=absX, alpha=0.5, src2=absY, beta=0.5, gamma=0) # alpha是第一幅图片中元素的权重，beta是第二个的权重，gamma是加到最后结果上的一个值
cv.imshow("Image", dst); cv.waitKey(0)

# Laplacian算子：Laplace函数实现的方法是先用Sobel 算子计算二阶x和y导数，再求和
gray_lap = cv.Laplacian(img, ddepth=cv.CV_16S, ksize=3) # ksize是算子的大小，必须为1、3、5、7。默认为1
dst = cv.convertScaleAbs(gray_lap)
cv.imshow("Image", dst); cv.waitKey(0)

# Canny边缘检测：其中较大的阈值2用于检测图像中明显的边缘，但一般情况下检测的效果不会那么完美，边缘检测出来是断断续续的。所以这时候用较小的第一个阈值用于将这些间断的边缘连接起来
img = cv.GaussianBlur(img, (3,3), 0)
canny = cv.Canny(img, threshold1=50, threshold2=150, apertureSize=3) # 可选参数中apertureSize就是Sobel算子的大小
cv.imshow("Image", canny); cv.waitKey(0)

# 霍夫变换：Hough变换是经典的检测直线的算法
#img = cv.GaussianBlur(img, (3,3), 0)
#edges = cv.Canny(img, threshold1=50, threshold2=150, apertureSize=3)
#lines = cv.HoughLines(edges, rho=1, theta=np.pi/180, threshold=118) #这里对最后一个参数使用了经验型的值
#cv.imshow("Image", lines); cv.waitKey(0)

# 轮廓检测
img = cv.imread(".\\elephant.jpg")
gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY) # 先转成灰度
ret, binary = cv.threshold(gray, thresh=127, maxval=255, type=cv.THRESH_BINARY) # 二值图
contours, hierarchy, x = cv.findContours(binary, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours, contourIdx=-1, color=(0,0,255), lineType=3) 


# 图片缩放
img = cv.imread(".\\elephant.jpg")
cv.imshow("Image", img); cv.waitKey(0)
print(img.shape) # (224, 224, 3)

fscale_width = 0.7; fscale_height = 0.3
width, height = int(img.shape[0]*fscale_width), int(img.shape[1]*fscale_height)
img_2 = cv.resize(img, dsize=(width,height))
cv.imshow("Image", img_2); cv.waitKey(0)

# 图像平移
img = cv.imread(".\\elephant.jpg")
width, height = img.shape[0], img.shape[1]
M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img, M, dsize=(width,height))
cv.imshow("Image", dst); cv.waitKey(0)

# 图像旋转
img = cv.imread(".\\elephant.jpg")
width, height, channel = img.shape
angle = 180; scale = 1
M = cv.getRotationMatrix2D((width/2,height/2), angle, scale)
dst = cv.warpAffine(img, M, dsize=(width,height))
cv.namedWindow("Image") 
cv.imshow("Image", dst); cv.waitKey(0)
cv.destroyAllWindows()

# 图像翻转
img = cv.imread(".\\elephant.jpg")
img_vertical = cv.flip(img, flipCode=0) # 垂直
cv.imshow("Image", img_vertical); cv.waitKey(0)
img_horizontal = cv.flip(img, flipCode=1) # 水平
cv.imshow("Image", img_horizontal); cv.waitKey(0)
img_both = cv.flip(img, flipCode=-1) # 垂直+水平
cv.imshow("Image", img_both); cv.waitKey(0)

# 仿射变换
img = cv.imread(".\\elephant.jpg")
width, height, channel = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv.getAffineTransform(pts1, pts2)
dst = cv.warpAffine(img, M, (width,height))
cv.imshow("Image", dst); cv.waitKey(0)

# Face Detection
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load('D:\\file\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
eye_cascade.load('D:\\file\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml')
img = cv.imread(".\\pu.jpg")
#cv.imshow("Image", img); cv.waitKey(0)
print(img.shape) # (675, 1200, 3)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow("Image", gray); cv.waitKey(0)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)
len(faces)
"发现%d个人脸!" % (len(faces))
for x,y,w,h in faces:
    img = cv.rectangle(img, pt1=(x,y), pt2=(x+w,y+h), color=(255,0,0), thickness=2, lineType=1)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for ex,ey,ew,eh in eyes:
        cv.rectangle(roi_color, pt1=(ex,ey), pt2=(ex+ew,ey+eh), color=(0,255,0), thickness=1, lineType=1)
cv.imshow("Image", img); cv.waitKey(0)

x,y,w,h = faces[1]
crop = img[y:y+h, x:x+w]
cv.imshow("Image", crop); cv.waitKey(0)

# Body Detection
body_cascade = cv.CascadeClassifier("haarcascade_fullbody.xml")
body_cascade.load("D:\\file\\opencv\\sources\\data\\haarcascades\\haarcascade_fullbody.xml")
img = cv.imread(".\\two.jpg")
cv.imshow("Image", img); cv.waitKey(0)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Image", gray); cv.waitKey(0)

bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)
for x,y,w,h in bodies:
    img = cv.rectangle(img, pt1=(x,y), pt2=(x+w,y+h), color=(255,0,0), thickness=2, lineType=1)
cv.imshow("Image", img); cv.waitKey(0)
