# -*- coding: utf-8 -*-
from __future__ import absolute_import # 模块绝对引入
from __future__ import division # 精确除法
from __future__ import print_function
import os
os.getcwd()
os.chdir(".\\picture\\sample")
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import cv2 as cv
from PIL import Image
from keras.preprocessing import image
import cnn_layers_tf as clt

def img_resize(filename, height, width):
    pic = image.load_img(filename) # 载入图片
    pic_array = image.img_to_array(pic, "channels_last").astype("uint8") # 转换成像素矩阵
    pic_resize = cv.resize(pic_array, dsize=(height,width)) # 调整大小
    pic_update = Image.fromarray(pic_resize) # 重新拼成图片
    return pic_update.save(".\\resize_" + filename) # 保存

def img_flip(filename, flipCode):
    pic = image.load_img(filename) # 载入图片
    pic_array = image.img_to_array(pic, "channels_last").astype("uint8") # 转换成像素矩阵
    pic_flip = cv.flip(pic_array, flipCode) # 镜像， flipCode>0 水平； flipCode=0 垂直； flipCode<0 水平+垂直
    pic_update = Image.fromarray(pic_flip) # 重新拼成图片
    return pic_update.save(".\\flip_" + filename) # 保存

filename = "54895_room_3.jpg"
def img_blur(filename, method, config):
    pic = image.load_img(filename) # 载入图片
    pic_array = image.img_to_array(pic, "channels_last").astype("uint8") # 转换成像素矩阵
    if method == "Blur":
        if config is None: config = {}
        config.setdefault("ksize", (5,5))
        pic_blur = cv.blur(pic_array, config["ksize"]) # 模糊处理
    elif method == "GaussianBlur":
        if config is None: config = {}
        config.setdefault("ksize", (5,5))
        config.setdefault("sigma", 1.5)
        pic_blur = cv.GaussianBlur(pic_array, config["ksize"], config["sigma"]) # 高斯模糊
    else:
        raise ValueError("method should be in ('Blur', 'GaussianBlur')")
    pic_update = Image.fromarray(pic_blur) # 重新拼成图片
    return pic_update.save(".\\blur_" + filename) # 保存
img_blur(filename, method="GaussianBlur", config={"sigma":1.5})
    
def img_rotate(filename, angle=20, scale=1):
    pic = image.load_img(filename) # 载入图片
    pic_array = image.img_to_array(pic, "channels_last").astype("uint8") # 转换成像素矩阵
    H, W, C = pic_array.shape
    M = cv.getRotationMatrix2D((int(W/2), int(H/2)), angle, scale) # angle:旋转角度，scale:放大缩小
    pic_rotate = cv.warpAffine(pic_array, M, dsize=(W,H))
    pic_update = Image.fromarray(pic_rotate) # 重新拼成图片
    return pic_update.save(".\\rotate_" + filename) # 保存

shrink_rate=0.8; WW=200; HH=300
def img_sub(filename, shrink_rate, HH, WW):
    pic = image.load_img(filename) # 载入图片
    pic_array = image.img_to_array(pic, "channels_last").astype("uint8") # 转换成像素矩阵
    H, W, C = pic_array.shape
    patchSize = (int(W*shrink_rate), int(H*shrink_rate)) # 剪裁后图片的大小
    W_remain = (W-patchSize[0])/2 # 所剩边界宽度
    H_remain = (H-patchSize[1])/2 # 所剩边界宽度
    center = (int(W/2+WW), int(H/2+HH)) # 确定剪裁中心点
    assert np.abs(WW) <= W_remain, "abs(WW) is too big to cross the border"
    assert np.abs(HH) <= H_remain, "abs(HH) is too big to cross the border"
    pic_sub = cv.getRectSubPix(pic_array, patchSize, center)
    pic_update = Image.fromarray(pic_sub) # 重新拼成图片
    return pic_update.save(".\\sub_" + filename) # 保存
img_sub(filename, shrink_rate=0.5, HH=-220, WW=-320)
    
def img_colorShifting(filename):
    pic = image.load_img(filename) # 载入图片
    pic_array = image.img_to_array(pic, "channels_first").astype("float32") # 转换成像素矩阵
    # 通道分离
    pic_R = pic_array[0]
    pic_G = pic_array[1]
    pic_B = pic_array[2]
    colorShift = np.random.randint(-50, 50, 3) # 随机colorShift
    # 颜色变换
    pic_R += colorShift[0]
    pic_G += colorShift[1]
    pic_B += colorShift[2]
    # 控制最大最小值
    pic_R = np.where(pic_R > 255, 255, pic_R).astype("uint8")
    pic_R = np.where(pic_R < 0, 0, pic_R).astype("uint8")
    pic_G = np.where(pic_G > 255, 255, pic_G).astype("uint8")
    pic_G = np.where(pic_G < 0, 0, pic_G).astype("uint8")
    pic_B = np.where(pic_B > 255, 255, pic_B).astype("uint8")
    pic_B = np.where(pic_B < 0, 0, pic_B).astype("uint8")
    # 通道合成
    R = Image.fromarray(pic_R)
    G = Image.fromarray(pic_G)
    B = Image.fromarray(pic_B)
    pic_update = Image.merge(mode="RGB", bands=(R,G,B))
    return pic_update.save(".\\shift_" + filename) # 保存
filename = "54895_room_3.jpg"
img_colorShifting(filename)

def SVD(A):
    B = A.T.dot(A)
    B_eigenvalue, V = np.linalg.eig(B)
    B_eigenvalue = B_eigenvalue.astype(np.float32)
    V = V.astype(np.float32)
    
    C = A.dot(A.T)
    C_eigenvalue, U = np.linalg.eig(C)
    C_eigenvalue = C_eigenvalue.astype(np.float32)
    U = U.astype(np.float32)
    
    S = np.linalg.inv(U).dot(A.dot(V))
    U = np.round(U, 7)
    S = np.round(S, 7)
    V = np.round(V, 7)
    return U, S, V.T
    
def img_pca(filename):
    pic = image.load_img(filename) # 载入图片
    pic_array = image.img_to_array(pic, "channels_first").astype("float32") # 转换成像素矩阵
    # 通道分离
    pic_R = pic_array[0]
    pic_G = pic_array[1]
    pic_B = pic_array[2]
    # 高斯扰动
    a = np.cov(pic_R)
    lamda, p = np.linalg.eig(a)
    
    
    img = np.asanyarray(pic, dtype = 'float32')  
          
    img = img / 255.0  
    img_size = int(img.size / 3)
    img1 = img.reshape(img_size, 3)  
    img1 = np.transpose(img1)  
    img_cov = np.cov([img1[0], img1[1], img1[2]])  
    lamda, p = np.linalg.eig(img_cov)  
    
    
#    Ur, Sr, VTr = SVD(pic_R)
    
    from sklearn.decomposition import PCA
    PCA(n_components=3).fit(pic_R)
    
    Ur, Sr, VTr = np.linalg.svd(pic_R)
    Sr *= np.random.normal(0, 1.5, len(Sr))
    Sr = np.diag(Sr)
    Sr_new = np.zeros(pic_R.shape, dtype=np.float32)
    Sr_new[0:len(Sr),0:len(Sr)] = Sr
    pic_R_new = Ur.dot(Sr_new).dot(VTr).astype("uint8")
    
    Ug, Sg, VTg = np.linalg.svd(pic_G)
    Sg *= np.random.normal(0, 1.5, len(Sg))
    Sg = np.diag(Sg)
    Sg_new = np.zeros(pic_G.shape, dtype=np.float32)
    Sg_new[0:len(Sg),0:len(Sg)] = Sg
    pic_G_new = Ug.dot(Sg_new).dot(VTg).astype("uint8")
    
    Ub, Sb, VTb = np.linalg.svd(pic_B)
    Sb *= np.random.normal(0, 1.5, len(Sb))
    Sb = np.diag(Sb)
    Sb_new = np.zeros(pic_B.shape, dtype=np.float32)
    Sb_new[0:len(Sb),0:len(Sb)] = Sb
    pic_B_new = Ub.dot(Sb_new).dot(VTb).astype("uint8")
    # 通道合成
    R = Image.fromarray(pic_R_new)
    G = Image.fromarray(pic_G_new)
    B = Image.fromarray(pic_B_new)
    pic_update = Image.merge(mode="RGB", bands=(R,G,B))
    
    
    
    
    
    
    
    np.diag(np.random.normal(0, 0.1, len(Sr))) + Sr
    
    Sr += np.random.normal(0, 0.1, len(Sr))
    Sr = np.diag(Sr)
    
    Ur.dot(Sr_new).shape
    
    Ur.dot(Sr_new).dot(VTr).shape
    
    A = np.random.randn(5,4)
U, S, VT = np.linalg.svd(A)
    
A = pic_R
    
    B_eigenvalue.astype(np.float32)
    
    
    
    
