# -*- coding: utf-8 -*-
from __future__ import absolute_import # 模块绝对引入
from __future__ import division # 精确除法
from __future__ import print_function
import os
os.getcwd()
os.chdir("D:/my_project/Python_Project/test/DL")
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import pickle
import cv2 as cv
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from keras.preprocessing import image
import cnn_layers_tf as clt

#def img_resize(filename, height, width):
#    pic = image.load_img(filename) # 载入图片
#    pic_array = image.img_to_array(pic, "channels_last").astype("uint8") # 转换成像素矩阵
#    pic_resize = cv.resize(pic_array, dsize=(height,width)) # 调整大小
#    pic_update = Image.fromarray(pic_resize) # 重新拼成图片
#    return pic_update.save(".\\resize_" + filename) # 保存

# 调整大小
def pic_resize(filename, height, width):
#    pic = image.load_img(filename) # 载入图片
    pic = Image.open(filename, mode="r") # 载入图片
    pic_update = pic.resize((height, width), Image.BICUBIC)
    return pic_update.save("./resize_" + filename) # 保存

#def img_rotate(filename, angle=20, scale=1):
#    pic = image.load_img(filename) # 载入图片
#    pic_array = image.img_to_array(pic, "channels_last").astype("uint8") # 转换成像素矩阵
#    H, W, C = pic_array.shape
#    M = cv.getRotationMatrix2D((int(W/2), int(H/2)), angle, scale) # angle:旋转角度，scale:放大缩小
#    pic_rotate = cv.warpAffine(pic_array, M, dsize=(W,H))
#    pic_update = Image.fromarray(pic_rotate) # 重新拼成图片
#    return pic_update.save(".\\rotate_" + filename) # 保存

# 随机旋转
def pic_rotate(filename, rotate_from, rotate_to):
    pic = Image.open(filename, mode="r") # 载入图片
    random_angle = np.random.randint(rotate_from, rotate_to)
    pic_update = pic.rotate(random_angle, Image.BICUBIC)
    return pic_update.save("./rotate_" + filename) # 保存

# 图片剪裁
def pic_crop(filename, init_range, crop_size):
    pic = Image.open(filename, mode="r") # 载入图片
    W, H = pic.size
    x0 = np.random.randint(0, init_range)
    y0 = np.random.randint(0, init_range)
    x1 = x0+crop_size
    y1 = y0+crop_size
    assert x0 <= W and x1 <= W, "pic_update is cross the border"
    assert y0 <= W and y1 <= H, "pic_update is cross the border"
    random_region = (x0, y0, x1, y1)
    pic_update = pic.crop(random_region)
    return pic_update.save(".\\crop_" + filename) # 保存

#def img_sub(filename, shrink_rate, HH, WW):
#    pic = image.load_img(filename) # 载入图片
#    pic_array = image.img_to_array(pic, "channels_last").astype("uint8") # 转换成像素矩阵
#    H, W, C = pic_array.shape
#    patchSize = (int(W*shrink_rate), int(H*shrink_rate)) # 剪裁后图片的大小
#    W_remain = (W-patchSize[0])/2 # 所剩边界宽度
#    H_remain = (H-patchSize[1])/2 # 所剩边界宽度
#    center = (int(W/2+WW), int(H/2+HH)) # 确定剪裁中心点
#    assert np.abs(WW) <= W_remain, "abs(WW) is too big to cross the border"
#    assert np.abs(HH) <= H_remain, "abs(HH) is too big to cross the border"
#    pic_sub = cv.getRectSubPix(pic_array, patchSize, center)
#    pic_update = Image.fromarray(pic_sub) # 重新拼成图片
#    return pic_update.save(".\\sub_" + filename) # 保存
#img_sub(filename, shrink_rate=0.5, HH=-220, WW=-320)

#def img_colorShifting(filename):
#    pic = image.load_img(filename) # 载入图片
#    pic_array = image.img_to_array(pic, "channels_first").astype("float32") # 转换成像素矩阵
#    # 通道分离
#    pic_R = pic_array[0]
#    pic_G = pic_array[1]
#    pic_B = pic_array[2]
#    colorShift = np.random.randint(-50, 50, 3) # 随机colorShift
#    # 颜色变换
#    pic_R += colorShift[0]
#    pic_G += colorShift[1]
#    pic_B += colorShift[2]
#    # 控制最大最小值
#    pic_R = np.where(pic_R > 255, 255, pic_R).astype("uint8")
#    pic_R = np.where(pic_R < 0, 0, pic_R).astype("uint8")
#    pic_G = np.where(pic_G > 255, 255, pic_G).astype("uint8")
#    pic_G = np.where(pic_G < 0, 0, pic_G).astype("uint8")
#    pic_B = np.where(pic_B > 255, 255, pic_B).astype("uint8")
#    pic_B = np.where(pic_B < 0, 0, pic_B).astype("uint8")
#    # 通道合成
#    R = Image.fromarray(pic_R)
#    G = Image.fromarray(pic_G)
#    B = Image.fromarray(pic_B)
#    pic_update = Image.merge(mode="RGB", bands=(R,G,B))
#    return pic_update.save(".\\shift_" + filename) # 保存

# 颜色调整
def pic_colorShift(filename):
    pic = Image.open(filename, mode="r") # 载入图片
    random_factor = np.random.randint(0, 31) / 10
    pic_update = ImageEnhance.Color(pic).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10
    pic_update = ImageEnhance.Brightness(pic_update).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10
    pic_update = ImageEnhance.Contrast(pic_update).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10
    pic_update = ImageEnhance.Sharpness(pic_update).enhance(random_factor) # 调整图像锐度
    return pic_update.save(".\\shift_" + filename) # 保存   

# 高斯噪声
def pic_gaussNois(filename, mean, std):
    pic = Image.open(filename, mode="r") # 载入图片
    pic_array = image.img_to_array(pic, "channels_first").astype("float32") # 转换成像素矩阵
    C, H, W = pic_array.shape
    pic_update = np.zeros((H, W), dtype=np.float32)
    pic_update = np.expand_dims(pic_update, axis=0)
    for i in range(C):
        # i = 0
        pic_C = pic_array[i]
        pic_C_flatten = pic_C.flatten() # 展平
        random_factor = np.random.normal(mean, std, len(pic_C_flatten)) # 高斯噪声
        pic_C_flatten += random_factor
        pic_C_update = pic_C_flatten.reshape(H, W) # 变回
        pic_C_update = np.expand_dims(pic_C_update, axis=0)
        pic_update = np.concatenate((pic_update, pic_C_update), axis=0)
    pic_update = pic_update[1:].transpose(1,2,0).astype("uint8")
    pic_update = Image.fromarray(pic_update) # 重新拼成图片
    return pic_update.save(".\\gaussNois_" + filename) # 保存   
      
# 图片翻转
def img_flip(filename, flipCode):
    pic = Image.open(filename, mode="r") # 载入图片
    pic_array = image.img_to_array(pic, "channels_last").astype("uint8") # 转换成像素矩阵
    pic_flip = cv.flip(pic_array, flipCode) # 镜像， flipCode>0 水平； flipCode=0 垂直； flipCode<0 水平+垂直
    pic_update = Image.fromarray(pic_flip) # 重新拼成图片
    return pic_update.save(".\\flip_" + filename) # 保存
    
# 图片模糊
def img_blur(filename, ksize):
    pic = Image.open(filename, mode="r") # 载入图片
    pic_array = image.img_to_array(pic, "channels_last").astype("uint8") # 转换成像素矩阵
    pic_blur = cv.blur(pic_array, ksize) # 模糊处理
    pic_update = Image.fromarray(pic_blur) # 重新拼成图片
    return pic_update.save("./blur_" + filename) # 保存


# PCA jitter
def pca(x):
    '''
    直接算协方差矩阵，做特征值分解
    '''
    x_cov = np.cov(x.T)
    S, V = np.linalg.eig(x_cov)
    return S, V

def img_pca(filename, mean, std):
    pic = Image.open(filename, mode="r") # 载入图片
    pic_array = image.img_to_array(pic, "channels_first").astype("float32") # 转换成像素矩阵
    # reshape
    C, H, W = pic_array.shape
    pic_reshape = pic_array.reshape(H*W, C)
    # pca
    S, V = pca(pic_reshape)
    # 高斯扰动
    alpha = np.random.normal(mean, std, 3)
    S_new = alpha * S
    add = S_new.dot(V)
    pic_array[0] += add[0]
    pic_array[1] += add[1]
    pic_array[2] += add[2]
    pic_array_new = pic_array.transpose(1,2,0).astype("uint8")
    pic_update = Image.fromarray(pic_array_new) # 重新拼成图片
    return pic_update.save("./pca_" + filename) # 保存
