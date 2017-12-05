# -*- coding: utf-8 -*-
import os
os.getcwd()
from __future__ import absolute_import # 模块绝对引入
from __future__ import division # 精确除法
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:\\my_project\\Python_Project\\test\\NN\\tf", one_hot=True) # 读取数据集
# 建立抽象模型
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
z = tf.matmul(x, w) + b
a = tf.nn.softmax(z)
# 定义损失函数和训练方法
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(a), reduction_indices=1)) # -np.sum(np.log(a[np.arange(N), y])) / N
opti = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = opti.minimize(loss)
# 定义评估标准
is_equal = tf.equal(tf.argmax(a, axis=1), tf.argmax(y, axis=1)) # np.sum(a == y)
accu = tf.reduce_mean(tf.cast(is_equal, dtype=tf.float32)) # /len(y), tf.cast将boolean数组转成int数组

# 训练
sess = tf.InteractiveSession() # 建立交互式会话
tf.global_variables_initializer().run() # 所有变量初始化
#init = tf.global_variables_initializer()
#sess.run(init)
for i in range(1000):
    x_batch, y_batch = mnist.train.next_batch(100) # 获得一批100个数据
    train.run({x:x_batch, y:y_batch}) # 给训练模型提供输入和输出
print(sess.run(fetches=accu, feed_dict={x:mnist.test.images, y:mnist.test.labels}))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
tf.nn.max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)

