# -*- coding: utf-8 -*-
from __future__ import absolute_import # 模块绝对引入
from __future__ import division # 精确除法
from __future__ import print_function
import os
os.getcwd()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cnn_layers_tf as clt

mnist = input_data.read_data_sets("D:\\my_project\\Python_Project\\test\\NN\\MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None,28,28,1])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)


out1 = clt.conv_bn_relu_pool(x, 
                         filters=32, 
                         kernel_size=(5,5), 
                         conv_strides=1, 
                         kernel_initializer=tf.random_normal_initializer(), 
                         pool_size=(2,2), 
                         pool_strides=2, 
                         trainable=True)
out1.shape
out2 = clt.conv_bn_relu_pool(out1, 
                         filters=64, 
                         kernel_size=(3,3), 
                         conv_strides=1, 
                         kernel_initializer=tf.random_normal_initializer(), 
                         pool_size=(2,2), 
                         pool_strides=2, 
                         trainable=True)
out2.shape
out2_flatten = clt.flatten(out2)
out2_flatten.shape

out3 = clt.affine_bn_relu(out2_flatten, 
                      units=128, 
                      kernel_initializer=tf.random_normal_initializer(), 
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001), 
                      keep_prob=0.8,
                      trainable=True)
out3.shape

probs, loss = clt.affine_bn_softmax_loss(out3, y, 
                                     units=10, 
                                     kernel_initializer=tf.random_normal_initializer(), 
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001), 
                                     trainable=True)

opti_obj = clt.opti(func="adam", loss=loss)

accu = clt.metrics(probs, y)

x_train, y_train = mnist.train.images, mnist.train.labels
N = 3000
x_train, y_train = x_train[0:N], y_train[0:N]
y_train = np.float32(y_train)

batch_size=10
#N // batch_size

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
'''
log_device_placement=True : 是否打印设备分配日志
allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
'''
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver() # 生成saver

loss_train_global = []
accu_train_global = []
#with tf.device("/gpu:0"):
for epoch in range(5):
    loss_train_epoch = accu_train_epoch = 0.0
    batchs = 0.0
    for x_batch, y_batch in clt.batch_func(x_train, y_train, batch_size):
        batchs += 1
        x_batch_rshp = x_batch.reshape(x_batch.shape[0], 28, 28, 1)
        _, loss_train, accu_train = sess.run([opti_obj,loss,accu], feed_dict={x:x_batch_rshp, y:y_batch, keep_prob:1.0})
        loss_train_epoch += loss_train
        accu_train_epoch += accu_train
    loss_train_epoch /= batchs
    accu_train_epoch /= batchs
    loss_train_global.append(loss_train_epoch)
    accu_train_global.append(accu_train_epoch)
    print("epoch: %d, loss_train: %g, accu_train: %g" % (epoch, loss_train_epoch, accu_train_epoch))

saver.save(sess, save_path=".\\model", global_step=0, write_meta_graph=True, write_state=True)

# load
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
path = tf.train.get_checkpoint_state(".\\")
saver.restore(sess, path.model_checkpoint_path)

x_test, y_test = mnist.test.images, mnist.test.labels
y_test = np.float32(y_test)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
probs_test, loss_test, accu_test = sess.run([probs,loss,accu], feed_dict={x:x_test, y:y_test, keep_prob:1.0})
y_pred = np.argmax(probs_test, axis=1)
y_test_new = np.argmax(y_test, axis=1)
pd.crosstab(index=y_test_new, columns=y_pred, margins=True)
