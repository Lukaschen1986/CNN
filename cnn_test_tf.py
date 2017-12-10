# -*- coding: utf-8 -*-
from __future__ import absolute_import # 模块绝对引入
from __future__ import division # 精确除法
from __future__ import print_function
import os
os.getcwd()
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from scipy.stats import itemfreq
from sklearn.cross_validation import train_test_split
import cnn_layers_tf as clt

#mnist = input_data.read_data_sets("D:\\my_project\\Python_Project\\test\\NN\\MNIST_data", one_hot=True)

#def load_CIFAR_batch(path, data_format):
#    """ load single batch of cifar """
#    with open(path, "rb") as f:
#        dicts = pickle.load(f, encoding="latin1")
#        X = dicts["data"]
#        y = dicts["labels"]
#        label = dicts["filenames"]
#        if data_format == "channels_first":
#            X = X.reshape(10000, 3, 32, 32)
#        elif data_format == "channels_last":
#            X = X.reshape(10000, 32, 32, 3)
#        else:
#            raise ValueError("data_format must in ('channels_first', channels_last)")
#        y = np.array(y)
#        return X, y, label
#    
#X_1, y_1, label_1 = load_CIFAR_batch("D:\\file\\Py_project\\CNN\\cifar-10-python\\data_batch_1", "channels_first")
#X_2, y_2, label_2 = load_CIFAR_batch("D:\\file\\Py_project\\CNN\\cifar-10-python\\data_batch_2", "channels_first")
#X_3, y_3, label_3 = load_CIFAR_batch("D:\\file\\Py_project\\CNN\\cifar-10-python\\data_batch_3", "channels_first")
#X_4, y_4, label_4 = load_CIFAR_batch("D:\\file\\Py_project\\CNN\\cifar-10-python\\data_batch_4", "channels_first")
#X_5, y_5, label_5 = load_CIFAR_batch("D:\\file\\Py_project\\CNN\\cifar-10-python\\data_batch_5", "channels_first")
#X_test, y_test, ls = load_CIFAR_batch("D:\\file\\Py_project\\CNN\\cifar-10-python\\test_batch", "channels_first")
#
#X = np.concatenate((X_1, X_2, X_3, X_4, X_5), axis=0)
#y = np.concatenate((y_1, y_2, y_3, y_4, y_5))
#
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
#itemfreq(y_valid)

#f = open("D:\\file\\Py_project\\CNN\\cifar-10-python\\X_train.txt", "wb")
#pickle.dump(X_train, f)
#f = open("D:\\file\\Py_project\\CNN\\cifar-10-python\\y_train.txt", "wb")
#pickle.dump(y_train, f)
#f = open("D:\\file\\Py_project\\CNN\\cifar-10-python\\X_valid.txt", "wb")
#pickle.dump(X_valid, f)
#f = open("D:\\file\\Py_project\\CNN\\cifar-10-python\\y_valid.txt", "wb")
#pickle.dump(y_valid, f)
#f = open("D:\\file\\Py_project\\CNN\\cifar-10-python\\X_test.txt", "wb")
#pickle.dump(X_test, f)
#f = open("D:\\file\\Py_project\\CNN\\cifar-10-python\\y_test.txt", "wb")
#pickle.dump(y_test, f)
#f.close()

f = open("D:\\file\\Py_project\\CNN\\cifar-10-python\\X_train.txt", "rb")
x_train = pickle.load(f)
f = open("D:\\file\\Py_project\\CNN\\cifar-10-python\\y_train.txt", "rb")
y_train = pickle.load(f)
f = open("D:\\file\\Py_project\\CNN\\cifar-10-python\\X_valid.txt", "rb")
x_valid = pickle.load(f)
f = open("D:\\file\\Py_project\\CNN\\cifar-10-python\\y_valid.txt", "rb")
y_valid = pickle.load(f)
f = open("D:\\file\\Py_project\\CNN\\cifar-10-python\\X_test.txt", "rb")
x_test = pickle.load(f)
f = open("D:\\file\\Py_project\\CNN\\cifar-10-python\\y_test.txt", "rb")
y_test = pickle.load(f)
f.close()

# 把数据变成float32更精确
val_max = np.max(x_train)
x_train = (x_train/val_max).astype("float32")
x_valid = (x_valid/val_max).astype("float32")
x_test = (x_test/val_max).astype("float32")

x_train = x_train.transpose(0,2,3,1)
x_valid = x_valid.transpose(0,2,3,1)
x_test = x_test.transpose(0,2,3,1)

y_train_oh = clt.one_hot(y_train)
y_valid_oh = clt.one_hot(y_valid)
y_test_oh = clt.one_hot(y_test)

x_train = x_train[0:20000]
y_train_oh = y_train_oh[0:20000]
y_train = y_train[0:20000]

_, H, W, C = x_train.shape
out_num = len(set(y_train))

x = tf.placeholder(tf.float32, shape=[None, H, W, C])
y = tf.placeholder(tf.float32, shape=[None, out_num])
keep_prob = tf.placeholder(tf.float32)
l2_lam = tf.placeholder(tf.float32)

out1 = clt.conv_bn_relu_x2_pool(x, 
                                filters=32, 
                                kernel_size=(5,5), 
                                conv_strides=1, 
                                kernel_initializer=tf.random_normal_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lam),
                                keep_prob=keep_prob,
                                pool_size=(2,2), 
                                pool_strides=2, 
                                trainable=True)
out1.shape

out2 = clt.conv_bn_relu_x2_pool(out1, 
                                filters=64, 
                                kernel_size=(3,3), 
                                conv_strides=1, 
                                kernel_initializer=tf.random_normal_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lam),
                                keep_prob=keep_prob,
                                pool_size=(2,2), 
                                pool_strides=2, 
                                trainable=True)
out2.shape

out3 = clt.conv_bn_relu_x2_pool(out2, 
                                filters=128, 
                                kernel_size=(3,3), 
                                conv_strides=1, 
                                kernel_initializer=tf.random_normal_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lam),
                                keep_prob=keep_prob,
                                pool_size=(2,2), 
                                pool_strides=2, 
                                trainable=True)
out3.shape

out4 = clt.conv_bn_relu_x2_pool(out3, 
                                filters=256, 
                                kernel_size=(3,3), 
                                conv_strides=1, 
                                kernel_initializer=tf.random_normal_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lam),
                                keep_prob=keep_prob,
                                pool_size=(2,2), 
                                pool_strides=2, 
                                trainable=True)
out4.shape

out4_flatten = clt.flatten(out4)
out4_flatten.shape

out5 = clt.affine_bn_relu(out4_flatten, 
                          units=512, 
                          kernel_initializer=tf.random_normal_initializer(), 
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lam), 
                          keep_prob=keep_prob,
                          trainable=True)
out5.shape

out6 = clt.affine_bn_relu(out5, 
                          units=128, 
                          kernel_initializer=tf.random_normal_initializer(), 
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lam), 
                          keep_prob=keep_prob,
                          trainable=True)
out6.shape

probs, loss = clt.affine_bn_softmax_loss(out6, y, 
                                         units=10, 
                                         kernel_initializer=tf.random_normal_initializer(), 
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lam), 
                                         trainable=True)
probs.shape

opti_obj = clt.opti(func="adam", loss=loss, config={"learning_rate":0.0001})
accu = clt.metrics(probs, y)

#x_train, y_train = mnist.train.images, mnist.train.labels
#N = 3000
#x_train, y_train = x_train[0:N], y_train[0:N]
#y_train = np.float32(y_train)

batch_size=128
x_train.shape[0] // batch_size # batch_sample

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
loss_valid_global = []
accu_valid_global = []
#with tf.device("/gpu:0"):
for epoch in range(20):
    loss_train_epoch = accu_train_epoch = 0.0
    loss_valid_epoch = accu_valid_epoch = 0.0
    # train
    batch_sample = 0.0
    for x_batch, y_batch in clt.batch_func(x_train, y_train_oh, batch_size):
        batch_sample += 1
#        x_batch_rshp = x_batch.reshape(x_batch.shape[0], 32, 32, 3)
        _, loss_train, accu_train = sess.run([opti_obj, loss, accu], 
                                             feed_dict={x:x_batch, y:y_batch, keep_prob:0.5, l2_lam:0.0001})
        loss_train_epoch += loss_train
        accu_train_epoch += accu_train
    loss_train_epoch /= batch_sample
    accu_train_epoch /= batch_sample
    # valid
    loss_valid_epoch, accu_valid_epoch = sess.run([loss, accu],
                                                  feed_dict={x:x_valid, y:y_valid_oh, keep_prob:1.0, l2_lam:0.0001})
    # append
    loss_train_global.append(loss_train_epoch)
    accu_train_global.append(accu_train_epoch)
    loss_valid_global.append(loss_valid_epoch)
    accu_valid_global.append(accu_valid_epoch)
    print("epoch: %d, loss_train: %g, accu_train: %g, loss_valid: %g, accu_valid: %g" % (epoch, loss_train_epoch, accu_train_epoch, loss_valid_epoch, accu_valid_epoch))    

probs_test, loss_test, accu_test = sess.run([probs,loss,accu], feed_dict={x:x_test, y:y_test_oh, keep_prob:1.0})
print("loss_test: %g, accu_test: %g" % (loss_test, accu_test))

df = pd.DataFrame({"loss_train": loss_train_global,
                   "loss_valid": loss_valid_global,
                   "loss_test": loss_test})
df.plot()

saver.save(sess, save_path=".\\model", global_step=0, write_meta_graph=True, write_state=True)

# load
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
path = tf.train.get_checkpoint_state(".\\")
saver.restore(sess, path.model_checkpoint_path)

#x_test, y_test = mnist.test.images, mnist.test.labels
#y_test = np.float32(y_test)
#x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
probs_test, loss_test, accu_test = sess.run([probs,loss,accu], feed_dict={x:x_test, y:y_test, keep_prob:1.0})
y_pred = np.argmax(probs_test, axis=1)
y_test_new = np.argmax(y_test, axis=1)
pd.crosstab(index=y_test_new, columns=y_pred, margins=True)

