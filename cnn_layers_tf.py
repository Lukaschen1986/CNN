# -*- coding: utf-8 -*-
from __future__ import division # 精确除法
from __future__ import print_function
import tensorflow as tf
import numpy as np

one_hot = lambda y: np.eye(len(set(y)), dtype=np.float32)[y]

def conv_bn_relu_pool(x, filters, kernel_size, conv_strides, kernel_initializer, kernel_regularizer, keep_prob, pool_size, pool_strides, trainable):
    z = tf.layers.conv2d(x, 
                         filters, 
                         kernel_size, 
                         conv_strides, 
                         padding="same",
                         data_format='channels_last',
                         activation=None, 
                         use_bias=False, 
                         kernel_initializer=None, 
                         kernel_regularizer=None, 
                         trainable=trainable)
    zn = tf.layers.batch_normalization(z, 
                                       axis=-1, 
                                       momentum=0.9, 
                                       epsilon=10**-8, 
                                       center=True, 
                                       scale=True, 
                                       beta_initializer=tf.zeros_initializer(), 
                                       gamma_initializer=tf.ones_initializer(), 
                                       moving_mean_initializer=tf.zeros_initializer(), 
                                       moving_variance_initializer=tf.ones_initializer(), 
                                       trainable=trainable)
    a = tf.nn.relu(zn)
    ad = tf.layers.dropout(a, rate=1-keep_prob)
    out = tf.layers.max_pooling2d(ad, 
                                  pool_size, 
                                  strides=pool_strides)
    return out
# kernel_initializer: tf.random_normal_initializer(); tf.truncated_normal_initializer(); tf.random_uniform_initializer()

#filters=32; kernel_size=(3,3); conv_strides=1; trainable=True
def conv_bn_relu_x2_pool(x, filters, kernel_size, conv_strides, kernel_initializer, kernel_regularizer, keep_prob, pool_size, pool_strides, trainable):
    z1 = tf.layers.conv2d(x, 
                         filters, 
                         kernel_size, 
                         conv_strides, 
                         padding="same",
                         data_format='channels_last',
                         activation=None, 
                         use_bias=False, 
                         kernel_initializer=None, 
                         kernel_regularizer=None, 
                         trainable=trainable)
    zn1 = tf.layers.batch_normalization(z1, 
                                       axis=-1, 
                                       momentum=0.9, 
                                       epsilon=10**-8, 
                                       center=True, 
                                       scale=True, 
                                       beta_initializer=tf.zeros_initializer(), 
                                       gamma_initializer=tf.ones_initializer(), 
                                       moving_mean_initializer=tf.zeros_initializer(), 
                                       moving_variance_initializer=tf.ones_initializer(), 
                                       trainable=trainable)
    a1 = tf.nn.relu(zn1)
    ad1 = tf.layers.dropout(a1, rate=1-keep_prob)
    z2 = tf.layers.conv2d(ad1, 
                         filters, 
                         kernel_size, 
                         conv_strides, 
                         padding="same",
                         data_format='channels_last',
                         activation=None, 
                         use_bias=False, 
                         kernel_initializer=None, 
                         kernel_regularizer=None, 
                         trainable=trainable)
    zn2 = tf.layers.batch_normalization(z2, 
                                       axis=-1, 
                                       momentum=0.9, 
                                       epsilon=10**-8, 
                                       center=True, 
                                       scale=True, 
                                       beta_initializer=tf.zeros_initializer(), 
                                       gamma_initializer=tf.ones_initializer(), 
                                       moving_mean_initializer=tf.zeros_initializer(), 
                                       moving_variance_initializer=tf.ones_initializer(), 
                                       trainable=trainable)
    a2 = tf.nn.relu(zn2)
    ad2 = tf.layers.dropout(a2, rate=1-keep_prob)
    out = tf.layers.max_pooling2d(ad2, 
                                  pool_size, 
                                  strides=pool_strides)
    return out

#def flatten(x):
#    out = tf.layers.flatten(x)
#    return out

def flatten(x):
    _, H, W, C = x.shape
    out = tf.reshape(x, shape=[-1,int(H*W*C)])
    return out

def affine_bn_relu(x, units, kernel_initializer, kernel_regularizer, keep_prob, trainable):
    z = tf.layers.dense(x, 
                        units, 
                        activation=None,
                        use_bias=False,
                        kernel_initializer=None,
                        kernel_regularizer=None,
                        trainable=trainable)
    zn = tf.layers.batch_normalization(z, 
                                       axis=-1, 
                                       momentum=0.9, 
                                       epsilon=10**-8, 
                                       center=True, 
                                       scale=True, 
                                       beta_initializer=tf.zeros_initializer(), 
                                       gamma_initializer=tf.ones_initializer(), 
                                       moving_mean_initializer=tf.zeros_initializer(), 
                                       moving_variance_initializer=tf.ones_initializer(), 
                                       trainable=trainable)
    a = tf.nn.relu(zn)
    out = tf.layers.dropout(a, rate=1-keep_prob)
    return out

def affine_relu(x, units, kernel_initializer, bias_initializer, kernel_regularizer, keep_prob, trainable):
    z = tf.layers.dense(x, 
                        units, 
                        activation=None,
                        use_bias=True,
                        kernel_initializer=None,
                        bias_initializer=None,
                        kernel_regularizer=None,
                        trainable=trainable)
    a = tf.nn.relu(z)
    out = tf.layers.dropout(a, rate=1-keep_prob)
    return out

def affine_bn_softmax_loss(x, y, units, kernel_initializer, kernel_regularizer, trainable):
    z = tf.layers.dense(x, 
                        units, 
                        activation=None,
                        use_bias=False,
                        kernel_initializer=None,
                        kernel_regularizer=None,
                        trainable=trainable)
    zn = tf.layers.batch_normalization(z, 
                                       axis=-1, 
                                       momentum=0.9, 
                                       epsilon=10**-8, 
                                       center=True, 
                                       scale=True, 
                                       beta_initializer=tf.zeros_initializer(), 
                                       gamma_initializer=tf.ones_initializer(), 
                                       moving_mean_initializer=tf.zeros_initializer(), 
                                       moving_variance_initializer=tf.ones_initializer(), 
                                       trainable=trainable)
    probs = tf.nn.softmax(zn)
    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(probs), reduction_indices=1))
    return probs, loss

def affine_softmax_loss(x, y, units, kernel_initializer, bias_initializer, kernel_regularizer, trainable):
    z = tf.layers.dense(x, 
                        units, 
                        activation=None,
                        use_bias=True,
                        kernel_initializer=None,
                        bias_initializer=None,
                        kernel_regularizer=None,
                        trainable=trainable)
    probs = tf.nn.softmax(z)
    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(probs), reduction_indices=1))
    return probs, loss

def opti(func, loss, config=None):
    if func == "sgd":
        if config is None: config = {}
        config.setdefault("learning_rate", 0.001)
        opti_func = tf.train.GradientDescentOptimizer(learning_rate=config["learning_rate"])
    elif func == "adam":
        if config is None: config = {}
        config.setdefault("learning_rate", 0.001)
        config.setdefault("beta1", 0.9)
        config.setdefault("beta2", 0.999)
        config.setdefault("epsilon", 10**-8)
        opti_func = tf.train.AdamOptimizer(learning_rate=config["learning_rate"], 
                                           beta1=config["beta1"], 
                                           beta2=config["beta2"], 
                                           epsilon=config["epsilon"])
    else:
        raise ValueError("func should be in 'sgd' or 'adam'")
    opti_obj = opti_func.minimize(loss)
    return opti_obj

def metrics(probs, y):
    is_equal = tf.equal(tf.argmax(probs, axis=1), tf.argmax(y, axis=1)) # np.sum(probs == y)
    accu = tf.reduce_mean(tf.cast(is_equal, dtype=tf.float32)) # /len(y), tf.cast将boolean数组转成int数组
    return accu

def batch_func(x, y, batch_size):
    N = x.shape[0]
    batchs = N // batch_size
    for i in range(batchs):
        begin = batch_size * i
        end = batch_size + begin
        x_batch = x[begin:end]
        y_batch = y[begin:end]
        yield x_batch, y_batch

def img_batch_toarray(path, channel, height, width, data_format, y_label):
    assert isinstance(y_label, int), "y_label must astype int"
    os.chdir(path)
    if data_format == "channels_first": x = np.zeros((channel, height, width), dtype="float32")
    if data_format == "channels_last": x = np.zeros((height, width, channel), dtype="float32")
    x = np.expand_dims(x, axis=0)
    for filename in os.listdir():
#        filename = "1408154505_room_0.jpg"
        pic = Image.open(filename, mode="r")  # keras load style
        pic_array = image.img_to_array(pic, data_format).astype("float32") # data_format="channels_first"
        pic_array = np.expand_dims(pic_array, axis=0)
        x = np.concatenate((x, pic_array), axis=0)
    x = x[1:] # 删除第一个0数据
    y = np.tile(y_label, len(x)).astype("int32")
    data = {"target":x, "label":y}
    return data    
