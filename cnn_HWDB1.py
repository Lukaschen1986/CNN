# -*- coding: utf-8 -*-
from __future__ import absolute_import # 模块绝对引入
from __future__ import division # 精确除法
from __future__ import print_function
import os
os.chdir("D:/my_project/Python_Project/test/deeplearning")
os.getcwd()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from scipy.stats import itemfreq
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from PIL import Image, ImageEnhance, ImageOps, ImageFile
#import cnn_layers_tf as clt

#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
#from keras.layers.normalization import BatchNormalization
#from keras.layers.core import Activation
#from keras import backend as K
#K.image_data_format()
#K.set_image_data_format('channels_first')
from keras.preprocessing import image
#from keras.utils.np_utils import to_categorical
#from keras import initializers
#from keras.regularizers import l1, l2
#from keras.optimizers import SGD, Adam
#from keras.utils import plot_model
#from keras.models import load_model

f = open("./HWDB1/char_dict", "rb")
char_dict = pickle.load(f); f.close()

char_dict.keys()
char_dict.values()

words = list(char_dict.keys())[0:10]
folders = list(char_dict.values())[0:10]
folders_res = []
for folder in folders:
    # folder = 1126
    if len(str(folder)) == 1:
        folder_new = "0000" + str(folder)
    elif len(str(folder)) == 2:
        folder_new = "000" + str(folder)
    elif len(str(folder)) == 3:
        folder_new = "00" + str(folder)
    else:
        folder_new = "0" + str(folder)
    folders_res.append(folder_new)

path = "D:/my_project/Python_Project/test/deeplearning/HWDB1/demo"
os.chdir(path)
for folder in os.listdir():
    # folder = "00121"
    os.chdir(path + "/" + folder)
    for filename in os.listdir(): 
        pic_bright(filename, range_from=0.5, range_to=1.0)
        pic_contrast(filename, range_from=0.5, range_to=5.0)
        pic_sharpness(filename, range_from=0.1, range_to=3.0)

def pic_resize(filename, height, width):
    pic = Image.open(filename, mode="r") # 载入图片
    pic_update = pic.resize((height, width), Image.BICUBIC)
    return pic_update.save(filename) # 保存

path = "D:/my_project/Python_Project/test/deeplearning/HWDB1/demo"
os.chdir(path)
for folder in os.listdir():
    os.chdir(path + "/" + folder)
    for filename in os.listdir(): 
        pic_resize(filename, height=64, width=64)
    
path = "D:/my_project/Python_Project/test/deeplearning/HWDB1/demo"
#channel=1; height=32; width=32; data_format="channels_first"
def pic_to_array(path, channel, height, width, data_format):
    os.chdir(path)
    if data_format == "channels_first": x = np.zeros((channel, height, width), dtype="float32")
    if data_format == "channels_last": x = np.zeros((height, width, channel), dtype="float32")
    x = np.expand_dims(x, axis=0)
    y = np.zeros(1, dtype="int32")
    flag = -1
    folder_list = os.listdir()
    for folder in folder_list:
        # folder = "00121"
        flag += 1
        os.chdir(path + "/" + folder)
        file_list = os.listdir()
        for filename in file_list:
#            filename = "247845.png"
            pic = Image.open(filename, mode="r")  # keras load style
            pic_array = image.img_to_array(pic, data_format).astype("float32") # data_format="channels_first"
#            pic_array = pic_array[2:] # 提取单通道
#            pic_array = pic_array[::2] # 提取单通道
            pic_array = np.expand_dims(pic_array, axis=0)
            x = np.concatenate((x, pic_array), axis=0)
        y_batch = np.tile(flag, len(file_list))
        y = np.concatenate((y, y_batch))
    x = x[1:] # 删除第一个0数据
    y = y[1:]
    dataSet = {"target":x, "label":y}
    return dataSet
dataSet = pic_to_array(path="D:/my_project/Python_Project/test/deeplearning/HWDB1/demo", 
                       channel=3, height=64, width=64, 
                       data_format="channels_last")
x = dataSet["target"]; y = dataSet["label"]
x, y = shuffle(x, y, random_state=0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)
itemfreq(y_test)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
itemfreq(y_valid)

val_max = np.max(x_train)
#x_train = tf.convert_to_tensor((x_train/val_max).astype("float32"))
#x_valid = tf.convert_to_tensor((x_valid/val_max).astype("float32"))
#x_test = tf.convert_to_tensor((x_test/val_max).astype("float32"))
x_train = (x_train/val_max).astype("float32")
x_valid = (x_valid/val_max).astype("float32")
x_test = (x_test/val_max).astype("float32")

out_num = len(set(y_train))
#y_train_ot = tf.one_hot(y_train, out_num)
#y_valid_ot = tf.one_hot(y_valid, out_num)
#y_test_ot = tf.one_hot(y_test, out_num)
one_hot = lambda y: np.eye(len(set(y)), dtype=np.int32)[y]
y_train_ot = one_hot(y_train)
y_valid_ot = one_hot(y_valid)
y_test_ot = one_hot(y_test)

################################################
_, H, W, C = x_train.shape
x = tf.placeholder(tf.float32, shape=[None, H, W, C])
y = tf.placeholder(tf.float32, shape=[None, out_num])
keep_prob = tf.placeholder(tf.float32)
#l2_lam = tf.placeholder(tf.float32)

filter1 = 32; fsize1 = 3
w1 = tf.Variable(tf.random_normal(shape=[fsize1,fsize1,C,filter1], mean=0.0, stddev=0.01, name="w1"))
'''
random_normal, truncated_normal, random_uniform
'''
#b1 = tf.Variable(tf.zeros(shape=[filter1], name="b1"))
#conv1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding="SAME") + b1
conv1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding="SAME")
bn1 = tf.layers.batch_normalization(conv1, axis=-1, momentum=0.9, epsilon=10**-8, center=True, scale=True, beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(), moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer())
active1 = tf.nn.relu(bn1)
dropout1 = tf.nn.dropout(active1, keep_prob)
pool1 = tf.nn.max_pool(dropout1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

filter2 = 32; fsize2 = 3
w2 = tf.Variable(tf.random_normal(shape=[fsize2,fsize2,int(pool1.shape[3]),filter2], mean=0.0, stddev=0.01, name="w2"))
#b2 = tf.Variable(tf.zeros(shape=[filter2], name="b2"))
#conv2 = tf.nn.conv2d(pool1, w2, strides=[1,1,1,1], padding="SAME") + b2
conv2 = tf.nn.conv2d(pool1, w2, strides=[1,1,1,1], padding="SAME")
bn2 = tf.layers.batch_normalization(conv2, axis=-1, momentum=0.9, epsilon=10**-8, center=True, scale=True, beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(), moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer())
active2 = tf.nn.relu(bn2)
dropout2 = tf.nn.dropout(active2, keep_prob)
pool2 = tf.nn.max_pool(dropout2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

filter3 = 64; fsize3 = 3
w3 = tf.Variable(tf.random_normal(shape=[fsize3,fsize3,int(pool2.shape[3]),filter3], mean=0.0, stddev=0.01, name="w3"))
#b3 = tf.Variable(tf.zeros(shape=[filter3], name="b3"))
#conv3 = tf.nn.conv2d(pool2, w3, strides=[1,1,1,1], padding="SAME") + b3
conv3 = tf.nn.conv2d(pool2, w3, strides=[1,1,1,1], padding="SAME")
bn3 = tf.layers.batch_normalization(conv3, axis=-1, momentum=0.9, epsilon=10**-8, center=True, scale=True, beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(), moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer())
active3 = tf.nn.relu(bn3)
dropout3 = tf.nn.dropout(active3, keep_prob)
pool3 = tf.nn.max_pool(dropout3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

_, HP, WP, FP = pool3.shape
flatten = tf.reshape(pool3, shape=[-1,int(HP*WP*FP)])

unit4 = 64
w4 = tf.Variable(tf.random_normal(shape=[int(flatten.shape[1]),unit4], mean=0.0, stddev=0.01, name="w4"))
#b4 = tf.Variable(tf.zeros(shape=[unit4], name="b4"))
#dense4 = tf.matmul(flatten, w4) + b4
dense4 = tf.matmul(flatten, w4)
bn4 = tf.layers.batch_normalization(dense4, axis=-1, momentum=0.9, epsilon=10**-8, center=True, scale=True, beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(), moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer())
active4 = tf.nn.relu(bn4)
dropout4 = tf.nn.dropout(active4, keep_prob)

w5 = tf.Variable(tf.random_normal(shape=[int(dropout4.shape[1]),out_num], mean=0.0, stddev=0.01, name="w5"))
#b5 = tf.Variable(tf.zeros(shape=[out_num], name="b5"))
#dense5 = tf.matmul(dropout4, w5) + b5
dense5 = tf.matmul(dropout4, w5)
bn5 = tf.layers.batch_normalization(dense5, axis=-1, momentum=0.9, epsilon=10**-8, center=True, scale=True, beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(), moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer())
probs = tf.nn.softmax(bn5)

'''
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)  
'''
epochs = 10
learning_init = 0.001
#global_step = tf.placeholder(tf.int32, shape=1)
global_step = tf.Variable(0)
decay_steps = epochs
decay_rate = 0.04
learning_rate = tf.train.exponential_decay(learning_init, global_step, decay_steps, decay_rate, staircase=False)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(probs+10**-8), reduction_indices=1))
opti_func = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=10**-8)
opti_obj = opti_func.minimize(loss, global_step)
is_equal = tf.equal(tf.argmax(probs, axis=1), tf.argmax(y, axis=1)) # np.sum(probs == y)
accuracy = tf.reduce_mean(tf.cast(is_equal, dtype=tf.float32)) # /len(y), tf.cast将boolean数组转成int数组

def batch_func(x, y, batch_size):
    N = x.shape[0]
    batchs = N // batch_size
    for i in range(batchs):
        begin = batch_size * i
        end = batch_size + begin
        x_batch = x[begin:end]
        y_batch = y[begin:end]
        yield x_batch, y_batch


init = tf.global_variables_initializer()
with tf.session() as sess:
    sess.run(init)
    saver = tf.train.Saver() # 生成saver
    
    loss_train_global = []
    accu_train_global = []
    loss_valid_global = []
    accu_valid_global = []
    batch_size = 4
#    x_train.shape[0] // batch_size
    for epoch in range(epochs):
        loss_train_epoch = 0.0
        accu_train_epoch = 0.0
        loss_valid_epoch = 0.0
        accu_valid_epoch = 0.0
        # train
        batch_sample = 0.0
        for x_batch, y_batch in batch_func(x_train, y_train_ot, batch_size):
            batch_sample += 1
            _, loss_train, accu_train = sess.run([opti_obj, loss, accuracy], 
                                                 feed_dict={x:x_batch, y:y_batch, keep_prob:1.0})
            loss_train_epoch += loss_train
            accu_train_epoch += accu_train
        loss_train_epoch /= batch_sample
        accu_train_epoch /= batch_sample
        # lr
        lr = sess.run(learning_rate)
        # valid
        loss_valid_epoch, accu_valid_epoch = sess.run([loss, accuracy],
                                                      feed_dict={x:x_valid, y:y_valid_ot, keep_prob:1.0})
        # append
        loss_train_global.append(loss_train_epoch)
        accu_train_global.append(accu_train_epoch)
        loss_valid_global.append(loss_valid_epoch)
        accu_valid_global.append(accu_valid_epoch)
        print("epoch:%d, loss_train:%g, accu_train:%g, loss_valid:%g, accu_valid:%g" % \
              (epoch, loss_train_epoch, accu_train_epoch, loss_valid_epoch, accu_valid_epoch))
        print("learning_rate: %g" % lr)
        # early_stopping
        if loss_train_epoch < 0.1:
            break
        
    probs_test, loss_test, accu_test = sess.run([probs,loss,accuracy], feed_dict={x:x_test, y:y_test_ot, keep_prob:1.0})
    print("loss_test: %g, accu_test: %g" % (loss_test, accu_test))

    df = pd.DataFrame({"loss_train": loss_train_global,
                       "loss_valid": loss_valid_global,
                       "loss_test": loss_test})
    df.plot()
    saver.save(sess, save_path="./model", global_step=0, write_meta_graph=True, write_state=True)

# load
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
path = tf.train.get_checkpoint_state("./")
saver.restore(sess, path.model_checkpoint_path)

probs_test, loss_test, accu_test = sess.run([probs,loss,accuracy], feed_dict={x:x_test, y:y_test, keep_prob:1.0})
y_pred = np.argmax(probs_test, axis=1)
y_test_new = np.argmax(y_test, axis=1)
pd.crosstab(index=y_test_new, columns=y_pred, margins=True)
