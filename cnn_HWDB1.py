# -*- coding: utf-8 -*-
from __future__ import absolute_import # 模块绝对引入
from __future__ import division # 精确除法
from __future__ import print_function
import os
os.chdir("D:/my_project/Python_Project/test/deeplearning")
os.getcwd()
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from scipy.stats import itemfreq
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

f = open("./HWDB1/char_dict", "rb")
char_dict = pickle.load(f); f.close()

words = list(char_dict.keys())
folders = list(char_dict.values())
folders_res = []
for folder in folders:
    if len(str(folder)) == 1:
        folder_new = "0000" + str(folder)
    elif len(str(folder)) == 2:
        folder_new = "000" + str(folder)
    elif len(str(folder)) == 3:
        folder_new = "00" + str(folder)
    else:
        folder_new = "0" + str(folder)
    folders_res.append(folder_new)

char_df = pd.DataFrame({"word":words, "folder":folders_res}, columns=["word","folder"])

char_set = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"

char_set = "的一是了我不人在他有这个上们"

img_size = 64
X = np.zeros((1, img_size, img_size), dtype="uint8")
X = np.expand_dims(X, axis=0)
Y = np.zeros(1, dtype="int32")
Z = []

flag = -1
for i in char_set:
    # i = "的"
    if i in char_df.word.values:
        flag += 1
        folder_name = char_df.iloc[np.where(i == char_df.word.values)[0][0], 1]
        os.chdir("D:/my_project/Python_Project/test/deeplearning/HWDB1/train/" + folder_name)
#        os.getcwd()
        images_list = os.listdir()
        Y_tmp = np.tile(flag, len(images_list))
        Y = np.concatenate((Y, Y_tmp))
        
        for filename in images_list:
            # filename = "724733.png"
            pic = Image.open(filename, mode="r")
            
            random_factor = rd.uniform(1.0, 1.3)
            pic_update = ImageEnhance.Brightness(pic).enhance(random_factor)
            random_factor = rd.uniform(0.8, 1.2)
            pic_update = ImageEnhance.Contrast(pic_update).enhance(random_factor)
            random_factor = rd.uniform(1.0, 3.0)
            pic_update = ImageEnhance.Sharpness(pic_update).enhance(random_factor)
            
            pic_resize = pic_update.resize((img_size, img_size), Image.BICUBIC)
            pic_array = image.img_to_array(pic_resize, "channels_first").astype("uint8")[2:]
            pic_array = np.expand_dims(pic_array, axis=0)
            X = np.concatenate((X, pic_array), axis=0)
            Z.append(i)
    else:
        continue
X = X[1:] # 删除第一个0数据
Y = Y[1:]
dataSet = {"target":X, "label":Y, "word":Z}

x = dataSet["target"]; y = dataSet["label"]; z = dataSet["word"]
x, y, z = shuffle(x, y, z, random_state=0)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, random_state=0)
#itemfreq(y_valid)

val_max = np.max(x_train)
x_train = (x_train/val_max).astype("float32").transpose(0,2,3,1)
x_valid = (x_valid/val_max).astype("float32").transpose(0,2,3,1)
y_train_ot = to_categorical(y_train)
y_valid_ot = to_categorical(y_valid)
out_num = len(set(y_train))

_, H, W, C = x_train.shape
x = tf.placeholder(tf.float32, shape=[None, H, W, C])
y = tf.placeholder(tf.float32, shape=[None, out_num])
keep_prob = tf.placeholder(tf.float32)

FS_1 = 32; F = 3; CS = 1; PK = 2; PS = 2
w1 = tf.Variable(tf.random_normal(shape=[F,F,C,FS_1], mean=0.0, stddev=0.01, name="w1"))
b1 = tf.Variable(tf.zeros(shape=[FS_1], name="b1"))
conv1 = tf.nn.conv2d(x, w1, strides=[1,CS,CS,1], padding="SAME") + b1
#bn1 = tf.layers.batch_normalization(conv1, axis=-1, momentum=0.9, epsilon=10**-8, center=True, scale=True, beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(), moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer())
active1 = tf.nn.relu(conv1)
dropout1 = tf.nn.dropout(active1, keep_prob)
pool1 = tf.nn.max_pool(dropout1, ksize=[1,PK,PK,1], strides=[1,PS,PS,1], padding="VALID")  

FS_2 = 32; F = 3; CS = 1; PK = 2; PS = 2
w2 = tf.Variable(tf.random_normal(shape=[F,F,int(pool1.shape[3]),FS_2], mean=0.0, stddev=0.01, name="w2"))
b2 = tf.Variable(tf.zeros(shape=[FS_2], name="b2"))
conv2 = tf.nn.conv2d(pool1, w2, strides=[1,CS,CS,1], padding="SAME") + b2
#bn2 = tf.layers.batch_normalization(conv2, axis=-1, momentum=0.9, epsilon=10**-8, center=True, scale=True, beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(), moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer())
active2 = tf.nn.relu(conv2)
dropout2 = tf.nn.dropout(active2, keep_prob)
pool2 = tf.nn.max_pool(dropout2, ksize=[1,PK,PK,1], strides=[1,PS,PS,1], padding="VALID")        

_, HP, WP, FP = pool2.shape
flatten = tf.reshape(pool2, shape=[-1,int(HP*WP*FP)])

unit3 = 64
w3 = tf.Variable(tf.random_normal(shape=[int(flatten.shape[1]),unit3], mean=0.0, stddev=0.01, name="w3"))
b3 = tf.Variable(tf.zeros(shape=[unit3], name="b3"))
dense3 = tf.matmul(flatten, w3) + b3
#dense4 = tf.matmul(flatten, w4)
#bn4 = tf.layers.batch_normalization(dense4, axis=-1, momentum=0.9, epsilon=10**-8, center=True, scale=True, beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(), moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer())
active3 = tf.nn.relu(dense3)
dropout3 = tf.nn.dropout(active3, keep_prob)

w4 = tf.Variable(tf.random_normal(shape=[int(dropout3.shape[1]),out_num], mean=0.0, stddev=0.01, name="w4"))
b4 = tf.Variable(tf.zeros(shape=[out_num], name="b4"))
dense4 = tf.matmul(dropout3, w4) + b4
#dense5 = tf.matmul(dropout4, w5)
#bn5 = tf.layers.batch_normalization(dense5, axis=-1, momentum=0.9, epsilon=10**-8, center=True, scale=True, beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(), moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer())
probs = tf.nn.softmax(dense4)

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
# 1
opti_func = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=10**-8)
opti_obj = opti_func.minimize(loss, global_step)
# 2
opti_func = tf.train.AdamOptimizer(learning_init, beta1=0.9, beta2=0.999, epsilon=10**-8)
opti_obj = opti_func.minimize(loss)

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
with tf.Session() as sess:
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
#        lr = sess.run(learning_rate)
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
#        print("learning_rate: %g" % lr)
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
