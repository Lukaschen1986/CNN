# -*- coding: utf-8 -*-

data = img_batch_toarray(path="D:\\file\\Py_project\\CNN\\test", 
                         channel=3, height=224, width=224, 
                         data_format="channels_first", y_label=0)
x = data["target"]
y = data["label"]
val_max = np.max(x)
x = (x/val_max).astype("float32")
'''
x.shape # (2, 3, 224, 224)
'''

#1 Conv
w1, b1 = filter_init(x, F=4, HH=5, WW=5)
stride = 1
padding = get_padding(x, w1, stride)
conv_param1 = {"S":stride, "P":padding}
_, C, _, _, = conv_out1.shape
gamma1 = np.ones((1, C), dtype="float32")
beta1 = np.zeros((1, C), dtype="float32")
running_mean1 = running_var1 = np.zeros((1, C), dtype="float32")
bn_param1 = {"mode":"train", "momentum":0.9, "running_mean":running_mean1, "running_var":running_var1}
pool_param1 = {"S":2, "HP":2, "WP":2}

conv_out1, conv_cache1 = conv_forward(x, w1, b1, conv_param1)

def conv_bn_relu_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
#    x=x; w=w1; b=b1; gamma=gamma1; beta=beta1; conv_param=conv_param1; bn_param=bn_param1; pool_param=pool_param1
    a, conv_cache = conv_forward(x, w, b, conv_param)
    an, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    s, relu_cache = relu_forward(an)
    out, pool_cache = maxPooling_forward(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache
out.shape (2, 4, 112, 112)

w2, b2 = affine_init(x_flatten, units=512)
_, D = a.shape
gamma2 = np.ones((1, D), dtype="float32")
beta2 = np.zeros((1, D), dtype="float32")
running_mean2 = running_var2 = np.zeros((1, D), dtype="float32")
bn_param2 = {"mode":"train", "momentum":0.9, "running_mean":running_mean2, "running_var":running_var2}
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    x=out; w=w2; b=b2; gamma=gamma2; beta=beta2; bn_param=bn_param2
    x_flatten = flatten(x)
    a, fc_cache = affine_forward(x_flatten, w, b)
    a_bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache
#2 BN
#3 Relu
#4 Max
#5 Flatten
#6 FC
#7 Relu
#8 FC
#9 Softmax
#10 y_hat
