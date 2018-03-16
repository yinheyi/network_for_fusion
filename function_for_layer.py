# -*- coding: utf-8 -*-
"""
 1. 里面定义了一系列的常用的激活函数,及它们的导数;详细可以看一下说明文档;
 2. 里面定义了损失函数;
 3. 里面定义了权值初始化的函数;

 作者: 殷和义
 时间:218年03月12日
 邮箱: yinheyi@outlook.com
"""
import numpy as np 
from scipy import stats

######################################       激活函数的定义           #######################################################

# sigmoid函数及其导数的定义
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
def der_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))


# tanh函数及其导数的定义
def tanh(x):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def der_tanh(x):
	return 1 - tanh(x) * tanh(x)

# ReLU函数及其导数的定义
def relu(x):
	temp = np.zeros_like(x)
	if_bigger_zero = (x > temp)
	return x * if_bigger_zero
def der_relu(x):
	temp = np.zeros_like(x)
	if_bigger_equal_zero = (x >= temp)          #在零处的导数设为1
	return if_bigger_equal_zero * np.ones_like(x)


######################################       损失函数的定义           #######################################################

# SoftmaxWithLoss函数及其导数的定义
def softmaxwithloss(inputs, label):
	temp1 = np.exp(inputs)
	probability = temp1 / (np.tile(np.sum(temp1, 1), (inputs.shape[1], 1))).T
	temp3 = np.argmax(label, 1)   #纵坐标
	temp4 = [probability[i, j] for (i, j) in zip(np.arange(label.shape[0]), temp3)]
	loss = -1 * np.mean(np.log(temp4))
	return loss
def der_softmaxwithloss(inputs, label):
	temp1 = np.exp(inputs)
	temp2 = np.sum(temp1, 1)  #它得到的是一维的向量;
	probability = temp1 / (np.tile(temp2, (inputs.shape[1], 1))).T
	gradient = probability - label
	return gradient


######################################      权值初始化方法相关函数的定义  ############################################

# xavier 初始化方法
def xavier(num_neuron_inputs, num_neuron_outputs):
	temp1 =  np.sqrt(6) / np.sqrt(num_neuron_inputs+ num_neuron_outputs + 1)
	weights = stats.uniform.rvs(-temp1, 2 * temp1, (num_neuron_inputs, num_neuron_outputs))
	return weights

