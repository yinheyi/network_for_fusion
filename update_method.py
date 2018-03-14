# -*- coding: utf-8 -*-
"""
1. 定义了在网络训练过程中,学习率的变化机制;
2. 定义了权值的更新方法,其实它就对应了网络的训练方法吧.

另外说明一个需要让你不迷惑的概念:
1个 epoch 表示一个完整的数据集通过网络一次;
1个iteration 表示网络训练一次;  当使用batch_size时,1个epoch需要多个iteration 才能完成.


作者:殷和义
时间:2018年3月13日
邮箱:yinheyi@outlook.com
"""

import numpy as np

#定义一些需要的全局变量
momentum = 0.9
base_lr  = 0         # 在建造net是对它初始化;
iteration = -1       # 它常常需要在训练过程中修改


###########################      定义学习率的变化机制函数     ####################################

# inv方法  (它是什么意思, 我也不知道,反正caffe里有这个方法)
def inv(gamma = 0.0005, power = 0.75):
	if iteration == -1:
		assert False, '需要在训练过程中,改变update_method 模块里的 iteration 的值'
	return base_lr * np.power((1 + gamma * iteration), -power) 

# 固定方法
def fixed():
	return base_lr



###########################       定义更新方法函数              ####################################

# 基于批量的随机梯度下降法
def batch_gradient_descent(weights, grad_weights, previous_direction):			
	lr = inv()
	direction = momentum * previous_direction + lr * grad_weights
	weights_now = weights - direction
	return (weights_now, direction)

