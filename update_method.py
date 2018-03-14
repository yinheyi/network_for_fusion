# -*- coding: utf-8 -*-
"""
字义了权值的更新方法,其实它就对应了网络的训练方法吧.
我打算先定义一个标准的带动量项的的批量梯度下降法吧;

作者:殷和义
时间:2018年3月13日
邮箱:yinheyi@outlook.com
"""
momentum = 0.9
lr = 0.01
def batch_gradient_descent(weights, grad_weights, previous_direction):			
	direction = momentum * previous_direction + lr* grad_weights
	weights_now = weights - direction
	return (weights_now, direction)

