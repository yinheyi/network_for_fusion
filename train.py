# -*- coding: utf-8 -*-
"""
用于训练网络,很简单,就几行代码.

作者:殷和义
时间:2018年3月13日
邮箱:yinheyi@outlook.com
"""

import scipy.io
import random
import net
import numpy as np
import matplotlib.pyplot as plt

# 导入数据;
data = scipy.io.loadmat('data.mat')
fft = data['fft_tr180']
power = data['power_tr180']
dps3 = data['dps3_tr180']
train_label = data['train_label']
#  对数据进行洗牌,注意:一定要把三种特征及标签的顺序保持一致;
random_seed = random.sample(np.arange(200), 200)
fft = fft[random_seed]
power = power[random_seed]
dps3 = dps3[random_seed]
train_label = train_label[random_seed]


# 创建网络并加载样本
solver = net.net(64, 0.005, 0.01)
solver.load_sample_and_label(fft, power, dps3, train_label)

# 初始化权值;
solver.initial()


#初始化一些数组,用于保存需要的数据;
num_train = 1000
train_error = np.zeros(num_train)
weight1 = np.zeros(num_train)
weight2 = np.zeros(num_train)

# 训练
for i in range(num_train):
	net.layer.update_method.iteration  = i
	solver.forward()
	solver.backward()
	solver.update()

	#记录一些数值
	weight1[i] = solver.fu.weights[0]
	weight2[i] = solver.fu.weights[1]
	train_error[i] = solver.loss.loss


# 测试
net.layer.batch_size = 250
fft = data['fft_tst180']
power = data['power_tst180']
dps3 = data['dps3_tst180']
test_label = data['test_label']

solver.load_sample_and_label(fft, power, dps3, test_label)
for i in range(4):
	solver.forward()
	print solver.loss.accuracy

