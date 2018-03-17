# -*- coding: utf-8 -*-
"""
用于训练网络,很简单,就几行代码.

作者:殷和义
时间:2018年3月13日
邮箱:yinheyi@outlook.com
"""

import scipy.io
import random
import net1
import numpy as np
import matplotlib.pyplot as plt

# 导入数据;
data = scipy.io.loadmat('data.mat')

fft = data['fft_tr180']
power = data['power_tr180']
dps3 = data['dps3_tr180']
dps2 = data['dps2_tr180']
dps1 = data['dps1_tr180']
train_label = data['train_label']

fft_test = data['fft_tst180']
power_test = data['power_tst180']
dps3_test = data['dps3_tst180']
dps2_test = data['dps2_tst180']
dps1_test = data['dps1_tst180']
test_label = data['test_label']

#控制神经网络训练过程中的参数设置
num_train = 5000
test_interval = 20
train_batch_size = 50
test_batch_size = 250
lr = 0.1
weight_decay = 0.0001

# 创建网络并加载样本
solver = net1.net(train_batch_size, lr, weight_decay)
solver.load_sample_and_label(dps3, train_label)
solver.load_sample_and_label_test(dps3_test, test_label)

#对训练样本进行洗牌
solver.pre_process()

# 初始化权值;
solver.initial()

#初始化一些数组,用于保存需要的数据;
train_sequence = range(num_train)  #生成1-num_train的数组
test_sequence = range((num_train-1) // test_interval + 1)

train_error = np.zeros(num_train)
acc1 = np.zeros((num_train - 1)// test_interval + 1) 
acc2 = np.zeros((num_train - 1)// test_interval + 1) 
acc3 = np.zeros((num_train - 1)// test_interval + 1) 
acc4 = np.zeros((num_train - 1)// test_interval + 1) 

# 训练
for i in train_sequence:
	print '第', i, '次迭代'
	net1.layer.update_method.iteration  = i
	solver.forward()
	solver.backward()
	solver.update()

	#记录一些数值
	train_error[i] = solver.loss.loss
	if i % test_interval == 0:
		solver.turn_to_test(test_batch_size)

		solver.forward_test()
		acc1[i // test_interval] = solver.loss.accuracy
		solver.forward_test()
		acc2[i // test_interval] = solver.loss.accuracy
		solver.forward_test()
		acc3[i // test_interval] = solver.loss.accuracy
		solver.forward_test()
		acc4[i // test_interval] = solver.loss.accuracy

		solver.turn_to_train(train_batch_size)


plt.subplot(2, 1, 1)
plt.plot(train_sequence, train_error)
plt.subplot(2, 1, 2)
plt.plot(test_sequence, acc1, test_sequence, acc2, test_sequence, acc3, test_sequence, acc4)
plt.show()

print train_error[-1]
print acc1[-1], acc2[-1], acc3[-1], acc4[-1]

