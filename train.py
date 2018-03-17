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
dps2 = data['dps2_tr180']
dps1 = data['dps1_tr180']
train_label = data['train_label']

fft_test = data['fft_tst180']
power_test = data['power_tst180']
dps3_test = data['dps3_tst180']
dps2_test = data['dps2_tst180']
dps1_test = data['dps1_tst180']
test_label = data['test_label']
#  对训练数据进行洗牌,注意:一定要把三种特征及标签的顺序保持一致;
random_seed = random.sample(np.arange(200), 200)
fft = fft[random_seed]
power = power[random_seed]
dps3 = dps3[random_seed]
train_label = train_label[random_seed]

#一些相关的重要参数
num_train = 2000
test_interval = 20
lr = 0.1
weight_decay = 0.001
train_batch_size = 50
test_batch_size = 250

# 创建网络并加载样本
solver = net.net(train_batch_size, lr, weight_decay)
solver.load_sample_and_label(fft, power, dps3,train_label)
solver.load_sample_and_label_test(fft_test, power_test, dps3_test, test_label)

# 初始化权值;
solver.initial()


#初始化一些数组,用于保存需要的数据;
train_sequence = range(num_train)  #生成1-num_train的数组
test_sequence = range(num_train // test_interval)

train_error = np.zeros(num_train)
weight1 = np.zeros(num_train)
weight2 = np.zeros(num_train)
acc1 = np.zeros((num_train - 1)// test_interval + 1)
acc2 = np.zeros((num_train - 1)// test_interval + 1)
acc3 = np.zeros((num_train - 1)// test_interval + 1)
acc4 = np.zeros((num_train - 1)// test_interval + 1)

# 训练
for i in train_sequence:
	print '第', i, '次迭代'
	net.layer.update_method.iteration  = i
	solver.forward()
	solver.backward()
	solver.update()

	#记录一些数值
	weight1[i] = solver.fu.weights[0]
	weight2[i] = solver.fu.weights[1]
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
plt.plot(train_sequence, weight1, train_sequence, weight2, 
		train_sequence, np.ones(num_train) - weight1 - weight2, 
		train_sequence, train_error)
plt.subplot(2, 1, 2)
plt.plot(test_sequence, acc1, test_sequence, acc2, test_sequence, acc3, test_sequence, acc4)
plt.show()

print acc1[-1], acc2[-1], acc3[-1], acc4[-1]

