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
solver = net.net(64, 0.01, 0.005)
solver.load_sample_and_label(fft, power, dps3, train_label)

# 初始化权值;
solver.initial()

#初始化一些数组,用于保存需要的数据;
train_error = np.zeros(500)
weight1 = np.zeros(500)
weight2 = np.zeros(500)

# 训练
for i in range(500):
	net.layer.update_method.iteration  = i
	solver.forward()
	solver.backward()
	solver.update()

	#记录一些数值
	weight1[i] = solver.fu.weights[0]
	weight2[i] = solver.fu.weights[1]
	train_error[i] = solver.loss.loss

print solver.loss.accuracy
plt.plot(range(500), weight1, range(500), weight2, range(500), train_error)
#plt.plot(train_error)
plt.show()
