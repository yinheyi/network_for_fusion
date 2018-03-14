# -*- coding: utf-8 -*-
"""
用于训练网络,很简单,就几行代码.

作者:殷和义
时间:2018年3月13日
邮箱:yinheyi@outlook.com
"""

import scipy.io
import net
import numpy as np
import matplotlib.pyplot as plt

# 导入数据;
data = scipy.io.loadmat('data.mat')

# 创建网络并加载样本
solver = net.Net(200, 0.01, 0.2)
solver.load_sample_and_label(data['fft_tr180'], data['dps3_tr180'], data['power_tr180'], data['train_label'])

# 初始化权值;
solver.initial()

#初始化一些数组,用于保存需要的数据;
train_error = np.zeros(500)
weight1 = np.zeros(500)
weight2 = np.zeros(500)

# 训练
for i in range(500):
	solver.forward()
	solver.backward()
	solver.update()

	#记录一些数值
	weight1[i] = solver.data.weights[0]
	weight2[i] = solver.data.weights[1]
	train_error[i] = solver.loss.loss

print solver.fc1.weights
plt.plot(range(500), weight1, range(500), weight2)
#plt.plot(train_error)
plt.show()
