# -*- coding: utf-8 -*-
"""
定义了一个三层的神经网络,输入是三种特征的加权和
这个网络的作用:用于加权特征融合方法实验程序
输入层--节点为50的全连接层--激活函数层--节点为4的全连接层--损失函数层

作者:殷和义
时间:2018年3月13日
邮箱:yinheyi@outlook.com
"""
import layer

class net:
	def __init__(self, batch_size, lr = 0.01, weights_decay = 0.05):
		layer.batch_size = batch_size
		layer.update_method.base_lr = lr
		layer.weights_decay = weights_decay 

		self.da = layer.data()
		self.da_test = layer.data()
		self.fc1 = layer.fully_connected_layer(256, 50)
		self.ac = layer.activation_layer('tanh')
		self.fc2 = layer.fully_connected_layer(50, 4)
		self.loss = layer.loss_layer('SoftmaxWithLoss')
	
	def load_sample_and_label(self, sample, label):
		self.da.get_data(sample, label)

	def load_sample_and_label_test(self, sample, label):
		self.da_test.get_data(sample, label)

	def pre_process(self):
		#对训练样本进行洗牌
		self.da.shuffle()

	def initial(self):
		#初始化权值
		self.fc1.initialize_weights()
		self.fc2.initialize_weights()

	def forward(self):
		self.da.pull_data()
		self.fc1.get_inputs_for_forward(self.da.output_sample)
		self.fc1.forward()
		self.ac.get_inputs_for_forward(self.fc1.outputs)
		self.ac.forward()
		self.fc2.get_inputs_for_forward(self.ac.outputs)
		self.fc2.forward()
		self.loss.get_inputs_for_loss(self.fc2.outputs)
		self.loss.get_label_for_loss(self.da.output_label)
		self.loss.compute_loss_and_accuracy()
	
	def turn_to_test(self, batch_size_test):
		layer.batch_size = batch_size_test
	def turn_to_train(self, batch_size_train):
		layer.batch_size = batch_size_train
	def forward_test(self):
		self.da_test.pull_data()
		self.fc1.get_inputs_for_forward(self.da_test.output_sample)
		self.fc1.forward()
		self.ac.get_inputs_for_forward(self.fc1.outputs)
		self.ac.forward()
		self.fc2.get_inputs_for_forward(self.ac.outputs)
		self.fc2.forward()
		self.loss.get_inputs_for_loss(self.fc2.outputs)
		self.loss.get_label_for_loss(self.da_test.output_label)
		self.loss.compute_loss_and_accuracy()

	def backward(self):
		self.loss.compute_gradient()
		self.fc2.get_inputs_for_backward(self.loss.grad_inputs)
		self.fc2.backward()
		self.ac.get_inputs_for_backward(self.fc2.grad_inputs)
		self.ac.backward()
		self.fc1.get_inputs_for_backward(self.ac.grad_inputs)
		self.fc1.backward()

	def update(self):
		self.fc1.update()
		self.fc2.update()
