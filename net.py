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

		self.da1 = layer.data()
		self.da2 = layer.data()
		self.da3 = layer.data()
		self.fu = layer.fusion_layer(256)
		self.fc1 = layer.fully_connected_layer(256, 50)
		self.ac = layer.activation_layer('tanh')
		self.fc2 = layer.fully_connected_layer(50, 4)
		self.loss = layer.loss_layer('SoftmaxWithLoss')
	
	def load_sample_and_label(self, sample1, sample2, sample3, label):
		self.da1.get_data(sample1, label)
		self.da2.get_data(sample2, label)
		self.da3.get_data(sample3, label)

	def initial(self):
		self.fu.initialize_weights(0.3, 0.3)
		self.fc1.initialize_weights()
		self.fc2.initialize_weights()

	def forward(self):
		self.da1.pull_data()
		self.da2.pull_data()
		self.da3.pull_data()
		self.fu.get_inputs_for_forward(self.da1.output_sample, self.da2.output_sample, self.da3.output_sample)
		self.fu.forward()
		self.fc1.get_inputs_for_forward(self.fu.outputs)
		self.fc1.forward()
		self.ac.get_inputs_for_forward(self.fc1.outputs)
		self.ac.forward()
		self.fc2.get_inputs_for_forward(self.ac.outputs)
		self.fc2.forward()
		self.loss.get_inputs_for_loss(self.fc2.outputs)
		self.loss.get_label_for_loss(self.da1.output_label)
		self.loss.compute_loss_and_accuracy()
	
	def backward(self):
		self.loss.compute_gradient()
		self.fc2.get_inputs_for_backward(self.loss.grad_inputs)
		self.fc2.backward()
		self.ac.get_inputs_for_backward(self.fc2.grad_inputs)
		self.ac.backward()
		self.fc1.get_inputs_for_backward(self.ac.grad_inputs)
		self.fc1.backward()
		self.fu.get_inputs_for_backward(self.fc1.grad_inputs)
		self.fu.backward()

	def update(self):
		self.fu.update()
		self.fc1.update()
		self.fc2.update()


