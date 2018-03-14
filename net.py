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

class Net:
	def __init__(self):
		self.data = layer.fusion_layer(256)
		self.fc1 = layer.fully_connected_layer(256, 50)
		self.ac = layer.activation_layer(50, 'tanh')
		self.fc2 = layer.fully_connected_layer(50, 4)
		self.loss = layer.loss_layer(4, 'SoftmaxWithLoss')
	
	def load_sample_and_label(self, sample1, sample2, sample3, label):
		self.data.get_inputs_for_forward(sample1, sample2, sample3)
		self.loss.get_label_for_loss(label)

	def initial(self):
		self.data.initialize_weights(0.3, 0.3)
		self.fc1.initialize_weights()
		self.fc2.initialize_weights()

	def forward(self):
		self.data.forward()
		self.fc1.get_inputs_for_forward(self.data.outputs)
		self.fc1.forward()
		self.ac.get_inputs_for_forward(self.fc1.outputs)
		self.ac.forward()
		self.fc2.get_inputs_for_forward(self.ac.outputs)
		self.fc2.forward()
		self.loss.get_inputs_for_loss(self.fc2.outputs)
		self.loss.compute_loss_and_accuracy()
	
	def backward(self):
		self.loss.compute_gradient()
		self.fc2.get_inputs_for_backward(self.loss.grad_outputs_previous)
		self.fc2.backward()
		self.ac.get_inputs_for_backward(self.fc2.grad_outputs_previous)
		self.ac.backward()
		self.fc1.get_inputs_for_backward(self.ac.grad_outputs_previous)
		self.fc1.backward()
		self.data.get_inputs_for_backward(self.fc1.grad_outputs_previous)
		self.data.backward()

	def update(self):
		self.data.update()
		self.fc1.update()
		self.fc2.update()


