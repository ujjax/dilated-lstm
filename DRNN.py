from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np 



def dilation_layer(cell,inputs,rate, batch_size, state):
	n_steps = len(inputs)
	if rate < 0 or rate >= n_steps:
		raise ValueError('The rate variable needs to be adjusted.')

	if not (n_steps % rate) == 0:
		zero_tensor = torch.zeros_like(inputs[0])

		dilated_n_steps = n_steps // rate + 1
		print("Input length for sub-RNN: %d" % (dilated_n_steps))

		for i_pad in xrange(dilated_n_steps * rate - n_steps):
			inputs.append(zero_tensor)

	else:
		dilated_n_steps = n_steps // rate
		print("Input length for sub-RNN: %d" % (dilated_n_steps))

	dilated_inputs = [torch.cat(torch.split(inputs[i*rate : (i + 1)*rate],rate,dim=0),dim=0) for i in range(dilated_n_steps)]

	dilated_outputs = []


	for i in range(dilated_n_steps):
		print(dilated_inputs[i].size())
		dilated_output , state = cell(dilated_inputs[i],state)
		dilated_outputs.append(dilated_output)

	splitted_outputs = [torch.split(output, rate, dim=0)
					for output in dilated_outputs]

	unrolled_outputs = [output
					for sublist in splitted_outputs for output in sublist]
	# remove padded zeros
	outputs = unrolled_outputs[:n_steps]

	return outputs, state


class Model(nn.Module):
	def __init__(self, hidden_structure, dilations, num_steps, n_classes, batch_size, cell_type):
		super(Model, self).__init__()
		self.n_classes = n_classes
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.dilations = dilations
		self.hidden_structure = hidden_structure
		self.cell_type = cell_type
		
		self.cells = []


		for hidden_dim in self.hidden_structure:
			if self.cell_type == 'RNN':
				cell = nn.RNNCell(self.embedding_dim, hidden_dim)
			if self.cell_type == 'LSTM':
				cell = nn.LSTMCell(self.embedding_dim, hidden_dim)
			if self.cell_type == 'GRU':
				cell = nn.GRUCell(self.embedding_dim, hidden_dim)

			self.cells.append(cell)

	def forward(self, x, state,W,b):
		assert (len(self.hidden_structure) == len(self.dilations))	
		assert (len(self.cells)==len(self.dilations))

		for cell, rate in zip(self.cells, self.dilations):
			x, state = dilation_layer(cell, x, rate, self.batch_size, state)
			
			
		if self.dilations[0] ==1:
			#W = Variable(torch.randn(self.hidden_structure[-1], self.n_classes))
			#b = Variable(torch.randn(self.n_classes))

			x = torch.matmul(x[-1], W) + b

		else:
			#W = Variable(torch.randn(self.hidden_structure[-1] * self.dilations[0], self.n_classes))
			#b = Variable(torch.randn(self.n_classes))

			for idx, i in enumerate(range(-self.dilations[0], 0, 1)):
				if idx == 0:
					hidden_outputs_ = x[i]
				else:
					hidden_outputs_ = torch.cat(
						[hidden_outputs_, x[i]],
						axis=1)
			x = torch.matmul(hidden_outputs_, W) + b
		
		return x

	def initHidden(self):
		hx = Variable(torch.randn(self.batch_size, self.hidden_structure[0]))
		cx = Variable(torch.randn(self.batch_size, self.hidden_structure[0]))

		state = (cx.cuda(),hx.cuda())

		W = Variable(torch.randn(self.hidden_structure[-1] * self.dilations[0], self.n_classes))
		b = Variable(torch.randn(self.n_classes))



		return state , W.cuda(), b.cuda()
