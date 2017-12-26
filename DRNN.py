from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy


class DRNNCell(object):
	"""docstring for DRNNCell"""
	def __init__(self):
		super(DRNNCell, self).__init__()
		self.n_classes = n_classes
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.vocab_size = vocab_size

		if cell_type not in ["RNN", "LSTM", "GRU"]:
        	raise ValueError("The cell type is not currently supported.") 
		
		self.cells = []
		for hidden_dim in hidden_structure:
			if cell_type = 'RNN':
				cell = nn.RNNCell(self.embedding_dim, hidden_dim)
			if cell_type = 'LSTM':
				cell = nn.LSTMCell(self.embedding_dim, hidden_dim)
			if cell_type = 'GRU':
				cell = nn.GRUCell(self.embedding_dim, hidden_dim)

			self.cells.append(cell)

		assert (len(hidden_structs) == len(dilations))

		embeddings = nn.Embedding(vocab_size,self.embedding_dim)

	def _dilation_layer(self,cell,inputs,rate):
		n_steps = len(inputs)
		if rate < 0 or rate >= n_steps:
        	raise ValueError('The rate variable needs to be adjusted.')

        if not (n_steps % rate) == 0:
        	zero_tensor = torch.zeros_like(inputs[0])

        	dilated_n_steps = n_steps // rate + 1
	        print "=====> %d time points need to be padded. " % (
	            dilated_n_steps * rate - n_steps)
	        print "=====> Input length for sub-RNN: %d" % (dilated_n_steps)
	        for i_pad in xrange(dilated_n_steps * rate - n_steps):
	            inputs.append(zero_tensor)

	    else:
	        dilated_n_steps = n_steps // rate
	        print "=====> Input length for sub-RNN: %d" % (dilated_n_steps)

	    dilated_inputs = [torch.cat(inputs[i*rate : (i + 1)*rate],
                                dim=0) for i in range(dilated_n_steps)]

	    dilated_outputs = []

	    hx = Variable(torch.randn(self.batch_size, self.hidden_dim))
		cx = Variable(torch.randn(self.batch_size, self.hidden_dim))

		state = (cx,hx)

	    for i in range(dilated_n_steps):
	    	dilated_output , state = cell(dilated_inputs[i],state)
	    	dilated_outputs.append(dilated_output)

	    splitted_outputs = [torch.split(output, rate, dim=0)
                        for output in dilated_outputs]

	    unrolled_outputs = [output
                        for sublist in splitted_outputs for output in sublist]
	    # remove padded zeros
	    outputs = unrolled_outputs[:n_steps]

	    return outputs

	def forward(self, inputs,dilations):
		assert (len(self.cells)==len(dilations))

		x = copy.copy(inputs)

		for cell, dilation in zip(cells, dilations):
		    x = self._dilation_layer(cell, x, dilation)
		
		layer_outputs = x

		if dilations[0] ==1:
			W = Variable(torch.randn(hidden_structs[-1],self.n_classes))
			b = Variable(torch.randn(self.n_classes))

			pred = torch.matmul(layer_outputs[-1], weights) + bias

		else:
			W = Variable(torch.randn(hidden_structs[-1] * dilations[0], self.n_classes))
			b = Variable(torch.randn(self.n_classes))

			for idx, i in enumerate(range(-dilations[0], 0, 1)):
	            if idx == 0:
	                hidden_outputs_ = layer_outputs[i]
	            else:
	                hidden_outputs_ = torch.cat(
	                    [hidden_outputs_, layer_outputs[i]],
	                    axis=1)
	        pred = torch.matmul(hidden_outputs_, weights) + bias

	   	return pred





