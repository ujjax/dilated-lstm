from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from DRNN import Model

import numpy as np

num_steps = 28*28
input_dims = 1
n_classes = 10 

# model config
cell_type = "RNN"
assert(cell_type in ["RNN", "LSTM", "GRU"])
hidden_structs = [20] * 9
dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256]
assert(len(hidden_structs) == len(dilations))



# learning config
batch_size = 128
learning_rate = 1.0e-3
epochs = 50

is_cuda = True
# permutation seed 
seed = 92916


vocab_size = 10000
embedding_dim = 256 

if 'seed' in globals():
	rng_permute = np.random.RandomState(seed)
	idx_permute = rng_permute.permutation(num_steps)
else:
	idx_permute = np.random.permutation(num_steps)



kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=True, download=True,
				   transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=False, transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=batch_size, shuffle=True, **kwargs)


model = Model(hidden_structs, dilations, num_steps, n_classes , batch_size , cell_type)

print(list(model.parameters()))


if is_cuda:
	model.cuda()

#optimizer = optim.RMSprop(model.parameters(), lr= learning_rate, alpha=0.9, eps=1e-08, weight_decay=0, momentum=0, centered=False)


import sys
def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		
		data = data.view(batch_size,-1)
		print(data.size())
		data = data[:,idx_permute]
		data = data.view([batch_size ,num_steps, input_dims])
		


		if is_cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		#optimizer.zero_grad()
		output = model(data)
		loss = nn.MultiLabelSoftMarginLoss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))

def evaluate():
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		if is_cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))
	

for epoch in range(1, epochs + 1):
	train(epoch)
	test()

