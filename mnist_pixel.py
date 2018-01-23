
# coding: utf-8
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from DRNN import Model

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Model(hidden_structs, dilations, num_steps, n_classes , batch_size , cell_type)


#model = Net()

for p in model.parameters():
    if p.grad is not None:
        print(p.grad.data)

print(list(model.parameters()),']]]]]]]]]]]]]')


if is_cuda:
	model.cuda()



optimizer = optim.RMSprop(list(model.parameters()), lr= learning_rate, alpha=0.9, eps=1e-08, weight_decay=0, momentum=0, centered=False)

def _reform(self, tensor, input_dim, n_steps):
		tensor = torch.transpose(tensor,1,0)
		return [t for t in tensor]

def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data = data.squeeze(1)
		data = data.view(batch_size,-1,1)
		inputs = _reform(data, int(data.size()[2]), int(data.size()[1]))

		x = inputs
		print(data.size(),'-========')
		if is_cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data, dilations)
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

