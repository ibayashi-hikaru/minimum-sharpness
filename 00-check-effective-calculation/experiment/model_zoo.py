#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
import copy
import torch

# -------------------------------------------------------------------
class SmallCNN(torch.nn.Module):
	def __init__(self):
		super(SmallCNN, self).__init__()
		self.layer0	= torch.nn.Conv2d(1,  20, 5, stride=1, padding=0)
		self.layer1	= torch.nn.Conv2d(20, 20, 5, stride=1, padding=0)
		self.layer2	= torch.nn.Linear(4*4*20, 10)

	def forward(self, x):							# := (b, 1,28,28)
		x = torch.nn.functional.relu(self.layer0(x))# -> (b,20,24,24)
		x = torch.nn.functional.max_pool2d(x, 2, 2)	# -> (b,20,12,12)
		x = torch.nn.functional.relu(self.layer1(x))# -> (b,20, 8, 8)
		x = torch.nn.functional.max_pool2d(x, 2, 2)	# -> (b,20, 4, 4)
		x = x.flatten(1,-1)							# -> (b,20*4*4)
		return self.layer2(x)

	def get_param(self):
		state = copy.deepcopy(self.state_dict())
		for k in state:
			state[k] = state[k].cpu().detach().numpy()
		return state

	def set_param(self, param):
		for k in param:
			param[k] = torch.tensor(param[k],dtype=torch.float)
		self.load_state_dict(param)

class SmallMLP(torch.nn.Module):
	def __init__(self):
		super(SmallMLP, self).__init__()
		self.layer0	= torch.nn.Linear(28*28, 20)
		self.layer1	= torch.nn.Linear(  20,  20)
		self.layer2	= torch.nn.Linear(  20,  10)

	def forward(self, x):								# := (b, 1,28,28)	# (batch, channel, width, height)
		x = x.flatten(1,-1)								# -> (b, 1*28*28)
		x = torch.nn.functional.relu(self.layer0(x))	# -> (b,100)
		x = torch.nn.functional.relu(self.layer1(x))	# -> (b,100)
		return self.layer2(x)							# -> (b,10)

	def get_param(self):
		state = copy.deepcopy(self.state_dict())
		for k in state:
			state[k] = state[k].cpu().detach().numpy()
		return state

	def set_param(self, param):
		for k in param:
			param[k] = torch.tensor(param[k],dtype=torch.float)
		self.load_state_dict(param)

# -------------------------------------------------------------------
class LeNet(torch.nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.layer0	= torch.nn.Conv2d( 1,   6, 5, stride=1, padding=2)
		self.layer1	= torch.nn.Conv2d( 6,  16, 5, stride=1, padding=0)
		self.layer2	= torch.nn.Conv2d(16, 120, 5, stride=1, padding=0)
		self.layer3	= torch.nn.Linear(120*7*7, 84)
		self.layer4	= torch.nn.Linear(     84, 10)

	def forward(self, x):								# := (b,  1,28,28)	# (batch, channel, width, height)
		x = torch.nn.functional.relu(self.layer0(x))	# -> (b,  6,28,28)	# (28)/1 = 28?
		x = torch.nn.functional.max_pool2d(x, 2, 1)		# -> (b,  6,27,27)	# stride=1
		x = torch.nn.functional.relu(self.layer1(x))	# -> (b, 16,23,23)
		x = torch.nn.functional.max_pool2d(x, 2, 2)		# -> (b, 16,11,11)
		x = torch.nn.functional.relu(self.layer2(x))	# -> (b,120, 7, 7)
		x = x.flatten(1,-1)								# -> (b,120*7*7)
		x = torch.nn.functional.relu(self.layer3(x))	# -> (b,84)
		return self.layer4(x)							# -> (b,10)

	def get_param(self):
		state = copy.deepcopy(self.state_dict())
		for k in state:
			state[k] = state[k].cpu().detach().numpy()
		return state

	def set_param(self, param):
		for k in param:
			param[k] = torch.tensor(param[k],dtype=torch.float)
		self.load_state_dict(param)

class MLP(torch.nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.layer0	= torch.nn.Linear(28*28, 100)
		self.layer1	= torch.nn.Linear(  100, 100)
		self.layer2	= torch.nn.Linear(  100,  10)

	def forward(self, x):								# := (b, 1,28,28)	# (batch, channel, width, height)
		x = x.flatten(1,-1)								# -> (b, 1*28*28)
		x = torch.nn.functional.relu(self.layer0(x))	# -> (b,100)
		x = torch.nn.functional.relu(self.layer1(x))	# -> (b,100)
		return self.layer2(x)							# -> (b,10)

	def get_param(self):
		state = copy.deepcopy(self.state_dict())
		for k in state:
			state[k] = state[k].cpu().detach().numpy()
		return state

	def set_param(self, param):
		for k in param:
			param[k] = torch.tensor(param[k],dtype=torch.float)
		self.load_state_dict(param)

# -------------------------------------------------------------------
