#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(map(str,args)).replace('\n',''))





# -------------------------------------------------------------------
import torch
import copy
from torch.nn.utils import parameters_to_vector
from torch.nn.utils import vector_to_parameters
from torch.nn.functional import cross_entropy
import copy
import time
def tsuzuku_arxiv(data, model, num_sample=10, r=1e-5, is_report=False):
	def get_gradient():
		model.zero_grad()
		loss = cross_entropy(model(data.x),data.y,reduction='sum')
		loss.backward()
		for param in model.parameters():
			param.data = param.grad.data/data.n
		grad = parameters_to_vector(model.parameters())
		return grad.detach()

	origin = model.parameters()
	origin = parameters_to_vector(origin)
	origin = copy.deepcopy(origin.detach())

	trails = {'diagHs':[],'times':[]}
	accumulate_diagH = torch.zeros_like(origin)
	for i in range(num_sample):
		if is_report: report(f'step:{i}')

		time_start	= time.perf_counter()


		epsilon = torch.randn(origin.shape)
		if data.x.is_cuda: epsilon = epsilon.cuda()

		plus_param  = origin + r*epsilon
		vector_to_parameters(plus_param, model.parameters())
		plus_grad = get_gradient()

		minus_param = origin - r*epsilon
		vector_to_parameters(minus_param, model.parameters())
		minus_grad = get_gradient()

		accumulate_diagH += epsilon*(plus_grad - minus_grad)/(2*r)

		time_end	= time.perf_counter()

		diagH = accumulate_diagH / (i+1)
		diagH = copy.deepcopy(diagH).detach().cpu()
		trails['diagHs'].append(diagH)
		trails['times'].append(time_end - time_start)

	return trails





# -------------------------------------------------------------------
import torch
import copy
from torch.nn.utils import parameters_to_vector
from torch.nn.utils import vector_to_parameters
from torch.nn.functional import cross_entropy
import copy
import time
def tsuzuku_icml(data, model, num_sample=10, r=1e-5, epsilon=1e-8, is_report=False):
	def get_gradient():
		model.zero_grad()
		loss = cross_entropy(model(data.x),data.y,reduction='sum')
		loss.backward()
		for param in model.parameters():
			param.data = param.grad.data/data.n
		grad = parameters_to_vector(model.parameters())
		return grad.detach()

	origin = model.parameters()
	origin = parameters_to_vector(origin)
	origin = copy.deepcopy(origin.detach())

	trails = {'diagHs':[],'times':[]}
	accumulate_diagH = torch.zeros_like(origin)
	for i in range(num_sample):
		if is_report: report(f'step:{i}')

		time_start	= time.perf_counter()

		s = torch.randint(0,2,origin.shape)*2-1
		if data.x.is_cuda: s = s.cuda()

		plus_param  = origin + r*s
		vector_to_parameters(plus_param, model.parameters())
		plus_grad = get_gradient()

		minus_param = origin - r*s
		vector_to_parameters(minus_param, model.parameters())
		minus_grad = get_gradient()

		denominator = 2*r*origin.abs()+epsilon
		accumulate_diagH += s*(plus_grad - minus_grad)/denominator

		time_end	= time.perf_counter()
		
		diagH = accumulate_diagH / (i+1)
		diagH = torch.relu(diagH)
		diagH = copy.deepcopy(diagH).detach().cpu()
		trails['diagHs'].append(diagH)
		trails['times'].append(time_end - time_start)

	return trails




# -------------------------------------------------------------------
import copy
import torch
import gradient_per_example as hacks
def effective(inputs, model, batch_size):
	model.eval()

	# for all
	diagH = copy.deepcopy(model)
	for param in diagH.parameters():
		param.data.zero_()

	hacks.add_hooks(model)

	n = len(inputs)
	for minibatch in torch.split(inputs, batch_size):
		output		= model(minibatch)						# := (b,K)

		# for p*p^T part
		Z		= torch.logsumexp(output, dim=1)			# -> (b)
		loss 	= Z.sum()									# -> (1)

		loss.backward(retain_graph=True)
		hacks.compute_grad1(model)
		model.zero_grad()
		for param, diag in zip(model.parameters(), diagH.parameters()):
			grad1 = param.grad1**2 							# := (b,*)
			grad1 = torch.sum(grad1,dim=0) 					# -> (*)
			diag.data -= grad1.detach()
		hacks.clear_backword1(model)

		# for diag(p) part
		prob	= torch.softmax(output,dim=1)				# -> (b,K)
		output	= output.sum(dim=0)							# -> (K)
		for k,output_k in enumerate(output):
			output_k.backward(retain_graph=True)
			hacks.compute_grad1(model)
			model.zero_grad()
			for param, diag in zip(model.parameters(), diagH.parameters()):
				grad1 = param.grad1**2						# := (b,*)
				grad1 = grad1.flatten(1,-1)					# -> (b,prod(*))
				grad1 = prob[:,k].flatten() @ grad1 		# -> (prod(*))
				grad1 = grad1.reshape(param.data.shape)		# -> (*)
				diag.data += grad1.detach()
			hacks.clear_backword1(model)

	for param in diagH.parameters():
		param.data /= n
		
	hacks.remove_hooks(model)
	return diagH
