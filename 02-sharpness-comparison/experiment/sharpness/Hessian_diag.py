#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(map(str,args)).replace('\n',''))

# -------------------------------------------------------------------
import copy
import torch
from . import gradient_per_example as hacks
def effective(inputs, model, batch_size, is_report=False):
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
