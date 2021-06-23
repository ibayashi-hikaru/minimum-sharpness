#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(map(str,args)).replace('\n',''))

# -------------------------------------------------------------------
import torch
from . import gradient_per_example as hacks
def effective(inputs, model, batch_size):
	model.eval()
	
	# for all
	sum_diag_H = 0
	hacks.add_hooks(model)

	n = len(inputs)
	for minibatch in torch.split(inputs, batch_size):
		output		= model(minibatch)					# := (b,K)
		
		# for p*p^T part
		Z		= torch.logsumexp(output, dim=1)		# -> (b)
		loss 	= Z.sum()								# -> (1)

		loss.backward(retain_graph=True)
		hacks.compute_grad1(model)
		model.zero_grad()
		for param in model.parameters():
			grad1 = param.grad1**2						# := (b,*)
			grad1 = torch.sum(grad1)					# -> (,)
			sum_diag_H -= grad1.item()
		hacks.clear_grad1(model)

		# for diag(p) part
		prob		= torch.softmax(output,dim=1)		# -> (b,K)
		grad_norms	= output.new_zeros(output.shape)	# := (b,K)

		output	= output.sum(dim=0)						# -> (K)
		for k,output_k in enumerate(output):
			output_k.backward(retain_graph=True)
			hacks.compute_grad1(model)
			model.zero_grad()
			for param in model.parameters():
				grad1 = param.grad1**2					# := (b,*)
				grad1 = grad1.flatten(1,-1)				# -> (b,-1)
				grad1 = grad1.sum(dim=1)				# -> (b,)
				grad_norms[:,k] += grad1.detach()		# -> (b,)
			hacks.clear_grad1(model)

		sum_diag_H += torch.sum(grad_norms*prob).item()	

	sum_diag_H = sum_diag_H/n

	hacks.remove_hooks(model)
	return sum_diag_H

