#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(map(str,args)).replace('\n',''))

# -------------------------------------------------------------------
import torch
def diagonal(loss, model):
	params = list(model.parameters())
	n = sum(p.numel() for p in params)
	diagH = loss.new_zeros(n)

	k = 0	# index on H, s.t., 0<= k < n
	for i, param in enumerate(params):
		[grad]	= torch.autograd.grad(loss, param, create_graph=True)
		grad	= grad.flatten()

		for j in range(param.numel()):
			if not grad[j].requires_grad: continue

			H_k = torch.autograd.grad(grad[j], params[i], create_graph=False, retain_graph=True)
			H_k = torch.cat([h.flatten() for h in H_k])
			diagH[k] = H_k[j]

			k +=1
	return diagH

# -------------------------------------------------------------------
def full(loss, model):
	params = list(model.parameters())
	n = sum(p.numel() for p in params)
	H = loss.new_zeros(n, n)

	k = 0	# index on H, s.t., 0<= k < n
	for i, param in enumerate(params):
		[grad]	= torch.autograd.grad(loss, param, create_graph=True)
		grad	= grad.flatten()

		for j in range(param.numel()):
			if not grad[j].requires_grad: continue

			H_k = torch.autograd.grad(grad[j], params[i:], create_graph=False, retain_graph=True)
			H_k = torch.cat([h.flatten() for h in H_k])
			H_k = H_k[j:]

			H[k, k:] = H_k
			H[k:, k] = H_k
			k +=1
	return H
