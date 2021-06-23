#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(map(str,args)).replace('\n',''))


# -------------------------------------------------------------------
import torch
class LossFunctionOverAlpha(torch.nn.Module):
	def __init__(self, num_of_layer):
		super().__init__()
		self.exponent = torch.nn.Parameter(torch.zeros(num_of_layer-1).double())
		self.L = num_of_layer-1

	def forward(self, weights, biases):
		loss = 0
		alpha = torch.exp(self.exponent)
		prod_scale = 1
		for l in range(self.L):
			loss += weights[l]/(alpha[l]**2)
			prod_scale = prod_scale*alpha[l]
			loss += biases[l]/(prod_scale**2)
		loss += weights[self.L]*(prod_scale**2)
		loss += biases[self.L]
		return loss

import copy
import math
import sys
def _get_sharpness(weights,biases,lr,num_epoch=10000,decay=1e-3,is_report=True):
	# normalize 
	scale = sum(weights.values())+sum(biases.values())
	for k in weights:
		weights[k] /= scale
	for k in biases:
		biases[k] /= scale

	num_of_layer = max(len(weights), len(biases))
	model = LossFunctionOverAlpha(num_of_layer).double()
#	optimizer = torch.optim.Adam(model.parameters(), lr)	# best lr=1e-1
	optimizer = torch.optim.SGD(model.parameters(), lr)		# best lr=1e-0, faster than Adam? 
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e: 1/(e*decay+1))
	for epoch in range(num_epoch):
		loss = model(weights, biases)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
		loss = loss.item()
		assert not math.isnan(loss), 'nan found, use smaller steps size'
		if is_report and epoch%1000==0:	report(f'epoch:{epoch} loss:{loss: 18.7f}')
	if is_report:	print('-'*50)
	return loss*scale

# -------------------------------------------------------------------
from . import Hessian_diag as Hessian_diag
def effective(data, model, batch_size, lr):
	diagH = Hessian_diag.effective(data.x, model, batch_size)
	weights, biases = {},{}
	for name,param in diagH.named_parameters():
		if not name.startswith('layer'): continue
		name = name.replace('layer','')
		if name.endswith('.weight'):	# layer[i].weight
			name = name.replace('.weight','')
			l = int(name)
			weights[l] = param.sum().item()
		if name.endswith('.bias'):		# layer[i].bias
			name = name.replace('.bias','')
			l = int(name)
			biases[l] = param.sum().item()
	return _get_sharpness(weights, biases, lr)



#