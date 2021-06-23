#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(map(str,args)).replace('\n',''))


# -------------------------------------------------------------------
import torch
class LossFunctionOverSigma(torch.nn.Module):
	def __init__(self, h2, h1):
		super().__init__()
		self.sigma1 = torch.nn.Parameter(torch.zeros(h1).double())
		self.sigma2 = torch.nn.Parameter(torch.zeros(h2).double())
		self.loss_function_version = 'arxiv'
#		self.loss_function_version = 'icml'

	# these two loss functions do not return same results,
	#  even though ICML paper claims their correspondence
	def _arxiv(self,H,W):
		s1 = torch.exp(self.sigma1)
		s2 = torch.exp(self.sigma2)
		loss  = torch.dot(s2, torch.mv(H, s1))

		s1 = torch.exp(-self.sigma1)
		s2 = torch.exp(-self.sigma2)
		loss += torch.dot(s2, torch.mv(W, s1))
		return loss

	def _icml(self, H, W):
		x = torch.mv(H, torch.exp(self.sigma1))
		x = torch.mv(W.T, x)
		x = torch.dot(x, torch.exp(-self.sigma1))
		return x.sqrt()

	def forward(self, H, W):
		if self.loss_function_version=='arxiv':
			return self._arxiv(H,W)
		if self.loss_function_version=='icml':
			return self._icml(H,W)

import math
import sys
def _get_layer_sharpness(H,W,lr,num_epoch=10000,decay=1e-3,is_report=True):
	if is_report:
		print('H-sum:',H.sum().item())
		print('W-sum:',W.sum().item())
	# normalize 
	scale = H.sum()+W.sum()
	scale = scale.item()
	H = H/scale
	W = W/scale

	model = LossFunctionOverSigma(*H.shape).double()
	if H.data.is_cuda: model = model.cuda()
#	optimizer = torch.optim.Adam(model.parameters(), lr)	# best lr=1e-1
	optimizer = torch.optim.SGD(model.parameters(), lr)		# best lr=1e-0, faster than Adam? 
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e: 1/(e*decay+1))
	for epoch in range(num_epoch):
		loss = model(H,W)
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
	sharpness = 0
	for (name,Hl), Wl in zip(diagH.named_modules(), model.modules()):
		layer_type = Hl.__class__.__name__
		if layer_type not in {'Conv2d','Linear'}: continue

		Hl = Hl._parameters
		Wl = Wl._parameters

		Hl = Hl['weight'].data.double().detach()
		Wl = Wl['weight'].data.double().detach()

		if layer_type=='Linear':	# both ~ (dout, din)
			Wl = Wl**2

		if layer_type=='Conv2d':# Hl,Wl ~ (dout, din, width, height)
			Wl = Wl**2
			Wl = Wl.sum(dim=2)	# -> (dout, din, height)
			Wl = Wl.sum(dim=2)	# -> (dout, din)

			Hl = Hl.sum(dim=2)	# -> (dout, din, height)
			Hl = Hl.sum(dim=2)	# -> (dout, din)

		sharpness += _get_layer_sharpness(Hl, Wl, lr)
	return sharpness

#