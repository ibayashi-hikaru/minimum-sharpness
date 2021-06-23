#!/usr/bin/env python
# coding: utf-8

#"""
import torch
torch.set_num_threads(7)
torch.set_num_interop_threads(7)
torch.backends.cudnn.benchmark = True
#"""


# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+' '+' '.join(map(str,args)).replace('\n',''))

# -------------------------------------------------------------------
import torch
import torchvision
import numpy
from collections import namedtuple
def get_dataset(config):
	transform = []
	transform.append(torchvision.transforms.ToTensor())
	transform.append(torchvision.transforms.Normalize((0.5,), (0.5,)))
	transform = torchvision.transforms.Compose(transform)

	def _get_simplified_dataset(is_train):
		dataset  = torchvision.datasets.MNIST(root='DOWNLOADs', train=is_train, download=True,transform=transform)
		x = torch.stack([v[0] for v in dataset])
		y = dataset.targets

		if is_train:		# shuffle labels 
			n = len(y)
			num_misslabel	= int(n*config.ratio)
			random_index	= numpy.random.choice(n, num_misslabel, replace=False)
			random_order	= torch.randperm(num_misslabel)
			y[random_index]	= y[random_index][random_order]
		if 0<=config.core:
			x = x.cuda()
			y = y.cuda()
		return namedtuple('_','x y n')(x=x, y=y,n=len(y))

	train = _get_simplified_dataset(is_train=True)
	test  = _get_simplified_dataset(is_train=False)
	return namedtuple('_','train test')(train=train, test=test)

import model_zoo
def get_model(config):
	if config.model=='LeNet':	model = model_zoo.LeNet()
	if config.model=='MLP':		model = model_zoo.MLP()
	if 0<=config.core:			model = model.cuda()
	return model

# -------------------------------------------------------------------
import torch
import math
def optimize(dataset, model, config):
	loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
	optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=1e-5, momentum=0.9)
#	optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=1e-5, momentum=0.9)
#	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda e:1/(config.decay*e+1))

	def update():
		model.train()
		measure = {'loss':0,'accuracy':0}
		index = torch.randperm(dataset.train.n)
		for idx in torch.split(index, config.batch_size):
			x = dataset.train.x[idx]
			y = dataset.train.y[idx]
			o = model(x)
			loss = loss_func(o,y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			measure['loss'] 	+= loss.item()*len(idx)
			measure['accuracy'] += (o.max(dim=1)[1]==y).sum().item()	# logit -> index -> bool -> int
		measure['loss'] 	/= dataset.train.n
		measure['accuracy'] /= dataset.train.n
		return measure

	def evaluate():
		model.eval()
		with torch.no_grad():
			output	= model(dataset.test.x)			# logit
			loss	= loss_func(output, dataset.test.y)
			loss 	= loss.item()
			output	= output.max(dim=1)[1]			# logit -> index
			correct	= (output==dataset.test.y)		# index -> bool
			correct	= correct.float().mean().item()	# bool -> int(0,1) -> float
			return {'loss':loss, 'accuracy':correct}

	trails = []
	params = []
	for epoch in range(config.num_epoch):
		status = {}
		status['train'] = update()
		status['test']	= evaluate()
		trails.append(status)
		params.append(model.get_param())

		report(f'epoch:{epoch}/{config.num_epoch}')
		for mode in ['train','test']:
			message	 = [f'\t{mode:5}']
			message	+= [f"loss:{status[mode]['loss']: 18.7f}"]
			message	+= [f"accuracy:{status[mode]['accuracy']: 9.7f}"]
			report(*message)
	return trails, params

# ------------------------------------------------------
import argparse
import datetime
def get_config():
	args = argparse.ArgumentParser()
	# general
	args.add_argument('--snapshot',			default='snapshot',	type=str)
	args.add_argument('--archive',								type=str,	default=f"ARCHIVEs/{datetime.datetime.now().strftime('%Y-%m-%d=%H-%M-%S-%f')}")
	args.add_argument('--seed',				default=-1,			type=int)
	args.add_argument('--core',				default=-1,			type=int)
	args.add_argument('--model',			default='LeNet',	type=str)
	# miss-label ratio 
	args.add_argument('--ratio',			default=0.5,		type=float)
	# optim 
	args.add_argument('--num_epoch',		default=4,			type=int)
	args.add_argument('--batch_size',		default=2**10,		type=int)
	args.add_argument('--lr',				default=1e-1,		type=float)
	args.add_argument('--decay',			default=1e-4,		type=float)
	return args.parse_args()

# -------------------------------------------------------------------	
import random
import numpy
import torch
def set_seed(config):
	if config.seed<0:
		import time
		config.seed = int(time.time())
		report(f'get random-seed:{config.seed}')
	random.seed(config.seed)
	numpy.random.seed(config.seed)
	torch.manual_seed(config.seed)	

import torch
def set_device(config):
	if 0<=config.core<torch.cuda.device_count() and torch.cuda.is_available():
		report(f'use GPU; core:{config.core}')
		torch.cuda.set_device(config.core)
	else:
		report('use CPU in this trial')
		config.core = -1

# -------------------------------------------------------------------	
import os
import json
import pickle
def main():
	config	= get_config()
	
	set_seed(config)
	set_device(config)

	dataset	= get_dataset(config)
	model 	= get_model(config)
	trails, params	= optimize(dataset, model, config)

	os.makedirs(config.snapshot, exist_ok=True)
	with open(f'{config.snapshot}/config.json', 'w') as fp:
		json.dump(config.__dict__, fp, indent=2)
	with open(f'{config.snapshot}/trails.pkl', 'wb') as fp:
		pickle.dump(trails, fp)
	with open(f'{config.snapshot}/params.pkl', 'wb') as fp:
		pickle.dump(params, fp)

	os.makedirs(config.archive,	exist_ok=True)
	os.rename(config.snapshot,	config.archive)


if __name__=='__main__':
	main()

# -------------------------------------------------------------------

