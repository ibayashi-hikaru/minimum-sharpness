#!/usr/bin/env python
# coding: utf-8

"""
import torch
torch.set_num_threads(7)
torch.set_num_interop_threads(7)
torch.backends.cudnn.benchmark = True
"""

# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+' '+' '.join(map(str,args)).replace('\n',''))

# -------------------------------------------------------------------
import torch
import torchvision
from collections import namedtuple
def get_data(config):
	transform = []
	transform.append(torchvision.transforms.ToTensor())
	transform.append(torchvision.transforms.Normalize((0.5,), (0.5,)))
	transform = torchvision.transforms.Compose(transform)

	def _get_simplified_dataset(is_train):
		dataset  = torchvision.datasets.MNIST(root='DOWNLOADs', train=is_train, download=True,transform=transform)
		x = torch.stack([v[0] for v in dataset])
		y = dataset.targets
		if 0<=config.core:
			x = x.cuda()
			y = y.cuda()
		return namedtuple('_','x y n')(x=x, y=y,n=len(y))
	train = _get_simplified_dataset(is_train=True)
	return train


import model_zoo
def get_model(config):
	if config.model=='LeNet':	model = model_zoo.LeNet()
	if config.model=='MLP':		model = model_zoo.MLP()
	if 0<=config.core:			model = model.cuda()
	return model

import numpy
def get_selection(config, ratio):
	with open(f'{config.param_dir}/{ratio}/trails.pkl','rb') as fp:
		trails = pickle.load(fp)

	if config.selection.endswith('loss'):
		scores = numpy.array([status['train']['loss'] for status in trails])
	if config.selection.endswith('accuracy'):
		scores = numpy.array([status['train']['accuracy'] for status in trails])

	index = -1	# for 'last-epoch'
	if config.selection=='first-loss':
		index = numpy.argmin(scores)
	if config.selection=='last-loss':
		index = np.where(scores==scores.min())[0].max()
	if config.selection=='first-accuracy':
		index = numpy.argmax(scores)
	if config.selection=='last-accuracy':
		index = np.where(scores==scores.max())[0].max()	# for latest error-rate or loss

	file_name = f'{config.param_dir}/{ratio}/params.pkl'
	with open(file_name,'rb') as fp:
		params = pickle.load(fp)
	return index, params[index]


# ------------------------------------------------------
import argparse
import datetime
def get_config():
	args = argparse.ArgumentParser()
	# general
	args.add_argument('--snapshot',			default='snapshot',	type=str)
	args.add_argument('--archive',								type=str,	default=f"ARCHIVEs/{datetime.datetime.now().strftime('%Y-%m-%d=%H-%M-%S-%f')}")
	args.add_argument('--core',				default=7,			type=int)
	args.add_argument('--seed',				default=0,			type=int)	# need to use same data
	args.add_argument('--model',			default='MLP',		type=str)
	args.add_argument('--batch_size',		default=2**12,		type=int)	# need to prevent out of memory (GPU)
	args.add_argument('--lr',				default=1e-1,		type=float)	# need to obtain layer sharpness
	args.add_argument('--param_dir',		default='',			type=str)
	args.add_argument('--selection',		default='',			type=str, 	choices=['last-epoch','first-loss','last-loss','first-accuracy','last-accuracy'])
	return args.parse_args()

# -------------------------------------------------------------------	
import time
import random
import numpy as np
import torch
def set_seed(config):
	random.seed(config.seed)
	np.random.seed(config.seed)
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
import sharpness.Minimum as minimum_sharpness
def main():
	config	= get_config()
	
	set_seed(config)
	set_device(config)

	data	= get_data(config)
	model	= get_model(config)

	selected_index = {}
	sharpness = {}
	for ratio in sorted(os.listdir(config.param_dir)):
		if '.DS' in ratio: continue
		report(f'ratio:{ratio}')
		
		index, param = get_selection(config, ratio)
		model.set_param(param)
		selected_index[ratio] = index

		sharpness[ratio] = minimum_sharpness.effective(data, model, config.batch_size, config.lr)

	os.makedirs(config.snapshot, exist_ok=True)
	with open(f'{config.snapshot}/config.json', 'w') as fp:
		json.dump(config.__dict__, fp, indent=2)
	with open(f'{config.snapshot}/selected_index.pkl', 'wb') as fp:
		pickle.dump(selected_index, fp)
	with open(f'{config.snapshot}/sharpness.pkl', 'wb') as fp:
		pickle.dump(sharpness, fp)

	os.makedirs(config.archive,	exist_ok=True)
	os.rename(config.snapshot,	config.archive)


if __name__=='__main__':
	main()

# -------------------------------------------------------------------

