#! -*- coding:utf-8 -*-

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.backends.cudnn.benchmark = True

# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d/%H:%M:%S')+' '+' '.join(map(str,args)).replace('\n',''))

# -------------------------------------------------------------------
import torch
import torchvision
import random
from collections import namedtuple
def get_data(data_size, core=-1):
	transform = []
	transform.append(torchvision.transforms.ToTensor())
	transform.append(torchvision.transforms.Normalize((0.5,), (0.5,)))
	transform = torchvision.transforms.Compose(transform)
	def _get_simplified_dataset(is_train):
		dataset  = torchvision.datasets.MNIST(root='DOWNLOADs', train=is_train, download=True,transform=transform)
		x = torch.stack([v[0] for v in dataset])
		y = dataset.targets
		index = torch.randperm(len(y))
		x = x[index]
		y = y[index]
		x = x[:data_size]
		y = y[:data_size]
		if 0<=core:
			x = x.cuda(core)
			y = y.cuda(core)
		return namedtuple('_','x y n')(x=x, y=y,n=len(y))
	return _get_simplified_dataset(is_train=True)

# -------------------------------------------------------------------
import time
import copy
import torch.nn.functional as F
import Correct_Hessian as correct
import Hessian_diag as hack
from torch.nn.utils import parameters_to_vector
def evaluate_diag(origin, data):
	measure = {}

	report('diag[H]')

	model 		= copy.deepcopy(origin)
	time_start	= time.perf_counter()
	loss 		= F.cross_entropy(model(data.x), data.y, reduction='mean')
	correctH 	= correct.diagonal(loss, model)
	time_end	= time.perf_counter()

	report(f'\tCorrect   time:{time_end - time_start: 9.4f} sec')
	measure['correct-time'] = time_end - time_start

	model 		= copy.deepcopy(origin)
	time_start	= time.perf_counter()
	effectiveH	= hack.effective(data.x, model, batch_size=1024)
	time_end	= time.perf_counter()

	effectiveH = effectiveH.parameters()
	effectiveH = parameters_to_vector(effectiveH)

	report(f'\tEffective time:{time_end - time_start: 9.4f} sec')
	measure['proposal-time'] = time_end - time_start

	report(f'\tTr[CorrectH]  :{correctH.sum().item(): 9.4f}')
	report(f'\tTr[EffectiveH]:{effectiveH.sum().item(): 9.4f}')
	
	measure['correct-tr[H]']	= correctH.sum().item()
	measure['proposal-tr[H]']	= effectiveH.sum().item()

	diff = (correctH-effectiveH).norm().item()
	report(f'\t|CorrectH - EffectiveH|_2 {diff: 9.4f}')

	measure['dist'] = diff
	measure['correct']  = correctH.detach().cpu()
	measure['proposal'] = effectiveH.detach().cpu()

	return measure


# -------------------------------------------------------------------
import torch
def set_device(config):
	if 0<=config.core<torch.cuda.device_count() and torch.cuda.is_available():
		report(f'use GPU; core:{config.core}')
		torch.cuda.set_device(config.core)
	else:
		report('use CPU in this trial')
		config.core = -1

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
	args.add_argument('--num_data',			default=10,			type=int)	# need to use same data
	args.add_argument('--model',			default='',			type=str, choices=['SmallMLP','SmallCNN'])
	return args.parse_args()

# -------------------------------------------------------------------
import model_zoo as zoo
import pickle
import os
import copy
import json
def main():
	config	= get_config()

	set_device(config)
	torch.manual_seed(config.seed)
		
	if config.model=='SmallMLP':
		model = zoo.SmallMLP()
	if config.model=='SmallCNN':
		model = zoo.SmallCNN()

	if 0<=config.core:
		model = model.cuda()

	data  = get_data(data_size=config.num_data, core=config.core)

	measure = evaluate_diag(model, data)

	with open(f'{config.snapshot}/config.json', 'w') as fp:
		json.dump(config.__dict__, fp, indent=2)

	with open(f'{config.snapshot}/measure.pkl', 'wb') as fp:
		pickle.dump(measure, fp)

	os.makedirs(config.archive,	exist_ok=True)
	os.rename(config.snapshot,	config.archive)


# -------------------------------------

if __name__ == '__main__':
	main()


# -------------------------------------