 #!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d/%H-%M-%S.%f')+' '+' '.join(map(str,args)))

# ------------------------------------------------------
import time
import os
import subprocess
def pworker(deq,core):
	time.sleep(core*5)
	snapshot = f'snapshot/{core:02d}/'
	while len(deq)!=0:
		config = deq.popleft()
		
		command = ['nohup','python','-uO', config['script']]
#		command = ['nohup','python','-u', config['script']]
		for key, value in config.items():
			if key!='script': command += [f'--{key}', f'{value}']
		command += [f'--core',		f'{core}']
		command += [f'--snapshot',	f'{snapshot}']

		report(f'remains:{len(deq)} core:{core} command:{" ".join(command)}')
		if not __debug__:
			os.makedirs(snapshot,exist_ok=True)
			with open(f'{snapshot}/log.txt','w') as logfile:
				subprocess.call(command, stdout=logfile)


# ------------------------------------------------------
# ------------------------------------------------------
import itertools
def get_parallel_comb(hypara:dict) ->list:
	configs = []
	keys = list(hypara.keys())
	for values in itertools.product(*[hypara[k] for k in keys]):
		config	= {key:value for key,value in zip(keys,values)}
		configs.append(config)
	return configs

# ------------------------------------------------------
import copy
import itertools
import numpy as np
def get_config1(num_trial) -> list:
	report('get model-config1')
	configs = []
	for trial in range(num_trial):
		for ratio in [i/10 for i in range(11)][::-1]:
			for model,lr,epoch in zip(['LeNet','MLP'],[1e-2,1e-1],[1000,3000]):
				d = f'ARCHIVEs/train/{model}/{trial}/{ratio}/'
				if all(os.path.exists(f'{d}/{ff}.pkl') for ff in ['trails','params']): continue
				hypara = {}
				hypara['script']		= ['experiment/work-train.py'] 
				hypara['archive']		= [f'ARCHIVEs/train/{model}/{trial}/{ratio}']
				hypara['num_epoch']		= [epoch]
				hypara['lr']			= [lr]
				hypara['seed']			= [-1]
				hypara['ratio']			= [ratio]
				hypara['model']			= [model]
				configs += get_parallel_comb(hypara)
	print(f'model configs size:{len(configs)}')
	return configs

def get_config2(num_trial) -> list:
	report('get model-config2')
	configs = []
	for selection in ['last-epoch']:
		for trial in range(num_trial):
			for model in ['LeNet','MLP']:
				hypara = {}
				hypara['script']		= ['experiment/work-tsuzuku.py'] 
				hypara['param_dir']		= [f'ARCHIVEs/train/{model}/{trial}']
				hypara['archive']		= [f'ARCHIVEs/tsuzuku/{selection}/{model}/{trial}']
				hypara['batch_size']	= [2**12]
				hypara['lr'] 			= [1e-3]
				hypara['model']			= [model]
				hypara['selection']		= [selection]
				if os.path.exists(hypara['archive'][0]): continue
				configs += get_parallel_comb(hypara)
	return configs

def get_config3(num_trial) -> list:
	report('get model-config3')
	configs = []
	for selection in ['last-epoch']:
		for trial in range(num_trial):
			for model in ['LeNet','MLP']:
				hypara = {}
				hypara['script']		= ['experiment/work-proposal.py'] 
				hypara['param_dir']		= [f'ARCHIVEs/train/{model}/{trial}']
				hypara['archive']		= [f'ARCHIVEs/proposal/{selection}/{model}/{trial}']
				hypara['batch_size'] 	= [2**12]
				hypara['lr'] 			= [1e-3]
				hypara['model']			= [model]
				hypara['selection']		= [selection]
				if os.path.exists(hypara['archive'][0]): continue
				configs += get_parallel_comb(hypara)
	return configs

# ----------------------
from collections	import deque
from threading		import Thread
def main():
	print('-'*50)
	report('start')

	available_cores = list(range(1))
	print(f'available cores:{available_cores}')

	num_trial = 10
	config_list = []
	config_list.append(get_config1(num_trial))
	config_list.append(get_config2(num_trial))
	config_list.append(get_config3(num_trial))

	for configs in config_list:
		deq 	= deque(configs)
		print('-'*50)

		pool = []
		for core in available_cores:
			pool.append(Thread(target=pworker,args=(deq,core)))
		for p in pool:
			p.start()
		for p in pool:
			p.join()
		print('-'*50)

	if os.path.exists('snapshot'):
		os.rmdir('snapshot')

	report('finish')

if __name__ == '__main__':
	main()



# ----------------------
