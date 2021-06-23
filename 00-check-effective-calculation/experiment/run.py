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
		
#		command = ['nohup','python','-uO', config['script']]
		command = ['nohup','python','-u', config['script']]
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
def get_config() -> list:
	report('get model-config1')
	configs = []
	seed = 0
	for n in [10,100,1000]:
		for trail in range(10):
			for model in ['SmallMLP','SmallCNN']:
				seed += 1
				hypara = {}
				hypara['script']		= ['experiment/mainProposal.py'] 
				hypara['archive']		= [f'ARCHIVEs/proposal/{model}/{trail}/{n}']
				hypara['num_data']		= [n]
				hypara['seed']			= [seed]
				hypara['model']			= [model]
				configs += get_parallel_comb(hypara)
	print(f'model configs size:{len(configs)}')
	return configs


# ----------------------
from collections	import deque
from threading		import Thread
def main():
	print('-'*50)
	report('start')

	available_cores = list(range(1))
	print(f'available cores:{available_cores}')

	config_list = []
	config_list.append(get_config())

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
