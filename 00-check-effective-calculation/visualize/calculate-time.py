#!/usr/bin/env python
# coding: utf-8

import sys


# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(map(str,args)).replace('\n',''))


# -------------------------------------------------------------------
# -------------------------------------------------------------------
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
def draw(correct, proposal, file_name):
	print(f'draw:{file_name}')
	fig = plt.figure(figsize=(3.7,3.5))
	plt.subplots_adjust(left=0.20, right=0.99, top=0.96, bottom=0.15)
	ax = fig.add_subplot(1,1,1)
#	ax.set_aspect('equal')

	count = 0
	for model in ['SmallMLP','SmallCNN']:
		for n in [10,100,1000]:
			x,y = [],[]
			for trail in range(10):
				x.append(correct[model, trail, n])
				y.append(proposal[model, trail, n])

			print(f'model:{model} n:{n} correct:{np.mean(x)}/{np.std(x)} proposal:{np.mean(y)}/{np.std(y)} ')

# -------------------------------------------------------------------
import os
import json
import pickle
import math
def main():
	report('start draw')	

	archive = 'ARCHIVEs/proposal/'
	if not os.path.exists(archive): return

	correct  = {}
	proposal = {}
	for model in ['SmallCNN','SmallMLP']:
		for trail in os.listdir(f'{archive}/{model}'):
			if '.DS' in trail: continue
			for n in os.listdir(f'{archive}/{model}/{trail}'):
				if '.DS' in n: continue

				with open(f'{archive}/{model}/{trail}/{n}/config.json','r') as fp:
					config = json.load(fp)

				with open(f'{archive}/{model}/{trail}/{n}/measure.pkl','rb') as fp:
					measure = pickle.load(fp)

				trail = int(trail)
				n = int(n)
				correct[model, trail, n ] = measure['correct-time']
				proposal[model, trail, n] = measure['proposal-time']


	output_file = f"VIEW/proposal-time2.png"
	draw(correct, proposal, output_file)

	report('finish')	
	print('-'*50)


# -------------------------------------------------------------------
if __name__=='__main__':
	main()


