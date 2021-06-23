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

from matplotlib import pyplot as plt
#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']

def draw(correct, proposal, file_name):
	print(f'draw:{file_name}')
	fig = plt.figure(figsize=(3.5,3.7))
	plt.subplots_adjust(left=0.17, right=0.99, top=0.9, bottom=0.15)
	ax = fig.add_subplot(1,1,1)
	ax.set_aspect('equal')

	rets = {}
	labels = {}

	for model in ['SmallMLP','SmallCNN']:
		rets[model]=[]
		labels[model] = []
		for n in [10,100,1000]:
			x,y = [],[]
			for trail in range(10):
				y.append(correct[model, trail, n])
				x.append(proposal[model, trail, n])

			if model=='SmallMLP':
#				label = rf'n={{{n}}}'
				label = f'n={n}'
				if n==10:		color = cm.viridis(0.2)
				if n==100:		color = cm.viridis(0.4)
				if n==1000:		color = cm.viridis(0.8)
			if model=='SmallCNN':
#				label = rf'n={{{n}}}'
				label = f'n={n}'
				if n==10:		color = cm.hot(0.2)
				if n==100:		color = cm.hot(0.4)
				if n==1000:		color = cm.hot(0.6)

			labels[model]+=[label]
			rets[model].append(ax.scatter(x, y, s=100, color=color, zorder=2, marker='x', lw=2))

	ax.plot([0,100],[0,100],color='black',zorder=0,lw=1.0)
	ax.set_xlim(10,73)
	ax.set_ylim(10,73)

	model = 'SmallMLP'
	leg1 = ax.legend(rets[model],labels[model], loc='upper left', title='FCNNs', title_fontsize=15, fontsize=16, framealpha=1.0,borderaxespad=0.2, columnspacing=0.2,handletextpad=0.2, frameon=False,bbox_to_anchor=(-0.03, 1.13))
	leg1.set_zorder(-1)

	model = 'SmallCNN'
	from matplotlib.legend import Legend
	leg = ax.legend(rets[model],labels[model], loc='lower right',title='CNNs', title_fontsize=15, fontsize=16, framealpha=1.0,borderaxespad=0.2, columnspacing=0.2,handletextpad=0.2, frameon=False,bbox_to_anchor=(1.08, 0.0))
	leg.set_zorder(-1)

	ax.add_artist(leg1);

#	ax.set_xlabel(r'Proposal Tr$[\bm{H}]$', fontsize=18)
#	ax.set_ylabel(r'Baseline Tr$[\bm{H}]$', fontsize=18)
	ax.set_xlabel('Proposal Tr[H]', fontsize=18)
	ax.set_ylabel('Baseline Tr[H]', fontsize=18)

	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_bounds(10, 65)
	ax.spines['bottom'].set_bounds(10, 65)

	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

	ticks = [20,40,60]
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)

	os.makedirs(os.path.dirname(file_name),exist_ok=True)
	plt.savefig(file_name,dpi=120)
	plt.style.use('default');plt.clf();plt.cla();plt.close()



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
				correct[model, trail, n ] = measure['correct-tr[H]']
				proposal[model, trail, n] = measure['proposal-tr[H]']


	output_file = f"VIEW/proposal-accuracy.png"
	draw(correct, proposal, output_file)

	report('finish')	
	print('-'*50)


# -------------------------------------------------------------------
if __name__=='__main__':
	main()


