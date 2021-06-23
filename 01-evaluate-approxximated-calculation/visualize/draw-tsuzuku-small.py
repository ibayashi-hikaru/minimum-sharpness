#!/usr/bin/env python
# coding: utf-8

import sys


# -------------------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(map(str,args)).replace('\n',''))


# -------------------------------------------------------------------
# -------------------------------------------------------------------
import math
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def sumup(buckets, file_name, version):
	print(f'draw:{file_name}')
	fig = plt.figure(figsize=(4,3.2))
	plt.subplots_adjust(left=0.18, right=0.97, top=0.87, bottom=0.2)
		
	ax = fig.add_subplot(1,1,1)

	setting2index = {}

	count = 0
	for model in ['SmallMLP','SmallCNN']:
		for n in [10,100,1000]:
			for trail in range(10):
				measure = buckets[model, trail, n]

				x,y = [],[]
				for diff, consumed_time in  zip(measure[version]['diffs'], measure[version]['times']):
					y.append(diff)

					if len(x)==0:	x.append(consumed_time)
					else:			x.append(consumed_time+x[-1])

				if model=='SmallMLP':
					label = rf'FNNNs:$n={{{n}}}$'
				if model=='SmallCNN':
					label = rf'CNNs: $n={{{n}}}$'

				if trail==0:
					ax.plot(x,y, lw= 0.5, color=cm.tab10(count), label=label)
				else:
					ax.plot(x,y, lw= 0.5, color=cm.tab10(count))

				ax.scatter(x[0],y[0],s=30,facecolor='none',edgecolor=cm.tab10(count))

			count += 1


	if version=='icml':		title = f'ICML-version'
	if version=='arxiv':	title = f'arXiv-version'
	leg = ax.legend(title=title, ncol=2, loc='upper center', title_fontsize=12, fontsize=10, framealpha=1.0, borderaxespad=0.2, columnspacing=0.5, handletextpad=0.5, handlelength=1.0, markerscale=2,frameon=False,bbox_to_anchor=(0.5, 1.17))

	leg.set_zorder(2)
	for line in leg.get_lines():
		line.set_linewidth(2.0)

	ax.set_yscale('log')
	ax.set_xscale('log')

	ax.set_xlabel('Time (log)', fontsize=13)
	ax.set_ylabel(r'Errors (log)', fontsize=13)

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')


	if version=='icml':
		yticks = [1e+2,1e+4,1e+6]
		ax.set_yticks(yticks)

		ax.set_ylim(2e+1, 1e+8)
		ax.spines['left'].set_bounds(2e+1, 1e+7)
		
		ax.set_xlim(1e-3, 6e+1)		
		ax.spines['bottom'].set_bounds(1e-3, 2e+1)
	if version=='arxiv':
		yticks = [1e-1,1e+1,1e+3]
		ax.set_yticks(yticks)

		ax.set_ylim(1e-1,1e+6)
		ax.spines['left'].set_bounds(1e-1, 1e+5)

		ax.set_xlim(1e-3, 6e+1)
		ax.spines['bottom'].set_bounds(1e-3, 2e+1)

	ax.minorticks_off()

	os.makedirs(os.path.dirname(file_name),exist_ok=True)
	plt.savefig(file_name,dpi=120)
	plt.style.use('default');plt.clf();plt.cla();plt.close()


# -------------------------------------------------------------------
import os
import json
import pickle
import math
from collections import defaultdict
def main():
	report('start draw')

	buckets = {}
	archive = 'ARCHIVEs/tsuzuku/'
	
	for model in ['SmallMLP','SmallCNN']:
		for trail in range(10):
			for n in [10,100,1000]:
				print('load',model,trail,n)
				with open(f'{archive}/{model}/{trail}/{n}/measure.pkl','rb') as fp:
					measure = pickle.load(fp)

				buckets[model, trail, n] = measure

	for version in ['icml','arxiv']:
		file_name = f'VIEW/tsuzuku-sharpness-small-{version}.png'
		sumup(buckets, file_name, version)

	report('finish')	
	print('-'*50)


# -------------------------------------------------------------------
if __name__=='__main__':
	main()


