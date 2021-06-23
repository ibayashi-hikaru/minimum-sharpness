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
def sumup(gaps, sharpness, file_name,title):
	print(f'draw:{file_name}')
	fig = plt.figure(figsize=(4,2.4))
	plt.subplots_adjust(left=0.16, right=0.96, top=0.8, bottom=0.2)

	ratio2color = {i/10:cm.viridis(i/10) for i in range(11)}

	x = defaultdict(list)
	y = defaultdict(list)
	for key in gaps.keys() & sharpness.keys():
		_,ratio = key
		ratio = float(ratio)
		x[ratio].append(sharpness[key])
		y[ratio].append(gaps[key])
		
	ax = fig.add_subplot(1,1,1)
	for ratio in sorted(ratio2color.keys()):
		if ratio not in x: continue
		if ratio not in y: continue
		ax.scatter(x[ratio],y[ratio],s=20,color=ratio2color[ratio], label=f'{ratio}')

	ax.set_xlabel('Normalized Sharpness', fontsize=13)
	ax.set_ylabel('Generalization Gap', fontsize=13)

	legend = ax.legend(title='Shuffled-Label Ratio', title_fontsize=13, loc='upper left', ncol=11, borderaxespad=0.2,columnspacing=-1.7,handletextpad=0.0,labelspacing=0.0,edgecolor='none',facecolor='none',markerscale=0.7,bbox_to_anchor=(0.04, 1.38))
	legend.set_zorder(-1)
	for txt in legend.get_texts():
		txt.set_ha("center")
		txt.set_x(-15) # x-position
		txt.set_y(-20) # y-position	

	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

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
	for selection in ['last-epoch','first-loss','last-loss','first-accuracy','last-accuracy']:
		for model in ['LeNet','MLP']:

			archive = 'ARCHIVEs/train/'
			gaps = {}
			if not os.path.exists(f'{archive}/{model}'): continue
			for seed in os.listdir(f'{archive}/{model}'):
				if not os.path.exists(f'{archive}/{model}/{seed}'): continue
				for ratio in os.listdir(f'{archive}/{model}/{seed}'):

					with open(f'{archive}/{model}/{seed}/{ratio}/trails.pkl','rb') as fp:
						trails = pickle.load(fp)

					f = lambda status : status['train']['accuracy'] - status['test']['accuracy']				
					gaps[seed, ratio] = [f(status) for status in trails]

			archive = 'ARCHIVEs/tsuzuku/'
			buckets = {}
			if not os.path.exists(f'{archive}/{selection}/{model}'): continue
			for seed in os.listdir(f'{archive}/{selection}/{model}'):
				if '.DS' in seed: continue

				with open(f'{archive}/{selection}/{model}/{seed}/selected_index.pkl','rb') as fp:
					selected_index = pickle.load(fp)
				for ratio,index in selected_index.items():
					gaps[seed, ratio] = gaps[seed, ratio][index]

				with open(f'{archive}/{selection}/{model}/{seed}/sharpness.pkl','rb') as fp:
					sharpness = pickle.load(fp)
				for ratio in sharpness:
					buckets[seed, ratio] = sharpness[ratio]
			
			file_name = f'VIEW/model={model}/tsuzuku-sharpness.png'
			sumup(gaps, buckets, file_name, title=f'Model:{model}')

	report('finish')	
	print('-'*50)


# -------------------------------------------------------------------
if __name__=='__main__':
	main()


