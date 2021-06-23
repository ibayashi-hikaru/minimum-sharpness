 #!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------
import datetime
def report(*args):
	print(datetime.datetime.now().strftime('%Y-%m-%d/%H-%M-%S.%f')+' '+' '.join(map(str,args)))

# ------------------------------------------------------
import os
def main():
	scripts = []
		
	scripts.append('draw-gap-proposal.py')
	scripts.append('draw-gap-tsuzuku.py')

	for script in scripts:
		command = f'python -uO visualize/{script}'
		report(f'command:{command}')
		os.system(command)
		print('~'*50)
	report('finish')

if __name__ == '__main__':
	main()
