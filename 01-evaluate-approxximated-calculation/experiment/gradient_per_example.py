#!/usr/bin/env python
# coding: utf-8

# ref
# - https://github.com/cybertronai/autograd-hacks
# - https://github.com/jusjusjus/autograd-hacks

import torch

def _is_support_layer(layer):
	return layer.__class__.__name__ in {'Linear', 'Conv2d'}
def _get_layer_type(layer):
	return layer.__class__.__name__

# -------------------------------------------------------
def add_hooks(model):
	handles = []
	for layer in model.modules():
		if _is_support_layer(layer):
			handles.append(layer.register_forward_hook(_capture_forwardprops))
			handles.append(layer.register_backward_hook(_capture_backwardprops))

	model.__dict__.setdefault('autograd_hacks_hooks', []).extend(handles)

def remove_hooks(model):
	clear_forward1(model)
	clear_backword1(model)
	if hasattr(model, 'autograd_hacks_hooks'):
		for handle in model.autograd_hacks_hooks:
			handle.remove()
		del model.autograd_hacks_hooks

# [0]する理由はtupleが渡されるため以下を参照のこと
# 詳細は以下を参照のこと
#  https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
def _capture_forwardprops(layer, inputs, _outputs):
	layer._forwardprop1		= inputs[0].detach()

def _capture_backwardprops(layer, _grad_inputs, grad_outputs):
	layer._backwardprops1	= grad_outputs[0].detach()

# -------------------------------------------------------
def _compute_grad1_for_linear(layer, Fp, Bp):
	layer.weight.grad1 = torch.einsum('ni,nj->nij', Bp, Fp)
	if layer.bias is not None:
		layer.bias.grad1 = Bp

def _compute_grad1_for_conv2d(layer, Fp, Bp):
	n = Fp.shape[0]
	Fp = torch.nn.functional.unfold(Fp, layer.kernel_size, layer.dilation, layer.padding, layer.stride)
	Bp = Bp.reshape(n, -1, Fp.shape[-1])
	grad1 = torch.einsum('ijk,ilk->ijl', Bp, Fp)
	shape = [n] + list(layer.weight.shape)
	grad1 = grad1.reshape(shape)
	layer.weight.grad1 = grad1
	if layer.bias is not None:
		layer.bias.grad1 = torch.sum(Bp, dim=2)

def compute_grad1(model):
	for layer in model.modules():
		if _is_support_layer(layer):
			assert hasattr(layer, '_forwardprop1'), 	'no _forwardprop1.   run forward  after add_hooks(model)'
			assert hasattr(layer, '_backwardprops1'),   'no _backwardprops1. run backward after add_hooks(model)'

			layer_type = _get_layer_type(layer)
			Fp = layer._forwardprop1
			Bp = layer._backwardprops1

			if layer_type=='Linear':
				_compute_grad1_for_linear(layer, Fp, Bp)
			if layer_type=='Conv2d':
				_compute_grad1_for_conv2d(layer, Fp, Bp)

# -------------------------------------------------------
def clear_forward1(model):
	for layer in model.modules():
		if hasattr(layer, '_forwardprop1'):
			del layer._forwardprop1

def clear_backword1(model):
	for layer in model.modules():
		if hasattr(layer, '_backwardprops1'):
			del layer._backwardprops1


#
