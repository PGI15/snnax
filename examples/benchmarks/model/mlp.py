#!/bin/python
#-----------------------------------------------------------------------------
# File Name : model_mlp.py
# Author: Emre Neftci
#
# Creation Date : Wed 01 Mar 2023 12:11:51 PM CET
# Last Modified : Mon 18 Mar 2024 04:33:18 PM CET
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from .common import *


class MLP(eqx.Module):
    layers: eqx.nn.MLP
    width_size: int
    depth: int
    init_key: int

    def __init__(self, in_size, out_size=10, width_size=100, depth=3, act_fun = jax.nn.relu, norm=False, init_key=None):
        self.width_size = width_size
        self.depth = depth
        self.init_key = init_key
        if type(act_fun) == str:
            act_fun = eval(act_fun)

        # create the layers
        self.layers = []
        for i in range(depth):
            if i == 0:
                self.layers.append(eqx.nn.Sequential([eqx.nn.Linear(np.prod(in_size), width_size, key=self.init_key),
                                                      eqx.nn.LayerNorm(width_size, elementwise_affine=False) if norm else eqx.nn.Identity(),
                                                      eqx.nn.Lambda(act_fun)]))
            elif i < depth -1:
                self.layers.append(eqx.nn.Sequential([eqx.nn.Linear(width_size, width_size, key=self.init_key),
                                                      eqx.nn.LayerNorm(width_size, elementwise_affine=False) if norm else eqx.nn.Identity(),
                                                      eqx.nn.Lambda(act_fun)]))
            else:
                self.layers.append(eqx.nn.Sequential([eqx.nn.Linear(width_size, out_size, key=self.init_key)]))

    def __len__(self):
        return len(self.layers)

    def __call__(self, x, key: jax.random.PRNGKey):
        if len(x.shape)>1:
            x = einops.rearrange(x, "c h w -> (c h w)")        
        for layer in self.layers:
            x = layer(x, key=key)
        return x


    def intermediates(self, x, key: jax.random.PRNGKey):
        if len(x.shape)>1:
            x = einops.rearrange(x, "c h w -> (c h w)")
        intermediate_outputs = []
        h = x
        for layer in self.layers:
            h = layer(jax.lax.stop_gradient(h), key=key)
            intermediate_outputs.append(h)
        return intermediate_outputs

    def local_predictions(self, x, targets, key: jax.random.PRNGKey):
        if len(x.shape)>1:
            x = einops.rearrange(x, "c h w -> (c h w)")
        preds = []
        targets = [x] + targets
        for i, feature in enumerate(self.layers):
            preds.append(feature(jax.lax.stop_gradient(targets[i]), key=key))
        return preds

def mlp_init(model, init_key = None):
    if init_key is None:
        init_key = jax.random.PRNGKey(0)

    w_key, b_key = jax.random.split(init_key,2)
    kaiming_init_fn = lambda leaf: jax.nn.initializers.kaiming_normal()(w_key, leaf.shape, leaf.dtype)
    zeros_init_fn = lambda leaf: jax.nn.initializers.zeros(b_key, leaf.shape, leaf.dtype)
    ones_init_fn = lambda leaf: jax.nn.initializers.ones(b_key, leaf.shape, leaf.dtype)

    model = apply_to_tree_leaf_bytype(model, eqx.nn.Linear, 'weight', kaiming_init_fn)
    model = apply_to_tree_leaf_bytype(model, eqx.nn.Linear, 'weight', kaiming_init_fn)
    
    filter_spec = jax.tree_util.tree_map(lambda _: False, model)

    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.Linear, 'weight', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.Linear, 'bias', lambda _: True)
    return model, filter_spec


def mlp(input_size, out_channels, key, dt=None, **kwargs):
    return mlp_init(MLP(input_size, out_channels, init_key = key, **kwargs))
