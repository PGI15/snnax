import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx
from equinox import static_field

from .stateful import StatefulLayer
from ..layers.batchnorm import BatchNormLayer
from chex import Array, PRNGKey

from typing import Sequence

import snnax.snn as snn


# snn.ResNetBlock(kernel_dims=[[in_channels, out_channels, kernel_size, stride, padding], [in_features, out_features]]) --> one conv, one linear layer

class ResNetBlock(eqx.Module):

    layers: Sequence[eqx.Module]

    def __init__(self, layer_order: Sequence, layer_params: Sequence, stateful_layer_type: str='LIF', key=None):
        # layer_order = ['c', 'c', 'l']
        # layer_params = [  [in_channels, out_channels, kernel_size, stride, padding],    [in_channels, out_channels, kernel_size, stride, padding],    [in_features, out_features] ].
        super().__init__()

        layers = []
        for ilayer, layer in enumerate(layer_order):
            if layer == 'c':
                key1, key = jrand.split(key, 2)
                layers.append(eqx.nn.Conv2d(*layer_params[ilayer], key=key1, use_bias = False))
                layers.append(snn.BatchNormLayer(eps=1e-10, forget_weight=0.2, gamma=0.7))
                if stateful_layer_type == 'LIF':
                    layers.append(snn.LIF([0.9, 0.7]))
                elif stateful_layer_type == 'SigmaDelta':
                    layers.append(snn.SigmaDelta())
            elif layer == 'l':
                key1, key = jrand.split(key, 2)
                layers.append(eqx.nn.Linear(*layer_params[ilayer], key=key1, use_bias=False))
                layers.append(snn.BatchNormLayer(eps=1e-10, forget_weight=0.2, gamma=0.7))
                if stateful_layer_type == 'LIF':
                    layers.append(snn.LIF([0.9, 0.7]))
                elif stateful_layer_type == 'SigmaDelta':
                    layers.append(snn.SigmaDelta())
        self.layers = layers

    def init_state(self, input_shape, key: PRNGKey):
        initial_states = []

        input_shape_cached = input_shape

        keys = jrand.split(key, len(self.layers))

        for layer, keyx in zip(self.layers, keys):
            if isinstance(layer, BatchNormLayer):
                init_moving_mean, init_moving_var = layer.init_state(input_shape)
                init_out = jnp.zeros(input_shape)
                initial_states.append([init_moving_mean, init_moving_var, init_out])
            elif isinstance(layer, StatefulLayer):
                init_state = layer.init_state(input_shape, key=keyx)
                init_out = layer.init_out(input_shape)
                input_shape = init_out.shape
                initial_states.append(init_state)
            elif isinstance(layer, eqx.Module):
                init_out = layer(jnp.zeros(input_shape))
                input_shape = init_out.shape
                initial_states.append([init_out])
        
        if input_shape_cached != input_shape:
            raise ValueError(f'Input shape and the output shape of the ResNet block should match. Got ', input_shape_cached, ' and ', input_shape)

        return initial_states, init_out
    
    def __call__(self, states, inputs, key: PRNGKey):
        
        keys = jrand.split(key, len(self.layers))
        new_states = []

        # states is a list that contains the states of each layer in the resnet block.
        for layer, state, key in zip(self.layers, states, keys):


            if isinstance(layer, BatchNormLayer):
                moving_mean, moving_var, new_out = layer(inputs, state[0], state[1], key=key)
                new_states.append([moving_mean, moving_var, new_out])
                inputs = new_out
            elif isinstance(layer, StatefulLayer):
                new_state, new_out = layer(state, inputs, key=key)
                new_states.append(new_state)
                inputs = new_out
            elif isinstance(layer, eqx.Module):
                new_out = layer(inputs, key=key)
                new_states.append([new_out])
                inputs = new_out
            
        return new_states, new_out # states of all the layers, output of only the last layer.
                
