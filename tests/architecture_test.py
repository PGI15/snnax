import unittest

import jax
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx
import equinox.nn as nn

import snnax as snx
import snnax.snn as snn
from snnax.snn.architecture import (default_forward_fn, 
                                    debug_forward_fn, 
                                    delayed_forward_fn)


class TestArchitecture(unittest.TestCase):
    def test_default_forward_fn(self):
        print("Testing default_forward_fn and gradient computation...")
        key = jrand.PRNGKey(42)
        keys = jrand.split(key, 4)
        time_steps = 25

        model = snn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), use_bias=False, key=keys[0]),
            snn.LIF([.95, .85], shape=(16, 5, 5), key=keys[1]),

            nn.Conv2d(16, 16, kernel_size=(3, 3), use_bias=False, key=keys[2]),
            snn.LIF([.95, .85], shape=(16, 3, 3), key=keys[3]),

            snn.Flatten(),
            nn.Linear(16*3*3, 5, use_bias=False, key=keys[4]),
            snn.LIF([.95, .85], shape=(5,), key=keys[5]),
            forward_fn=default_forward_fn
        )

        input_spikes = jrand.uniform(key, (32, time_steps, 3, 7, 7))
        input_spikes = jnp.where(input_spikes > 0.3, 1., 0.)

        states = model.init_state((3, 7, 7), key=key)
        batched_model = eqx.filter_vmap(model, in_axes=(None, 0, None))

        def dummy_loss_fn(model, states, input_spikes, key):
            states, out = model(states, input_spikes, key)
            return jnp.sum(out[-1])

        grads = eqx.filter_grad(dummy_loss_fn)(batched_model, states, input_spikes, key)

        self.assertTrue(grads._fun.layers[0].weight.shape == (16, 3, 3, 3))
        self.assertTrue(jnp.abs(grads._fun.layers[0].weight).sum() != 0)

    def test_debug_forward_fn():
        pass

    def test_delayed_forward_fn():
        pass


if __name__ == '__main__':
    unittest.main()


key = jrand.PRNGKey(42)

key1, key2, init_key, key = jrand.split(key, 4)
# Bias is detrimental to model performance
layers = [
    eqx.nn.Linear(16, 16, use_bias=False, key=key1),
    snn.LIF([.95, .85], "superspike"),

    eqx.nn.Linear(16, 2, use_bias=False, key=key2),
    snn.LIF([.95, .85], "superspike")
]

graph = snn.GraphStructure(
    4, [0], [3], [[], [0], [1], [2]]
)

model = snn.StatefulModel(graph, layers)
print(model)

states = model.init_state((16,), key=key)
inputs = jnp.ones((10, 16))

print(model(states, inputs))

seq = snn.Sequential(
    eqx.nn.Linear(16, 16, use_bias=False, key=key1),
    snn.LIF([.95, .85], "superspike"),

    eqx.nn.Linear(16, 2, use_bias=False, key=key2),
    snn.LIF([.95, .85], "superspike")
)
print(seq)

