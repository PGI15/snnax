import unittest

import jax
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx

import snnax.snn as snn
from snnax.utils.tree import is_decay_constants
from snnax.utils.filter import filter_grad


class TestLayer(unittest.TestCase):
    def test_simple_lif(self):
        key = jrand.PRNGKey(42)
        layer = snn.SimpleLIF([0.95])
        shape = (16, 16)
        init_state = layer.init_state(shape, key=key)
        self.assertTrue(all(s.shape == shape for s in init_state))

        input_spikes = jrand.uniform(key, shape)
        input_spikes = jnp.where(input_spikes > 0.25, 1., 0.)
        state, output = layer(init_state, input_spikes)
        self.assertTrue(output.shape == shape)

        def _test_grad(layer, state, input_spikes):
            state, output = layer(state, input_spikes)
            state, output = layer(state, input_spikes)
            state, output = layer(state, input_spikes)
            return jnp.sum(state[-1])
        
        grad_fn = eqx.filter_grad(_test_grad)
        grads = grad_fn(layer, init_state, input_spikes)
        self.assertTrue(jnp.sum(grads.decay_constants) != 0)

        grad_fn = filter_grad(_test_grad, filter_spec=is_decay_constants)
        grads = grad_fn(layer, init_state, input_spikes)
        self.assertTrue(grads.decay_constants is None)

    def test_lif(self):
        key = jrand.PRNGKey(42)
        layer = snn.LIF([0.95, 0.85])
        shape = (16, 16)
        init_state = layer.init_state(shape, key=key)
        self.assertTrue(all(s.shape == shape for s in init_state))

        input_spikes = jrand.uniform(key, shape)
        input_spikes = jnp.where(input_spikes > 0.25, 1., 0.)
        state, output = layer(init_state, input_spikes)
        self.assertTrue(output.shape == shape)

        def _test_grad(layer, state, input_spikes):
            state, output = layer(state, input_spikes)
            state, output = layer(state, input_spikes)
            state, output = layer(state, input_spikes)
            return jnp.sum(state[-1])
        
        grad_fn = eqx.filter_grad(_test_grad)
        grads = grad_fn(layer, init_state, input_spikes)
        self.assertTrue(jnp.sum(grads.decay_constants) != 0)



        grad_fn = filter_grad(_test_grad, filter_spec=is_decay_constants)

        grads = grad_fn(layer, init_state, input_spikes)
        self.assertTrue(grads.decay_constants is None)

    def test_iaf(self):
        key = jrand.PRNGKey(42)
        layer = snn.IAF([0.95], [0.85])
        shape = (16, 16)
        init_state = layer.init_state(shape, key=key)
        self.assertTrue(all(s.shape == shape for s in init_state))

        input_spikes = jrand.uniform(key, shape)
        input_spikes = jnp.where(input_spikes > 0.25, 1., 0.)
        state, output = layer(init_state, input_spikes)
        self.assertTrue(output.shape == shape)

        def _test_grad(layer, state, input_spikes):
            state, output = layer(state, input_spikes)
            state, output = layer(state, input_spikes)
            state, output = layer(state, input_spikes)
            return jnp.sum(state[-1])
        
        grad_fn = eqx.filter_grad(_test_grad)
        grads = grad_fn(layer, init_state, input_spikes)
        self.assertTrue(jnp.sum(grads.decay_constants) != 0)

        grad_fn = filter_grad(_test_grad, filter_spec=is_decay_constants)
        grads = grad_fn(layer, init_state, input_spikes)
        self.assertTrue(grads.decay_constants is None)

    def test_li(self):
        key = jrand.PRNGKey(42)
        layer = snn.SimpleLI([0.95])
        shape = (16, 16)
        init_state = layer.init_state(shape, key=key)
        self.assertTrue(all(s.shape == shape for s in init_state))

        input_spikes = jrand.uniform(key, shape)
        input_spikes = jnp.where(input_spikes > 0.25, 1., 0.)
        state, output = layer(init_state, input_spikes)
        self.assertTrue(output.shape == shape)

        def _test_grad(layer, state, input_spikes):
            state, output = layer(state, input_spikes)
            state, output = layer(state, input_spikes)
            state, output = layer(state, input_spikes)
            return jnp.sum(state[-1])
        
        grad_fn = eqx.filter_grad(_test_grad)
        grads = grad_fn(layer, init_state, input_spikes)
        self.assertTrue(jnp.sum(grads.decay_constants) != 0)

        grad_fn = filter_grad(_test_grad, filter_spec=is_decay_constants)
        grads = grad_fn(layer, init_state, input_spikes)
        self.assertTrue(grads.decay_constants is None)
        

    def test_srm(self):
        pass


if __name__ == '__main__':
    unittest.main()