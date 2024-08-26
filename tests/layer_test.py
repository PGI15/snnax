import equinox as eqx
import jax.numpy as jnp
import jax.random as jrand
import pytest
import jax

import snnax.snn as snn
from snnax.utils.filter import filter_grad
from snnax.utils.tree import is_decay_constants

import matplotlib.pyplot as plt


@pytest.mark.parametrize("layer", [
    snn.SimpleLIF([0.95]),
    snn.LIF([0.95, 0.85]),
    snn.IAF([0.95], [0.85]),
    snn.SimpleLI([0.95]),
    snn.SigmaDelta([0.7, 0.4, 0.7]),
])
def test_layer(layer):
    key = jrand.PRNGKey(42)
    shape = (16, 16)
    init_state = layer.init_state(shape, key=key)
    assert all(s.shape == shape for s in init_state)

    input_spikes = jrand.uniform(key, shape)
    input_spikes = jnp.where(input_spikes > 0.25, 1., 0.)
    state, output = layer(init_state, input_spikes)
    assert output.shape == shape

    def _test_grad(layer, state, input_spikes):
        state, output = layer(state, input_spikes)
        state, output = layer(state, input_spikes)
        state, output = layer(state, input_spikes)
        return jnp.sum(state[-1])

    grad_fn = eqx.filter_grad(_test_grad)
    grads = grad_fn(layer, init_state, input_spikes)
    assert jnp.sum(grads.decay_constants) != 0

    grad_fn = filter_grad(_test_grad, filter_spec=is_decay_constants)
    grads = grad_fn(layer, init_state, input_spikes)
    assert grads.decay_constants is None


def test_srm():
    pass

