import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jrand

import snnax.snn as snn
from snnax.snn.composed import gen_feed_forward_struct, default_forward_fn


def test_mlp():
    key = jrand.PRNGKey(42)
    keys = jrand.split(key, 4)
    time_steps = 10

    model = snn.Sequential(
        nn.Linear(16, 16, use_bias=False, key=keys[0]),
        snn.LIF([.95, .85], shape=(16,), key=keys[1]),
        nn.Linear(16, 2, use_bias=False, key=keys[2]),
        snn.LIF([.95, .85], shape=(2,), key=keys[3]),
        forward_fn=default_forward_fn
    )

    input_spikes = jrand.uniform(key, (32, time_steps, 16))
    input_spikes = jnp.where(input_spikes > 0.5, 1., 0.)

    states = model.init_state((16,), key=key)
    batched_model = eqx.filter_vmap(model, in_axes=(None, 0, None))
    states, out = batched_model(states, input_spikes, key)
    assert out[0].shape == (32, time_steps, 2)


def test_convnet():
    key = jrand.PRNGKey(42)
    keys = jrand.split(key, 4)
    time_steps = 10

    model = snn.Sequential(
        nn.Conv2d(3, 16, kernel_size=(3, 3), use_bias=False, key=keys[0]),
        snn.LIF([.95, .85], shape=(16, 5, 5), key=keys[1]),

        nn.Conv2d(16, 16, kernel_size=(3, 3), use_bias=False, key=keys[2]),
        snn.LIF([.95, .85], shape=(16, 3, 3), key=keys[3]),

        snn.Flatten(),
        nn.Linear(16 * 3 * 3, 5, use_bias=False, key=keys[4]),
        snn.LIF([.95, .85], shape=(5,), key=keys[5]),
        forward_fn=default_forward_fn
    )

    input_spikes = jrand.uniform(key, (32, time_steps, 3, 7, 7))
    input_spikes = jnp.where(input_spikes > 0.5, 1., 0.)

    states = model.init_state((3, 7, 7), key=key)
    batched_model = eqx.filter_vmap(model, in_axes=(None, 0, None))
    states, out = batched_model(states, input_spikes, key)
    assert out[0].shape == (32, time_steps, 5)


def test_gen_feed_forward_struct():
    num_layers = 6
    graph_struct = gen_feed_forward_struct(num_layers)

    assert graph_struct.input_layer_ids == ((0), (), (), (), (), ())
    assert graph_struct.input_connectivity == ((), (0), (1), (2), (3), (4))

def test_compound_layer():
    pass
