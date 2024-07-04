import jax
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx

import snnax as snx
import snnax.snn as snn

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

