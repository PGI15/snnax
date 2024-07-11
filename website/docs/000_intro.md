---
slug: /
---

# Introduction

![image](../static/img/snnax.svg)

<!-- SNNAX is a lightweight library for implementing Spiking Neural Networks (SNN) in JAX which can be used for for simulating and programming at high speed with PyTorch-like intuitiveness and JAX-like performance.
SNNAX models are easily extended and customizable to fit the desired model specifications and target neuromorphic hardware.
Additionally, SNNAX offers key features for optimizing the training and deployment of SNN such as flexible automatic differentiation and just-in-time compilation.

It leverages the excellent and intuitive [Equinox Library](https://docs.kidger.site/equinox/).
The core of SNNAX is a module that stacks layers of pre-defined or custom defined SNNs and Equinox neural network modules, and providing the functions to call them in a single _scan_ loop. This mode of operation enables feedback loops across the layers of SNNs, while leveraging GPU acceleration as much as possible -->

SNNAX is a lightweight library that builds on **Equinox** and **JAX** to provide a spiking neural network (SNN) simulator for deep learning. It is designed to be easy to use and flexible, allowing users to define their own SNN layers while the common deep learning layers are provided by Equinox.
It is fully compatible with JAX and thus can fully leverage JAX' function transformation features like vectorization with `jax.vmap`, automatic differentiation and JIT compilation with XLA.

The following piece of source code demonstrates how to define a simple SNN in SNNAX where we are using [`snnax.snn.Sequential`](./200_architecture/201_composed.md#sequential) class to stack layers of SNNs and Equinox
layers into a feed-forward architecture:

```python
import jax
import jax.numpy as jnp

import equinox as eqx
import snnax.snn as snn

import optax

model = snn.Sequential(eqx.Conv2D(2, 32, 7, 2, key=key1),
                        snn.LIF((8, 8), [.9, .8], key=key2),
                        snn.flatten(),
                        eqx.Linear(64, 11, key=key3),
                        snn.LIF(11, [.9, .8], key=key4))
```

```mermaid
graph LR;
    X-->Conv2D;
    Conv2D-->LIF1[LIF];
    LIF1[LIF]-->Flatten;
    Flatten-->Linear;
    Linear-->LIF2[LIF];
    LIF2[LIF]-->Y;
```

Next, we simply define a loss function for a single sample and then use the vectorization features of JAX to create a batched loss function.
Note that the output of our model is a tuple of membrane potentials and spikes. The spike output is a list of spike trains for each layer of the SNN. For a feed-forward SNN, we can simply take the last element of the spike list, i.e., `out_spikes[-1]`, and sum the spikes across time to get the spike count.count.

```python
# Simple batched loss function

@partial(jax.vmap, in_axes=(0, 0, 0))
def loss_fn(in_states, in_spikes, tgt_class):
out_state, out_spikes = model(in_states, in_spikes)

    # Spikes from the last layer are summed across time
    pred = out_spikes[-1].sum(-1)
    loss = optax.softmax_cross_entropy(pred, tgt_class)
    return loss


# Calculating the gradient with Equinox PyTree filters and
# subsequently jitting the resulting function
@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_and_grad(in_states, in_spikes, tgt_class):
return jnp.mean(loss_fn(in_states, in_spikes, tgt_class))
```

Finally, we train the model by feeding our model the input spike trains and states. For this, we first have to initialize the states of the SNN using the `init_states` method of the [`snnax.snn.Sequential`](./200_architecture/201_composed.md#sequential) class.

```python
# ...
# Simple training loop

for in_spikes, tgt_class in tqdm(dataloader):
    # Initializing the membrane potentials of LIF neurons
    states = model.init_states(key)

    # Jitting with Equinox PyTree filters
    loss, grads = loss_and_grad(states, in_spikes, tgt_class)

    # Update parameter PyTree with Equinox and optax
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
```

Fully worked-out examples can be found in the examples directory.
