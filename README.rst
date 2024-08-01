.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/snnax.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/snnax
    .. image:: https://img.shields.io/coveralls/github/<USER>/snnax/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/snnax
    .. image:: https://img.shields.io/pypi/v/snnax.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/snnax/
    .. image:: https://img.shields.io/conda/vn/conda-forge/snnax.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/snnax
    .. image:: https://pepy.tech/badge/snnax/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/snnax
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/snnax


.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
     :alt: Project generated with PyScaffold
     :target: https://pyscaffold.org/

.. image:: https://readthedocs.org/projects/snnax/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://pgi15.github.io/snnax/

.. image:: https://img.shields.io/pypi/v/snnax.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/snnax/


.. image:: logo/snnax.svg
   :width: 200px
   :height: 100px
   :scale: 60 %


SNNAX is a lightweight library for implementing Spiking Neural Networks (SNNs) 
is JAX. It leverages the excellent and intuitive 
`equinox <https://docs.kidger.site/equinox/>`_.
The full documentation of ``snnax`` can be found at https://pgi15.github.io/snnax/.


Installation
============

You can install SNNAX from PyPI using pip:


.. code-block:: bash

    pip install snnax

Or you can install the latest version from GitHub using pip:


.. code-block:: bash

    pip install git+https://github.com/PGI15/snnax

Requires Python 3.9+, JAX 0.4.13+ and Equinox 0.11.1+.


Introduction
============

SNNAX is a lightweight library that builds on Equinox and JAX to provide a
spiking neural network (SNN) simulator for deep learning. It is designed to
be easy to use and flexible, allowing users to define their own SNN layers
while the common deep learning layers are provided by ``equinox``.
It is fully compatible with JAX and thus can fully leverage JAX' function
transformation features like vectorization with ``jax.vmap``, automatic 
differentiationand JIT compilation with XLA.

The following piece of source code demonstrates how to define a simple SNN in SNNAX:
We can use the ``snnax.snn.Sequential`` class to stack layers of SNNs and Equinox 
layers into a feed-forward architecture.


.. code-block:: python
    
    import jax
    import jax.numpy as jnp

    import equinox.nn as nn
    import snnax.snn as snn

    import optax

    model = snn.Sequential(
        nn.Conv2D(2, 32, 7, 2, key=key1),
        snn.LIF((8, 8), [.9, .8], key=key2),
        snn.flatten(),
        nn.Linear(64, 11, key=key3),
        snn.LIF(11, [.9, .8], key=key4)
    )


Next, we simply define a loss function for a single sample and then use the 
vectorization features of JAX to create a batched loss function.
Note that the output of our model is a tuple of membrane potentials and spikes.
The spike output is a list of spike trains for each layer of the SNN.
For out example, we can simply sum the spikes along the time axis to get the spike count.


.. code-block:: python

    # Simple batched loss function
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def loss_fn(in_states, in_spikes, tgt_class):
        out_state, out_spikes = model(in_states, in_spikes)

        # Spikes from the last layer are summed across time
        pred = out_spikes.sum(-1)
        loss = optax.softmax_cross_entropy(pred, tgt_class)
        return loss

    # Calculating the gradient with Equinox PyTree filters and
    # subsequently jitting the resulting function
    @eqx.filter_value_and_grad
    def loss_and_grad(in_states, in_spikes, tgt_class):
        return jnp.mean(loss_fn(in_states, in_spikes, tgt_class))

    # Finally, we update the parameters using a simple optimizer
    @eqx.filter_jit
    def update(model, opt_state, in_spiked, tgt_class):
        # Get gradients
        loss, grads = loss_and_grad(model, in_spikes, tgt_class)

        # Calculate parameter updates using the optimizer
        updates, opt_state = optim.update(grads, opt_state)

        # Update parameter PyTree with Equinox and optax
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss


Finally, we train the model by feeding our model the input spike trains
and states. For this, we first have to initialize the states of the SNN
using the ``init_states``` method of the ``Sequential`` class.


.. code-block:: python

    # ...
    # Simple training loop
    for in_spikes, tgt_class in tqdm(dataloader):
        # Initializing the membrane potentials of LIF neurons
        states = model.init_states(key)
        model, opt_state, loss = update(model, opt_state, states, in_spikes, tgt_class)


Fully worked-out examples can be found in the ``examples`` directory.


Citation
========

If you use SNNAX in your research, please cite the following paper:

.. code-block:: python

    @article{lohoff2024snnax,
        title={{SNNAX}: {S}piking {N}eural {N}etworks in {JAX}},
        author={Lohoff, Jamie and Finkbeiner, Jan and Neftci, Emre},
        journal={TBD},
        year={2024}
    }


JAX Ecosystem
=============

You can find JAX itself under https://github.com/google/jax.

``equinox`` is available under https://github.com/patrick-kidger/equinox.

Other JAX libraries for SNN training:

- ``spyx`` is very fast and built on ``haiku``:  https://github.com/kmheckel/spyx.
- ``slax`` is very fast and built on ``flax``:  https://github.com/kmheckel/spyx.
- ``rA9`` is another library that we have not tested yet: https://github.com/MarkusAI/rA9
- ``jaxsnn`` is a JAX-based library to train SNNs for deployment BrainScalesS2: https://github.com/electronicvisions/jaxsnn
- ``rockpool``` is a JAX-based library to train SNNs for deployment on Xylo: https://rockpool.ai/index.html

