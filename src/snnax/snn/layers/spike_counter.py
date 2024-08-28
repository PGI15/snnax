#!/bin/python
#-----------------------------------------------------------------------------
# File Name : spike_counter.py
# Author: Emre Neftci
#
# Creation Date : Tue 27 Aug 2024 10:22:23 AM CEST
# Last Modified : Tue 27 Aug 2024 10:43:10 AM CEST
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from typing import Callable, Optional, Sequence, Union

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
from jax.lax import stop_gradient, clamp

from .stateful import StatefulLayer, StateShape, default_init_fn, StatefulOutput
from chex import Array, PRNGKey

class SpikeCounter(StatefulLayer):
    """
    """

    def __init__(self, shape, **kwargs):
        super().__init__(shape = shape)

    def init_state(self, 
                   shape: Union[Sequence[int], int], 
                   key: PRNGKey, 
                   *args, 
                   **kwargs) -> Sequence[Array]:
        init_state_count = jnp.zeros(shape)
        init_state_spikes = jnp.zeros(shape)
        return [init_state_count, init_state_spikes]

        
    def __call__(self, 
                state: Array, 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> StatefulOutput:

        count, spike_output = state
        spike_output = synaptic_input 
        count = count + spike_output 
        
        state = [count, spike_output]
        return state, spike_output
