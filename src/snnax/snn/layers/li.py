from typing import Sequence, Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jrand

from equinox import static_field

from chex import Array, PRNGKey
from .stateful import StatefulLayer


class LI(StatefulLayer):
    """
    Implementation of a simple leaky integrator neuron layer which integrates 
    the synaptic inputs over time.

    Arguments:
        `decay_constants`: Decay constant of the leaky integrator.
        `init_fn`: Function to initialize the initial state of the spiking 
            neurons. Defaults to initialization with zeros if nothing else 
            is provided.
    """
    decay_constant: float = static_field()

    def __init__(self,
                decay_constant: float,
                init_fn: Optional[Callable] = None) -> None:
        super().__init__(init_fn)
        self.decay_constant = decay_constant

    def __call__(self, 
                state: Array, 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> Sequence[Array]:
        alpha = self.decay_constant
        mem_pot = state
        mem_pot = alpha*mem_pot + (1.-alpha)*synaptic_input
        
        return mem_pot, mem_pot

