from typing import Callable, Optional, Sequence, Union

from jax.lax import clamp

from chex import Array, PRNGKey
from .stateful import StatefulLayer, StateShape, default_init_fn, StatefulOutput

# TODO implement a version with synaptic current
class SimpleLI(StatefulLayer):
    """
    Implementation of a simple leaky integrator neuron layer which integrates 
    the synaptic inputs over time.

    Arguments:
        `decay_constants` (Array): Decay constant of the leaky integrator.
        `init_fn` (Callable): Function to initialize the initial state of the spiking 
            neurons. Defaults to initialization with zeros if nothing else 
            is provided.
        `shape` (StateShape): Shape of the state of the layer.
    """
    decay_constants: Array

    def __init__(self,
                decay_constants: Array,
                init_fn: Optional[Callable] = default_init_fn,
                shape: Optional[StateShape] = None,
                key: Optional[PRNGKey] = None,) -> None:
        super().__init__(init_fn, shape)
        self.decay_constants = self.init_parameters(decay_constants, shape)

    def init_state(self, 
                    shape: StateShape, 
                    key: PRNGKey, 
                    *args, 
                    **kwargs) -> Sequence[Array]:
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        
        return [init_state_mem_pot]


    def __call__(self, 
                state: Array, 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> StatefulOutput:
        alpha = clamp(0.5, self.decay_constants[0], 1.0)
        mem_pot = state[0]
        mem_pot = alpha*mem_pot + (1.-alpha)*synaptic_input
        
        return [mem_pot], mem_pot

