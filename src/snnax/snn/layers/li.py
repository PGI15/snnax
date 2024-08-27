from typing import Callable, Optional, Union, Sequence

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
    decay_constants: Union[Sequence[float], Array]

    def __init__(self,
                decay_constants: Union[Sequence[float], Array],
                init_fn: Optional[Callable] = default_init_fn,
                shape: Optional[StateShape] = None,
                key: Optional[PRNGKey] = None,) -> None:
        super().__init__(init_fn, shape)
        self.decay_constants = self.init_parameters(decay_constants, shape)

    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> StatefulOutput:
        """
        Takes the current layer state variables and synaptic input for the 
        given timestep and generates the layer states and membrane potential values 
        for the next timestep.
        
        Parameters:
            `state` (Sequence[Array]): Contains membrane potential state variable array, 
                which is the only state variable for the Leaky Integrator layer.
            `synaptic_input` (Array): Input values for the layer.
            `key` (Optional[PRNGKey]): Defaults to None. Random number generator key.

        Returns:
            `[0]` (Sequence[Array]): New state of the layer.
            `[1]` (Array): Membrane potential state variable.
        """
        alpha = clamp(0.5, self.decay_constants[0], 1.0)
        mem_pot = state[0]
        mem_pot = alpha*mem_pot + (1.-alpha)*synaptic_input
        
        return [mem_pot], mem_pot

