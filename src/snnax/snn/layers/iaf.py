from typing import Sequence, Union, Callable, Optional

import jax
import jax.lax as lax
import jax.numpy as jnp

from equinox import static_field

from chex import Array, PRNGKey
from .stateful import StatefulLayer
from ...functional.surrogate import superspike_surrogate


class SimpleIAF(StatefulLayer):
    """
    Simple implementation of a layer of integrate-and-fire neurons 
    which does not make explicit use of synaptic currents.
    It integrates the raw synaptic input without any decay.
    Requires one constant to simulate constant membrane potential leak.
    """
    leak: float = static_field()
    threshold: float = static_field()
    spike_fn: Callable = static_field()
    stop_reset_grad: bool = static_field()
    reset_val: Optional[float] = static_field()

    def __init__(self,
            spike_fn: Callable = superspike_surrogate(10.), 
            leak: float = 0.,
            threshold: float = 1.,
            stop_reset_grad: bool = True,
            reset_val: Optional[float] = None,
            init_fn: Optional[Callable] = None) -> None:
        """
        Arguments:
            - `spike_fn`: Spike treshold function with custom surrogate gradient.
            - `leak`: Describes the constant leak of the membrane potential.
                Defaults to zero, i.e. no leak.
            - `threshold`: Spike threshold for membrane potential. Defaults to 1.
            - `reset_val`: Reset value after a spike has been emitted. 
                            Defaults to None.
            - `stop_reset_grad`: Boolean to control if the gradient is propagated
                                through the refectory potential.
            - `init_fn`: Function to initialize the state of the spiking neurons.
                        Defaults to initialization with zeros if 
                        nothing else is provided.
        """

        super().__init__(init_fn)
        
        # TODO assert for numerical stability 0.999 leads to errors...
        self.threshold = threshold
        self.leak = leak
        self.spike_fn = spike_fn

        self.reset_val = reset_val if reset_val is not None else None
        self.stop_reset_grad = stop_reset_grad

    def __call__(self, 
                state: Array, 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> Sequence[jnp.ndarray]:

        mem_pot = state
        mem_pot = (mem_pot-self.leak) + synaptic_input
        spike_output = self.spike_fn(mem_pot-self.threshold)
        
        if self.reset_val is None:
            reset_pot = mem_pot*spike_output
        else:
            reset_pot = self.reset_val*spike_output
        # optionally stop gradient propagation through refectory potential       
        refectory_pot = lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_pot

        output = spike_output
        state = mem_pot
        return state, output


class IAF(StatefulLayer):
    """
    Implementation of an integrate-and-fire neuron with a constant leak
    or no leak at all. However, it has no leak by default.
    """
    decay_constants: Union[Sequence[float], Array] = static_field()
    leak: float = static_field()
    threshold: float = static_field()
    spike_fn: Callable = static_field()
    stop_reset_grad: bool = static_field()
    reset_val: Optional[float] = static_field()

    def __init__(self,
                decay_constants: Union[Sequence[float], Array],
                spike_fn: Callable = superspike_surrogate(10.),
                leak: float = 0.,
                threshold: float = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[float] = None,
                init_fn: Optional[Callable] = None) -> None:
        """
        Arguments:
            - `decay_constants`: Decay constants for the IAF neuron.
                - Index 0 describes the decay constant of the membrane potential,
                - Index 1 describes the decay constant of the synaptic current.
            - `spike_fn`: Spike treshold function with custom surrogate gradient.
            - `leak`: Describes the constant leak of the membrane potential.
                Defaults to zero, i.e. no leak.
            - `threshold`: Spike threshold for membrane potential. 
                            Defaults to 1.
            - `reset_val`: Reset value after a spike has been emitted. 
                            Defaults to None.
            - `stop_reset_grad`: Boolean to control if the gradient is propagated
                                through the refectory potential.
            - `init_fn`: Function to initialize the state of the spiking neurons.
                        Defaults to initialization with zeros if nothing 
                        else is provided.
        """

        super().__init__(init_fn)
        
        # TODO assert for numerical stability 0.999 leads to errors...
        self.threshold = threshold
        self.leak = leak
        self.decay_constants = decay_constants
        self.spike_fn = spike_fn
        self.reset_val = reset_val if reset_val is not None else None
        self.stop_reset_grad = stop_reset_grad

    def init_state(self, 
                shape: Union[int, Sequence[int]], 
                key: PRNGKey, 
                *args, 
                **kwargs):
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        init_state_syn_curr = jnp.zeros(shape)
        return jnp.stack([init_state_mem_pot, init_state_syn_curr])

    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> Sequence[Array]:
        mem_pot, syn_curr = state[0], state[1]

        alpha = self.decay_constants[0]
        beta = self.decay_constants[1]
        
        mem_pot = (mem_pot - self.leak) + (1. - alpha)*syn_curr
        # TODO Here the (1. - beta) seems to work fine
        syn_curr = beta*syn_curr + (1. - beta)*synaptic_input 
        spike_output = self.spike_fn(mem_pot - self.threshold)
        
        if self.reset_val is None:
            reset_pot = mem_pot*spike_output
        else:
            reset_pot = self.reset_val*spike_output
            
        # optionally stop gradient propagation through refactory potential       
        refectory_pot = lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_pot

        state = jnp.stack([mem_pot, syn_curr])
        return state, spike_output

