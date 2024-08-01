from typing import Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient, clamp

from chex import Array, PRNGKey
from .stateful import StatefulLayer, default_init_fn, StateShape, StatefulOutput
from ...functional.surrogate import superspike_surrogate, SpikeFn


class SimpleIAF(StatefulLayer):
    """
    Simple implementation of a layer of integrate-and-fire neurons 
    which does not make explicit use of synaptic currents.
    It integrates the raw synaptic input without any decay.
    Requires one constant to simulate constant membrane potential leak.

    Arguments:
        `leak` (Array): Describes the constant leak of the membrane potential.
            Defaults to zero, i.e. no leak.
        `threshold` (Array): Spike threshold for membrane potential. Defaults to 1.
        `spike_fn` (SpikeFn): Spike treshold function with custom surrogate 
            gradient.
        `stop_reset_grad` (bool): Boolean to control if the gradient is 
            propagated through the refectory potential.
        `reset_val` (Array): Reset value of the membrane potential after a spike 
            has been emitted. Defaults to None.
        `init_fn` (Callable): Function to initialize the state of the spiking neurons.
            Defaults to initialization with zeros if nothing else is provided.
    """
    leak: Array
    threshold: Array
    spike_fn: SpikeFn
    stop_reset_grad: bool
    reset_val: Optional[Array]

    def __init__(self,
                leak: Array = 0.,
                threshold: Array = 1.,
                spike_fn: SpikeFn = superspike_surrogate(10.), 
                stop_reset_grad: bool = True,
                reset_val: Optional[Array] = None,
                shape: Optional[StateShape] = None,
                init_fn: Optional[Callable] = default_init_fn) -> None:
        super().__init__(init_fn, shape)
        
        self.threshold = threshold
        self.spike_fn = spike_fn
        self.stop_reset_grad = stop_reset_grad
        self.shape = shape
        self.leak = self.init_parameters(leak, shape)
        self.reset_val = reset_val if reset_val is not None else None

    def __call__(self, 
                state: Array, 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> StatefulOutput:

        mem_pot, spike_output = state
        mem_pot = (mem_pot-self.leak) + synaptic_input
        spike_output = self.spike_fn(mem_pot-self.threshold)
        
        if self.reset_val is None:
            reset_pot = mem_pot*spike_output
        else:
            reset_pot = self.reset_val*spike_output
        # optionally stop gradient propagation through refectory potential       
        refectory_pot = stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_pot

        state = [mem_pot, spike_output]
        return state, spike_output


class IAF(StatefulLayer):
    """
    Implementation of an integrate-and-fire neuron with a constant leak
    or no leak at all. However, it has no leak by default.

    Arguments:
        `decay_constants`: Decay constants for the IAF neuron.
        `spike_fn`: Spike treshold function with custom surrogate gradient.
        `leak`: Describes the constant leak of the membrane potential.
            Defaults to zero, i.e. no leak.
        `threshold`: Spike threshold for membrane potential. 
                        Defaults to 1.
        `reset_val`: Reset value after a spike has been emitted. 
                        Defaults to None.
        `stop_reset_grad`: Boolean to control if the gradient is propagated
                            through the refectory potential.
        `init_fn`: Function to initialize the state of the spiking neurons.
                    Defaults to initialization with zeros if nothing 
                    else is provided.
    """
    decay_constants: Union[Sequence[float], Array]
    leak: Array
    threshold: Array
    spike_fn: SpikeFn
    stop_reset_grad: bool
    reset_val: Optional[Array]

    def __init__(self,
                decay_constants: Union[Sequence[float], Array],
                leak: Array = 0.,
                threshold: Array = 1.,
                spike_fn: Callable = superspike_surrogate(10.),
                stop_reset_grad: bool = True,
                reset_val: Optional[Array] = None,
                init_fn: Optional[Callable] = default_init_fn,
                shape: Optional[StateShape] = None) -> None:
        super().__init__(init_fn, shape)
        
        # TODO assert for numerical stability 0.999 leads to errors...
        self.threshold = threshold
        self.leak = self.init_parameters(leak, shape)
        self.decay_constants = self.init_parameters(decay_constants, shape)
        self.spike_fn = spike_fn
        self.reset_val = reset_val if reset_val is not None else None
        self.stop_reset_grad = stop_reset_grad

    def init_state(self, 
                   shape: StateShape, 
                   key: PRNGKey, 
                   *args, 
                   **kwargs) -> Sequence[Array]:
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        
        # The synaptic currents are initialized as zeros.
        init_state_syn_curr = jnp.zeros(shape) 
        
         # The spiking outputs are initialized as zeros
        init_state_spike_output = jnp.zeros(shape)
        return [init_state_mem_pot, init_state_syn_curr, init_state_spike_output]

    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> StatefulOutput:
        mem_pot, syn_curr, spiking_output = state

        beta  = clamp(0.5, self.decay_constants[0], 1.0)
        
        mem_pot = (mem_pot - self.leak) + syn_curr
        syn_curr = beta*syn_curr + (1.-beta)*synaptic_input 
        spike_output = self.spike_fn(mem_pot - self.threshold)
        
        if self.reset_val is None:
            reset_pot = mem_pot*spike_output
        else:
            reset_pot = self.reset_val*spike_output
            
        # optionally stop gradient propagation through refactory potential       
        refectory_pot = stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_pot

        state = [mem_pot, syn_curr, spike_output]
        return state, spike_output

