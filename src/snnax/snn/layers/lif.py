from typing import Sequence, Union, Callable, Optional, Tuple, List

import jax
import jax.lax as lax
import jax.numpy as jnp

from equinox import static_field, Module

from .stateful import StatefulLayer, TrainableArray
from ...functional.surrogate import superspike_surrogate
from chex import Array, PRNGKey



class SimpleLIF(StatefulLayer):
    """
    Simple implementation of a layer of leaky integrate-and-fire neurons 
    which does not make explicit use of synaptic currents.
    Requires one decay constant to simulate membrane potential leak.
    
    Arguments:
        - `decay_constants`: Decay constant of the simple LIF neuron.
        - `spike_fn`: Spike treshold function with custom surrogate gradient.
        - `threshold`: Spike threshold for membrane potential. Defaults to 1.
        - `reset_val`: Reset value after a spike has been emitted.
        - `stop_reset_grad`: Boolean to control if the gradient is propagated
                        through the refectory potential.
        - `init_fn`: Function to initialize the initial state of the 
                    spiking neurons. Defaults to initialization with zeros 
                    if nothing else is provided.
        - 'shape' if given, the parameters will be expanded into vectors and initialized accordingly
        - 'key' used to initialize the parameters when shape is not None
    """
    decay_constants: Union[Sequence[float], Array, TrainableArray] 
    threshold: float = static_field()
    spike_fn: Callable = static_field()
    stop_reset_grad: bool = static_field()
    reset_val: Optional[float] = static_field()

    def __init__(self,
                decay_constants: TrainableArray,
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: float = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[float] = None,
                init_fn: Optional[Callable] = None,
                shape: Union[Sequence[int],int,None] = None,
                key: PRNGKey = jax.random.PRNGKey(0),
                **kwargs) -> None:

        super().__init__(init_fn)
        # TODO assert for numerical stability 0.999 leads to errors...
        self.threshold = threshold
        self.decay_constants = decay_constants
        self.spike_fn = spike_fn
        self.reset_val = reset_val if reset_val is not None else None
        self.stop_reset_grad = stop_reset_grad

        self.decay_constants = self.init_parameters(decay_constants, shape)


    def __call__(self, 
                state: Array, 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> Sequence[Array]:
        alpha = jax.lax.clamp(0.5, self.decay_constants.data[0], 1.0)
        mem_pot = state[0]
        mem_pot = alpha*mem_pot + (1.-alpha)*synaptic_input # TODO with (1-alpha) or without ?
        spike_output = self.spike_fn(mem_pot-self.threshold)

        
        if self.reset_val is None:
            reset_pot = mem_pot*spike_output
        else:
            reset_val = jax.nn.softplus(self.reset_val)
            reset_pot = reset_val * spike_output
        # optionally stop gradient propagation through refectory potential       
        refectory_potential = lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_potential
        


        state = [mem_pot,  spike_output]
        return [state, spike_output]

class LIF(StatefulLayer):
    """
    TODO improve docstring
    Implementation of a leaky integrate-and-fire neuron with
    synaptic currents. Requires two decay constants to describe
    decay of membrane potential and synaptic current.
    """
    decay_constants: TrainableArray
    threshold: float = static_field()
    spike_fn: Callable = static_field()
    reset_val: float = static_field()
    stop_reset_grad: bool = static_field()

    def __init__(self, 
                decay_constants: Union[Sequence[float], Array],
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: float = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[float] = None,
                init_fn: Optional[Callable] = None,
                shape: Union[Sequence[int], int, None] = None,
                key: PRNGKey = jax.random.PRNGKey(0)) -> None:
        """
        Arguments:
            - `shape`: Shape of the neuron layer.
            - `decay_constants`: Decay constants for the LIF neuron.
                - Index 0 describes the decay constant of the membrane potential,
                - Index 1 describes the decay constant of the synaptic current.
            - `spike_fn`: Spike treshold function with custom surrogate gradient.
            - `threshold`: Spike threshold for membrane potential. Defaults to 1.
            - `reset_val`: Reset value after a spike has been emitted. 
                            Defaults to None.
            - `stop_reset_grad`: Boolean to control if the gradient is propagated
                                through the refectory potential.
            - `init_fn`: Function to initialize the state of the spiking neurons.
                        Defaults to initialization with zeros if 
                        nothing else is provided.
            - 'shape' if given, the parameters will be expanded into vectors and initialized accordingly
            - 'key' used to initialize the parameters when shape is not None
        """

        super().__init__(init_fn)
        # TODO assert for numerical stability 0.999 leads to errors...
        self.decay_constants = decay_constants
        self.threshold = threshold
        self.spike_fn = spike_fn
        self.reset_val = reset_val
        self.stop_reset_grad = stop_reset_grad
        self.decay_constants = self.init_parameters(decay_constants, shape)

    def init_state(self, 
                   shape: Union[int, Sequence[int]], 
                   key: PRNGKey, 
                   *args, 
                   **kwargs) -> Sequence[Array]:
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        init_state_syn_curr = jnp.zeros(shape) # The synaptic currents are initialized as zeros
        init_state_spike_output = jnp.zeros(shape) # The synaptic currents are initialized as zeros
        return [init_state_mem_pot, init_state_syn_curr, init_state_spike_output]


    def __call__(self, 
                state: Sequence[jnp.ndarray], 
                synaptic_input: jnp.ndarray,
                *, key: Optional[jax.random.PRNGKey] = None) -> Sequence[jnp.ndarray]:
        mem_pot, syn_curr, spike_output = state[0], state[1], state[2]
        
        if self.reset_val is None:
            reset_pot = mem_pot*spike_output 
        else:
            reset_pot = (mem_pot-self.reset_val)*spike_output 

        # optionally stop gradient propagation through refectory potential       
        refectory_potential = lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_potential

        alpha = jax.lax.clamp(0.5,self.decay_constants.data[0], 1.0)
        beta  = jax.lax.clamp(0.5,self.decay_constants.data[1], 1.0)
        
        mem_pot = alpha*mem_pot + (1.-alpha)*syn_curr
        syn_curr = beta*syn_curr + (1-beta)*synaptic_input

        spike_output = self.spike_fn(mem_pot - self.threshold)

        state = [mem_pot, syn_curr, spike_output]
        return [state, spike_output]
    
class LIFSoftReset(LIF):
    """
    Similar to LIF but reset is additive (relative) rather than absolute:
    If the neurons spikes: 
    $V \rightarrow V_{reset}$
    where $V_{reset}$ is the parameter reset_val
    """
    def __call__(self, 
                state: Sequence[jnp.ndarray], 
                synaptic_input: jnp.ndarray,
                *, key: Optional[jax.random.PRNGKey] = None) -> Sequence[jnp.ndarray]:
        mem_pot, syn_curr, spike_output = state[0], state[1], state[2]
        
        if self.reset_val is None:
            reset_pot = spike_output 
        else:
            reset_pot = self.reset_val*spike_output

        # optionally stop gradient propagation through refectory potential       
        refr_pot = lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refr_pot

        alpha = jax.lax.clamp(0.5,self.decay_constants.data[0], 1.0)
        beta  = jax.lax.clamp(0.5,self.decay_constants.data[1], 1.0)
 
        mem_pot = alpha*mem_pot + (1.-alpha)*syn_curr
        syn_curr = beta*syn_curr + (1-beta)*synaptic_input

        spike_output = self.spike_fn(mem_pot - self.threshold)

        state = [mem_pot, syn_curr, spike_output]
        return [state, spike_output]

class AdaptiveLIF(StatefulLayer):
    """
    Implementation of a adaptive exponential leaky integrate-and-fire neuron
    as presented in https://neuronaldynamics.epfl.ch/online/Ch6.S1.html.
    """
    decay_constants: TrainableArray
    threshold: float = static_field()
    spike_fn: Callable = static_field()
    ada_step_val: TrainableArray 
    ada_decay_constant: TrainableArray 
    ada_coupling_var: TrainableArray 
    stop_reset_grad: bool = static_field()
    reset_val: Optional[float] = static_field()

    def __init__(self,
                decay_constants: float,
                ada_decay_constant: float = [.8] ,
                ada_step_val: float = [1.0],
                ada_coupling_var: float = [.5],
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: float = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[float] = None,
                init_fn: Optional[Callable] = None,
                shape: Union[Sequence[int],int, None] = None,
                key: PRNGKey = jax.random.PRNGKey(0)) -> None:
        """**Arguments**:

        - `decay_constants`: Decay constants for the adaptive LIF neuron.
            - Index 0 describes the decay constant of the membrane potential,
            - Index 1 describes the decay constant of the synaptic current.
        - `spike_fn`: Spike treshold function with custom surrogate gradient.
        - `threshold`: Spike threshold for membrane potential. Defaults to 1.
        - `reset_val`: Reset value after a spike has been emitted. Defaults to None.
        - `stop_reset_grad`: Boolean to control if the gradient is propagated
            through the refectory potential.
        - `init_fn`: Function to initialize the initial state of the spiking neurons.
            Defaults to initialization with zeros if nothing else is provided.
        - 'shape' if given, the parameters will be expanded into vectors and initialized accordingly
        - 'key' used to initialize the parameters when shape is not None
        """

        super().__init__(init_fn)

        # TODO assert for numerical stability 0.999 leads to errors...
        self.threshold = threshold
        self.spike_fn = spike_fn
        self.reset_val = reset_val if reset_val is not None else None
        self.stop_reset_grad = stop_reset_grad


        self.decay_constants = self.init_parameters(decay_constants, shape)
        self.ada_decay_constant = self.init_parameters(ada_decay_constant, shape)
        self.ada_step_val = self.init_parameters(ada_step_val, shape)
        self.ada_coupling_var = self.init_parameters(ada_coupling_var, shape)
       
    def init_state(self, 
                    shape: Union[Sequence[int], int], 
                    key: PRNGKey, 
                    *args, 
                    **kwargs) -> Sequence[Array]:
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        init_state_ada = jnp.zeros(shape)
        init_state_spikes = jnp.zeros(shape)
        return [init_state_mem_pot, init_state_ada, init_state_spikes]

    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> Sequence[Array]:
        mem_pot, ada_var = state[0], state[1]

        alpha = jax.lax.clamp(0.5,self.decay_constants.data[0],1.)
        beta = jax.lax.clamp(0.5, self.ada_decay_constant.data[0], 1.) 
        a = jax.lax.clamp(-1.,self.ada_coupling_var.data[0],1.)
        b = jax.lax.clamp(0.,self.ada_step_val.data[0],2.)

        # Calculation of the membrane potential
        mem_pot = alpha*mem_pot + (1 - alpha)*(synaptic_input+ada_var)
        # membrane_potential_new = alpha*membrane_potential + (1-alpha)*(synaptic_input + ada_var) # TODO with (1-alpha) or without ?
        spike_output = self.spike_fn(mem_pot - self.threshold)
        
        # Calculation of the adaptive part of the dynamics
        ada_var_new = (1.-beta)*a * mem_pot + beta*ada_var - b*lax.stop_gradient(spike_output)

        if self.reset_val is None:
            reset_pot = mem_pot*spike_output
        else:
            reset_pot = self.reset_val * spike_output
            
        # optionally stop gradient propagation through refectory potential       
        refectory_pot = lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_pot

        # state = (membrane_potential_new, ada_var_new)
        state = [mem_pot, ada_var_new, spike_output]
        return [state, spike_output]

