from typing import Callable, Optional, Sequence, Union

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
from jax.lax import stop_gradient, clamp

from .stateful import StatefulLayer, StateShape, default_init_fn, StatefulOutput
from ...functional.surrogate import superspike_surrogate, SpikeFn
from chex import Array, PRNGKey


class SimpleLIF(StatefulLayer):
    """
    Simple implementation of a layer of leaky integrate-and-fire neurons 
    which does not make explicit use of synaptic currents.
    Requires one decay constant to simulate membrane potential leak.
    
    Attributes:
        `decay_constants` (snnaxArray): Decay constant of the simple LIF neuron.
        `spike_fn` (Array): Spike treshold function with custom surrogate gradient.
            Defaults to snnax.functional.surrogate.superspike_surrogate(10.).
        `threshold` (thresholdDtype): Spike threshold for membrane potential. Defaults to 1.
        `reset_val` (Array): Reset value after a spike has been emitted. Defaults to None.
        `stop_reset_grad` (bool): Boolean to control if the gradient is propagated
                        through the refectory potential. Defaults to True.
        `init_fn` (Optional[Callable]): Function to initialize the initial state of the 
                    spiking neurons. Defaults to initialization with zeros.
        `shape` (Optional[StateShape]): Shape of the neuron layer. 
            If given, the parameters will be expanded into vectors and initialized accordingly.
            Defaults to None.
        `key` (Optional[PRNGKey]): Random number generator key for initialization of parameters.
            Defaults to None.
    """
    decay_constants: Union[Sequence[float], Array] 
    threshold: Union[float, Array]
    spike_fn: SpikeFn
    stop_reset_grad: bool
    reset_val: Optional[Array]

    def __init__(self,
                decay_constants: Union[Sequence[float], Array],
                spike_fn: SpikeFn = superspike_surrogate(10.),
                threshold: Union[float, Array] = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[Array] = None,
                init_fn: Optional[Callable] = default_init_fn,
                shape: Optional[StateShape] = None,
                key: Optional[PRNGKey] = None,
                **kwargs) -> None:

        super().__init__(init_fn, shape)
        self.threshold = threshold
        self.spike_fn = spike_fn
        self.reset_val = reset_val
        self.stop_reset_grad = stop_reset_grad
        self.decay_constants = self.init_parameters(decay_constants, shape)

    def __call__(self, 
                state: Array, 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> StatefulOutput:
        '''
        Given the state and synaptic input of the layer, generate the new state and 
        spike outputs.

        Parameters:
            `state` (StateShape): A sequence of arrays composed of membrane potential 
                values at position 0 and spike outputs at position 1.
            `synaptic_input` (Array): An array of input values to the SimpleLIF layer.
            `key` (Optional[PRNGKey]): Random number generator key. Defaults to None.

        Returns:
            (StatefulOutput): A list that has the new state array at position 0 and SimpleLIF layer
                spike outputs at position 1.
        '''
        alpha = lax.clamp(0.5, self.decay_constants[0], 1.0)
        mem_pot, spike_output = state
        mem_pot = alpha*mem_pot + (1.-alpha)*synaptic_input
        spike_output = self.spike_fn(mem_pot-self.threshold)

        if self.reset_val is None:
            reset_pot = mem_pot*spike_output
        else:
            reset_val = jnn.softplus(self.reset_val)
            reset_pot = reset_val * spike_output
            
        # Optionally stop gradient propagation through refectory potential       
        refectory_potential = stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_potential
        
        state = [mem_pot,  spike_output]
        return [state, spike_output]


class LIF(StatefulLayer):
    """
    Implementation of a layer composed of Leaky Integrate and Fire neurons 
    as given in https://arxiv.org/abs/1901.09948. Input state variables for a 
    Leaky Integrate and Fire neuron is 'membrane potential', 'synaptic current' 
    and 'spike outputs'. Each LIF neuron also requires decay constants named 
    'alpha' and 'beta'. 'alpha' controls the decay of membrane potential and 
    synaptic current for generation of new membrane potential value. 'beta' 
    controls the decay of synaptic current and synaptic input for generation of 
    the new synaptic current. In addition, a differentiable spike generation function 
    'spike_fn' is required.

    Arguments:

        `decay_constants` (Union[Sequence[float], Array]): Decay constants for the LIF neuron,
            which are alpha and beta.
        `spike_fn` (SpikeFn): Spike treshold function with custom surrogate gradient. 
            Defaults to snnax.functional.surrogate.superspike_surrogate(10.).
        `threshold` (Union[float, Array]): Spike threshold for membrane potential. Defaults to 1.
        `reset_val` (Optional[Array]): Reset value after a spike has been emitted. 
                        Defaults to None.
        `stop_reset_grad` (bool): Boolean to control if the gradient is propagated
                            through the refectory potential. Defaults to True.
        `init_fn` (Optional[Callable]): Function to initialize the state of the spiking neurons.
                    Defaults to initialization with zeros.
        `shape` (Optional[StateShape]): Shape of the neuron layer. Defaults to None.
        `key` (Optional[PRNGKey]): Random number generator key for initialization of parameters.
            Defaults to None.
    """
    decay_constants: Union[Sequence[float], Array]
    threshold: Union[float, Array]
    spike_fn: SpikeFn
    reset_val: Array
    stop_reset_grad: bool

    def __init__(self, 
                decay_constants: Union[Sequence[float], Array],
                spike_fn: SpikeFn = superspike_surrogate(10.),
                threshold: Union[float, Array] = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[Array] = None,
                init_fn: Optional[Callable] = default_init_fn,
                shape: Optional[StateShape] = None,
                key: Optional[PRNGKey] = None) -> None:

        super().__init__(init_fn, shape)
        self.threshold = threshold
        self.spike_fn = spike_fn
        self.reset_val = reset_val
        self.stop_reset_grad = stop_reset_grad
        self.decay_constants = self.init_parameters(decay_constants, shape)

    def init_state(self, 
                    shape: StateShape, 
                    key: PRNGKey, 
                    *args, 
                    **kwargs) -> Sequence[Array]:
        '''
        Initialize and return the state variables membrane potential, synaptic current and 
        spike output for the layer.

        Parameters:
            `shape` (StateShape): Shape of the neuron layer.
            `key` (PRNGKey): Random number generator key for initialization of parameters.
            `*args`
            `**kwargs`

        Returns:
            (Sequence[Array]): Contains the initial membrane potential, synaptic current 
                and spike output state variables consecutively.
        '''
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        
        # The synaptic currents are initialized as zeros
        init_state_syn_curr = jnp.zeros(shape) 
        
         # The spiking outputs are initialized as zeros
        init_state_spike_output = jnp.zeros(shape)
        return [init_state_mem_pot, init_state_syn_curr, init_state_spike_output]

    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array,
                *, key: Optional[PRNGKey] = None) -> StatefulOutput:
        '''
        Given the state and synaptic input of the layer, generate the new state and 
        spike outputs.

        Parameters:
            `state` (Sequence[Array]): Contains membrane potential, synaptic current and 
                spike output state variables consecutively.
            `synaptic_input` (Array): Input values for the layer.
            `key` (Optional[PRNGKey]): Random number generator key. Defaults to None.

        Returns:
            (Sequence[Array]): Contains the new state at position 0 and layer spike outputs 
                at position 1.
        '''
        mem_pot, syn_curr, spike_output = state
        
        if self.reset_val is None:
            reset_pot = mem_pot*spike_output 
        else:
            reset_pot = (mem_pot-self.reset_val)*spike_output 

        # Optionally stop gradient propagation through refectory potential       
        refectory_potential = stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_potential

        alpha = clamp(0.5, self.decay_constants[0], 1.0)
        beta  = clamp(0.5, self.decay_constants[1], 1.0)
        
        mem_pot = alpha*mem_pot + (1.-alpha)*syn_curr
        syn_curr = beta*syn_curr + (1.-beta)*synaptic_input

        spike_output = self.spike_fn(mem_pot - self.threshold)

        state = [mem_pot, syn_curr, spike_output]
        return [state, spike_output]


class LIFSoftReset(LIF):
    """
    Similar to LIF but reset is additive (relative) rather than absolute.
    For the neurons that spike, reset potential is subtracted from the membrane
    potential. Otherwise, the membrane potential does not change prior to the 
    calculation of its value for the next timestep.
    
    If the neurons spikes: 
    $V \rightarrow V_{reset}$
    where $V_{reset}$ is the parameter reset_val
    """
    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array,
                *, key: Optional[PRNGKey] = None) -> StatefulOutput:
        '''
        Given the state and synaptic input of the layer, generate the new state and 
        spike outputs.

        Parameters:
            `state` (Sequence[Array]): Contains membrane potential, synaptic current and
                spike output state variables consecutively.
            `synaptic_input` (Array): Input values for the layer.
            `key` (Optional[PRNGKey]): Random number generator key. Defaults to None.
        
        Returns:
            (Sequence[Array]): Contains the new state at position 0 and layer spike outputs
                at position 1.
        '''
        mem_pot, syn_curr, spike_output = state
        
        if self.reset_val is None:
            reset_pot = spike_output 
        else:
            reset_pot = self.reset_val*spike_output

        # optionally stop gradient propagation through refectory potential       
        refr_pot = stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refr_pot

        alpha = clamp(0.5, self.decay_constants[0], 1.0)
        beta  = clamp(0.5, self.decay_constants[1], 1.0)
 
        mem_pot = alpha*mem_pot + (1.-alpha)*syn_curr
        syn_curr = beta*syn_curr + (1.-beta)*synaptic_input

        spike_output = self.spike_fn(mem_pot - self.threshold)

        state = [mem_pot, syn_curr, spike_output]
        return [state, spike_output]


class AdaptiveLIF(StatefulLayer):
    """
    Implementation of a adaptive exponential leaky integrate-and-fire neuron
    as presented in https://neuronaldynamics.epfl.ch/online/Ch6.S1.html.
    
    Arguments:
        `decay_constants` (Array): Decay constants for the LIF neuron.
        `spike_fn` (SpikeFn): Spike treshold function with custom surrogate gradient.
        `threshold` (Union[float, Array]): Spike threshold for membrane potential. Defaults to 1.
        `reset_val` (Optional[Array]): Reset value after a spike has been emitted. 
                        Defaults to None.
        `stop_reset_grad` (bool): Boolean to control if the gradient is propagated
                        through the refectory potential.
        `init_fn` (Optional[Callable]): Function to initialize the state of the spiking neurons.
            Defaults to initialization with zeros if nothing else is provided.
        `shape` (Optional[StateShape]): Shape of the neuron layer.
        `key` (Optional[PRNGKey]): Random number generator key for initialization of parameters.
    """
    decay_constants: Union[Sequence[float], Array]
    threshold: Union[float, Array]
    ada_step_val: Array 
    ada_decay_constant: Array 
    ada_coupling_var: Array 
    stop_reset_grad: bool
    reset_val: Optional[Array]

    def __init__(self,
                decay_constants: Union[Sequence[float], Array],
                ada_decay_constant: float = [.8] ,
                ada_step_val: float = [1.0],
                ada_coupling_var: float = [.5],
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: Union[float, Array] = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[float] = None,
                init_fn: Optional[Callable] = None,
                shape: Optional[StateShape] = None,
                key: Optional[PRNGKey] = None) -> None:
        super().__init__(init_fn)

        self.threshold = threshold
        self.spike_fn = spike_fn
        self.reset_val = reset_val 
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
        '''
        Initialize and return the state variables membrane potential, adaptive variables 
        and spike output for the layer.

        Parameters:
            `shape` (StateShape): Shape of the neuron layer.
            `key` (PRNGKey): Random number generator key for initialization of parameters.
            `*args`
            `**kwargs`

        Returns:
            (Sequence[Array]): Contains the initial membrane potential, adaptive variables
                and spike output state variables consecutively.
        '''
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        init_state_ada = jnp.zeros(shape)
        init_state_spikes = jnp.zeros(shape)
        return [init_state_mem_pot, init_state_ada, init_state_spikes]

    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array, 
                *, key: Optional[PRNGKey] = None) -> StatefulOutput:
        '''
        Given the state and synaptic input of the layer, generate the new state and 
        spike outputs.

        Parameters:
            `state` (Sequence[Array]): Contains membrane potential, adaptive variables and
                spike output state variables consecutively.
            `synaptic_input` (Array): Input values for the layer.
            `key` (Optional[PRNGKey]): Random number generator key.

        Returns:
            (Sequence[Array]): Contains the new state at position 0 and layer spike outputs
                at position 1.
        '''
        mem_pot, ada_var = state

        alpha = clamp(0.5,self.decay_constants[0],1.)
        beta = clamp(0.5, self.ada_decay_constant[0], 1.) 
        a = clamp(-1.,self.ada_coupling_var[0], 1.)
        b = clamp(0.,self.ada_step_val[0], 2.)

        # Calculation of the membrane potential
        mem_pot = alpha*mem_pot + (1.-alpha)*(synaptic_input+ada_var)
        spike_output = self.spike_fn(mem_pot - self.threshold)
        
        # Calculation of the adaptive part of the dynamics
        ada_var_new = (1.-beta)*a * mem_pot \
                    + beta*ada_var - b*stop_gradient(spike_output)

        if self.reset_val is None:
            reset_pot = mem_pot*spike_output
        else:
            reset_pot = self.reset_val * spike_output
            
        # Optionally stop gradient propagation through refectory potential       
        refectory_pot = stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_pot

        state = [mem_pot, ada_var_new, spike_output]
        return [state, spike_output]

