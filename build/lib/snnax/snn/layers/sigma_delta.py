from typing import Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax.lax import clamp, stop_gradient

from chex import PRNGKey, Array

from .stateful import StatefulLayer, StatefulOutput, StateShape
from ...functional.surrogate import superspike_surrogate, SpikeFn


class SigmaDelta(StatefulLayer):
    """
    Implementation of a Sigma-Delta neuron. A Sigma-Delta neuron consists of a Sigma-Decoder
    part and a Delta-Encoder part. 

    Sigma-Decoder accumulates the synaptic inputs of the neuron over the timesteps to generate 
    the sigma-value (i). At each timestep, the previous timestep's sigma-value is decayed with 
    input-decay and synaptic inputs for the current timestep is added to the calculated  value.
    
    
    Delta-Encoder calculates the current timestep activations using the accumulated spikes by the 
    Sigma-Decoder and generates the error value (e) by taking the difference of current timestep 
    activations and previous timestep activations. Then the previous timestep residue value decayed 
    with membrane_decay is added to generate the delta value (i_mem) which is the difference of delta value and neuron spike outputs 
    of previous timestep. This delta-value is used to generate the current timestep spike output 
    of the neuron. Afterwards, a feedback value (s) is calculated by decaying its value at the previous timestep
    with deedback_decay and adding the current timestep spikes. 
    The resource: https://api.semanticscholar.org/CorpusID:67771396.

   
    Arguments:
        `decay_constants` (Union[Sequence[float], Array]): A list that contains 
            input_decay, membrane_decay and feedback_decay parameters. These values
            should be between 0 and 1 and they should have float type.
        `threshold` (Array): Spike threshold for membrane potential. Defaults to 1.
        `spike_fn` (SpikeFn): Spike treshold function with custom surrogate gradient.
        `init_fn` (Callable): Function to initialize the state of the spiking neurons.
            Defaults to initialization with zeros if nothing else is provided.
    """
    decay_constants: Union[Sequence[float], Array] 
    threshold: float
    spike_fn: SpikeFn
    
    def __init__(self, 
                decay_constants: Union[Sequence[float], Array],
                threshold = 1., 
                spike_fn: Callable = superspike_surrogate(10.),
                init_fn: Optional[Callable]=None,
                shape: Optional[StateShape] = None):

        super().__init__(init_fn)
        self.decay_constants = self.init_parameters(decay_constants, shape)
        self.threshold = threshold
        self.spike_fn = spike_fn
        

    def init_state(self, 
                   shape: Union[int, Sequence[int]], 
                   key: Optional[PRNGKey] = None, 
                   *args, 
                   **kwargs) -> Sequence[Array]:
        i = jnp.zeros(shape, dtype=jnp.float32)
        e = jnp.zeros(shape, dtype=jnp.float32)
        i_mem = jnp.zeros(shape, dtype=jnp.float32)
        s = jnp.zeros(shape, dtype=jnp.float32)
        spike_out = jnp.zeros(shape, dtype=jnp.float32)
        return [i, e, i_mem, s, spike_out]
  
    def __call__(self, state: Sequence[Array], 
                 synaptic_input: Array, 
                 *, key: Optional[PRNGKey] = None) -> StatefulOutput:
        
        input_decay, membrane_decay, feedback_decay = self.decay_constants[0], self.decay_constants[1], self.decay_constants[2]
        
        i, e, i_mem, s, spike_out = state

        # Sigma-decoder stage:
        i = i * input_decay + synaptic_input # Sigma value.

        # Delta-encoder stage:
        e = i - s # Error value.
        i_mem = i_mem * membrane_decay + e # Delta value
        spike_out = self.spike_fn(i_mem - self.threshold) # Spike output.
        s = s * feedback_decay + spike_out # Feedback value.

        state = [i, e, i_mem, s, spike_out]


        return state, state[-1]

