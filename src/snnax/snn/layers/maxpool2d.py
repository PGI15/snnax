from typing import Callable, Optional, Sequence, Union, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from equinox import static_field
from chex import Array, PRNGKey

from .stateful import RequiresStateLayer
from ...functional.surrogate import superspike_surrogate, SpikeFn


class SpikingMaxPool2d(eqx.nn.MaxPool2d, RequiresStateLayer):
    """
    Simple module to flatten the output of a layer.

    Arguments:
        `threshold` (Union[float, Array]): Threshold of the spiking neuron.
        `spike_fn` (SpikeFn): Surrogate gradient function for the spiking neuron.
    """
    threshold: Union[float, Array] = static_field()
    spike_fn: SpikeFn = static_field()

    def __init__(self, 
                *args,
                threshold: Union[float, Array] = 1.0, 
                spike_fn: SpikeFn = superspike_surrogate(10.0), 
                **kwargs):
        self.threshold = threshold
        self.spike_fn = spike_fn
            
        super().__init__(*args, **kwargs)
    
    def __call__(self, x, key: Optional[PRNGKey] = None):
        out = super().__call__(x) 
        return self.spike_fn(out-self.threshold)

