from typing import Optional, Union

import equinox as eqx
from equinox import static_field
from chex import Array, PRNGKey

from ...functional.surrogate import superspike_surrogate, SpikeFn


class SpikingMaxPool1d(eqx.nn.MaxPool1d):
    """
    Simple implementation of a spiking max pooling layer in two dimensions.
    Instead of the max pooling operation, the layer emits a spike if the maximum
    value of the pooling window is above a certain threshold.

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


class SpikingMaxPool2d(eqx.nn.MaxPool2d):
    """
    Simple implementation of a spiking max pooling layer in two dimensions.
    Instead of the max pooling operation, the layer emits a spike if the maximum
    value of the pooling window is above a certain threshold.

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

