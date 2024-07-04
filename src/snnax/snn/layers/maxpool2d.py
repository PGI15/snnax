import jax
import jax.numpy as jnp
import equinox as eqx
from .stateful import RequiresStateLayer
from typing import Sequence, Union, Tuple, Any, Callable, Optional
from equinox import static_field
from ...functional.surrogate import superspike_surrogate

class MaxPool2d(eqx.nn.MaxPool2d, RequiresStateLayer):
    """
    Simple module to flatten the output of a layer.
    """
    spike_fn: Callable[[jnp.ndarray], jnp.ndarray] = static_field()
    threshold: Union[float, jnp.ndarray] = static_field()
    
    def __init__(self, *args, spike_fn = superspike_surrogate(10.0), threshold = 1.0, **kwargs, ):
        self.threshold = threshold
        
        self.spike_fn = spike_fn
            
        super().__init__(*args, **kwargs)
    
    def __call__(self, x, key: Optional[jax.random.PRNGKey] = None):
        out = super().__call__(x) 
        return self.spike_fn(out-self.threshold)
    
    
    

