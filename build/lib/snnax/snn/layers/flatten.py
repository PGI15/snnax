from typing import Optional

import equinox as eqx
from chex import Array, PRNGKey


class Flatten(eqx.Module):
    """
    Simple module to flatten the output of a layer. The input has to be a numpy
    or jax.numpy array with at least one dimension.
    """
    def __call__(self, 
                x: Array, *, 
                key: Optional[PRNGKey] = None) -> Array:
        return x.flatten()

