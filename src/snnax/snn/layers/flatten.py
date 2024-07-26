from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx
from chex import Array, PRNGKey


class Flatten(eqx.Module):
    """
    Simple module to flatten the output of a layer.
    """
    def __call__(self, 
                x: Array, *, 
                key: Optional[PRNGKey] = None) -> Array:
        return x.flatten()

