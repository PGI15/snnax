from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx


class Flatten(eqx.Module):
    """
    Simple module to flatten the output of a layer.
    """
    def __call__(self, x, *, key: Optional[jrand.PRNGKey] = None):
        return jnp.reshape(x, -1)

