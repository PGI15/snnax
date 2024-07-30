from typing import Sequence, Union, Callable, Optional, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx
from equinox import static_field

from .stateful import StatefulLayer
from ...functional.surrogate import superspike_surrogate
from chex import Array, PRNGKey


class BatchNormLayer(eqx.Module):

    gamma: float
    forget_weight: float = static_field()
    eps: float = static_field()

    def __init__(self, eps: Union[float, int], forget_weight: Union[float, int], gamma: Union[float, int] = 0.8):
        super().__init__()
        self.eps = eps
        self.forget_weight = forget_weight
        self.gamma = gamma

        if (forget_weight > 1):
            raise ValueError(f"Forget weight value cannot be greater than 1 for the batch norm layer!")
        if (gamma > 1):
            raise ValueError(f"Gamma value cannot be greater than 1 for the batch norm layer!")
        return 

    def init_state(self, input_shape, key: Optional[PRNGKey] = None):   # initialize the shapes of moving_mean and moving_var
        return jnp.zeros(input_shape), jnp.zeros(input_shape)
    
    def __call__(self, input_data, moving_mean, moving_var, key: Optional[PRNGKey] = None):
        
        mean = lax.pmean(input_data, axis_name="batch_axis")
        var = lax.pmean((input_data - mean) ** 2, axis_name="batch_axis")

        moving_mean = (1. - self.forget_weight) * moving_mean + self.forget_weight * mean
        moving_var = (1. - self.forget_weight) * moving_var + self.forget_weight * var

        outs = (input_data - mean) / jnp.sqrt(var + jnp.full(var.shape, self.eps))
        outs = outs * self.gamma
        
        return moving_mean, moving_var, outs

