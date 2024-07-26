from typing import Sequence, Union, Callable, Optional

import jax.numpy as jnp

import equinox as eqx
from equinox import static_field
from chex import Array, PRNGKey


class TrainableArray(eqx.Module):
    data: Array
    requires_grad: bool

    def __init__(self, data: Array, requires_grad: bool = True):
        self.data = data
        self.requires_grad = requires_grad


class StatefulLayer(eqx.Module):
    """
    Base class to define custom spiking neuron types.
    """
    init_fn: Callable = static_field()

    def __init__(self, init_fn: Callable = None):
        if init_fn is None:
            init_fn = lambda x, key, *args, **kwargs: jnp.zeros(x)
        self.init_fn = init_fn

    @staticmethod
    def init_parameters(    parameters: Union[float, Sequence[float]], 
                            shape: Optional[Union[int, Sequence[int]]] = None,
                            requires_grad: bool = True):
        if shape is None:
            _p = TrainableArray(parameters, requires_grad)
        else:
            if isinstance(parameters[0], Sequence):
                assert all([d.shape == shape for d in parameters]), "Shape of decay constants does not match the provided shape"
                _p = TrainableArray(_arr, requires_grad)
            else:
                _arr = jnp.array([jnp.ones(shape, dtype=jnp.float32)*d for d in parameters])
                _p = TrainableArray(_arr, requires_grad)
        return _p

    def init_state(self, 
                    shape: Union[int, Sequence[int]], 
                    key: PRNGKey, 
                    *args, 
                    **kwargs):
        return [self.init_fn(shape, key, *args, **kwargs), jnp.zeros(shape)]

    def init_out(self, 
                shape: Union[int, Sequence[int]], *, 
                key: Optional[PRNGKey] = None):
        # The initial ouput of the layer. Initialize as an array of zeros.
        return jnp.zeros(shape)

    def __call__(self, 
                state: Union[Array, Sequence[Array]], 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None):
        pass

