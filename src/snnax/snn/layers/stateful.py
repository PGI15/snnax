from typing import Callable, Optional, Sequence, Tuple, Union

import jax.numpy as jnp

import equinox as eqx
from chex import Array, PRNGKey


StateShape = Union[Sequence[int], int]

StatefulOutput = Tuple[Sequence[Array], Array]

default_init_fn = lambda x, key, *args, **kwargs: jnp.zeros(x)

class StatefulLayer(eqx.Module):
    """
    Base class to define custom spiking neuron types.

    Arguments:
        `init_fn` (Callable): Initialization function for the state of the layer.
        `shape` (StateShape): Shape of the state.
    """
    init_fn: Callable
    shape: StateShape

    def __init__(self, 
                init_fn: Callable = default_init_fn, 
                shape: Optional[StateShape] = None) -> None:
        self.init_fn = init_fn
        self.shape = shape

    @staticmethod
    def init_parameters(parameters: Union[float, Sequence[float]],
                        shape: StateShape) -> Array:
        # TODO rework this thing
        # if shape is None:
        #     _p = TrainableArray(parameters, requires_grad)
        # else:
        #     if isinstance(parameters[0], Sequence):
        #         assert all([d.shape == shape for d in parameters]), "Shape of decay constants does not match the provided shape"
        #         _p = TrainableArray(_arr, requires_grad)
        #     else:
        #         _arr = jnp.array([jnp.ones(shape, dtype=jnp.float32)*d for d in parameters])
        #         _p = TrainableArray(_arr, requires_grad)
        # return _p
        return parameters if isinstance(parameters, jnp.ndarray) else jnp.array(parameters)

    def init_state(self, shape: StateShape, key: PRNGKey, *args, **kwargs) -> Sequence[Array]:
        return [self.init_fn(shape, key, *args, **kwargs), jnp.zeros(shape)]

    def init_out(self, 
                shape: StateShape, *, 
                key: Optional[PRNGKey] = None) -> Array:
        # The initial ouput of the layer. Initialize as an array of zeros.
        return jnp.zeros(shape)

    def __call__(self, 
                state: Union[Array, Sequence[Array]], 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None):
        pass


class RequiresStateLayer(eqx.Module):
    """
    TODO check if this is obsolete
    Base class to define custom layers that do not have an internal state, 
    but require the previous layer state to compute the output (e.g. pooling).
    """
    def __call__(self, state):
        """
        Outputs:
        output_passed_to_next_layer: [Array]
        """
        raise NotImplementedError

