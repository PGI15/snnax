from typing import Callable, Optional, Sequence, Tuple

import equinox as eqx

from chex import Array, PRNGKey

from ..composed import CompoundLayer
from ..architecture import default_forward_fn


class ResNetBlock(CompoundLayer):
    """
    Residual block for a feed-forward network. It takes a sequence of layers and
    applies the input to the layers and adds the input to the output of the last
    layer. This is a simple residual block implementation.

    Arguments:
        `layers` (eqx.Module): Sequence of layers to be applied
        `init_fn` (Callable): Initialization function for the layers
    """
    def __init__(self,
                *layers: Sequence[eqx.Module], 
                init_fn: Callable = default_forward_fn) -> None:
        self.layers = layers
        super().__init__(init_fn=init_fn)
    
    def __call__(self, 
                states: Sequence[Array], 
                inputs: Array, *, 
                key: Optional[PRNGKey] = None) -> Tuple[Sequence, Sequence]:
        """
        Takes the current timestep state and synaptic input of the layer and generates
        the new state and spike output for the layer.

        Parameters:
            `state` (Sequence[Array]): Contains the state variables of the layer.
            `inputs` (Array): Input values for the layer.
            `key` (Optional[PRNGKey]): Defaults to None. Random number generator key.

        Returns:
            `[0]` (Sequence[Array]): New state of the layer.
            `[1]` (Array): Outputs of the layer.
        """
        assert inputs.shape == states[-1].shape
        new_states, outs = super().__call__(states, inputs, key=key)
        outs += inputs
        # states of all the layers, output of only the last layer.
        return new_states, outs 
                
