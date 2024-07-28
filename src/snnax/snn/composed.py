from typing import Callable, Sequence, Tuple
import equinox as eqx

from .architecture import StatefulModel, GraphStructure, default_forward_fn


class Sequential(StatefulModel):
    """
    Convenience class to construct a feed-forward spiking neural network in a
    simple manner. It supports the defined `StatefulLayer` neuron types as well 
    as equinox layers. Under the hood it constructs a connectivity graph 
    with a feed-forward structure and feeds it to the `StatefulModel` class.

    Arguments:
        `layers` (eqx.Module): Sequence containing the layers of the network in 
            causal order.
    """

    def __init__(self, 
                *layers: Sequence[eqx.Module],
                forward_fn: Callable = default_forward_fn) -> None:
        num_layers = len(list(layers))
        input_connectivity, input_layer_ids = gen_feed_forward_struct(num_layers)

        # Constructing the connectivity graph
        graph_structure = GraphStructure(
            num_layers = num_layers,
            input_layer_ids = input_layer_ids,
            input_connectivity = input_connectivity
        )

        super().__init__(graph_structure, list(layers), forward_fn = forward_fn)

    def __getitem__(self, idx: int) -> eqx.Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __call__(self, state, data, key, **kwargs) -> Tuple[Sequence, Sequence]:
        return super().__call__(state, data, key, **kwargs)


def gen_feed_forward_struct(num_layers: int) -> Sequence[Sequence[int]]:
    """
    Function to construct a simple feed-forward connectivity graph from the
    given number of layers. This means that every layer is just connected to 
    the next one. 

    Arguments:
        `num_layers` (int): Number of layers in the network.
    
    Returns:
        Tuple that contains the input connectivity and input layer ids.
    """
    input_connectivity = [[id] for id in range(-1, num_layers-1)]
    input_connectivity[0] = []
    input_layer_ids = [[] for _ in range(0, num_layers)]
    input_layer_ids[0] = [0]
    return input_connectivity, input_layer_ids

