from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx
from chex import PRNGKey, Array

from .architecture import StatefulModel, GraphStructure, default_forward_fn, ForwardFn
from .layers import StatefulLayer, RequiresStateLayer
from .layers.stateful import StateShape


def gen_feed_forward_struct(num_layers: int) -> GraphStructure:
    """
    Function to construct a simple feed-forward connectivity graph from the
    given number of layers. This means that every layer is just connected to 
    the next one. 

    Arguments:
        `num_layers` (int): Number of layers in the network.
    
    Returns:
        `GraphStructure`: Tuple that contains the input connectivity and input layer ids.
    """
    input_connectivity = [tuple([id]) for id in range(-1, num_layers-1)]
    input_connectivity[0] = tuple([])
    input_layer_ids = [tuple([]) for _ in range(0, num_layers)]
    input_layer_ids[0] = tuple([0])
    return GraphStructure(num_layers, tuple(input_layer_ids), tuple(input_connectivity))


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
                forward_fn: ForwardFn = default_forward_fn) -> None:
        num_layers = len(list(layers))
        graph_struct = gen_feed_forward_struct(num_layers)

        super().__init__(graph_struct, list(layers), forward_fn = forward_fn)

    def __getitem__(self, idx: int) -> eqx.Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __call__(self, state, data, key, **kwargs) -> Tuple[Sequence, Sequence]:
        return super().__call__(state, data, key, **kwargs)


class Parallel(eqx.Module):
    """
    Convenience class to concatenate layers in a spiking neural network in a
    simple manner. The inputs provided as a list in the same order as the
    layers are distributed to each layer. The output is the sum of all layers.
    It supports the defined `StatefulLayer` neuron types as well as equinox
    layers. 

    Arguments:
        `layers`: Sequence containing the equinox modules and snnax stateful
                models of the network order. The order used must be the same as the
                order used in the __call__ function. The output dimensions of layers
                must be broadcastable to the same shape under a sum operation.
    """
    layers: Sequence[eqx.Module]

    def __init__(self, layers: Sequence[eqx.Module]) -> None:
        self.layers = layers

    def __call__(self, inputs, key: Optional[PRNGKey] = None):
        """
        Arguments:
            `inputs`: Sequence containing the inputs to each layer
            `key`: JAX PRNGKey
        """
        h = [l(x) for l,x in zip(self.layers, inputs)]
        return sum(h)


class CompoundLayer(StatefulLayer):
    """
    This class that groups together several Equinox modules. This 
    is useful for convieniently addressing compound layers as a single one.
    It is essentially like an equinox module but with the proper handling 
    of the compound state.
    
    Example:
    `layers = [eqx.Linear(),
               eqx.LayerNorm(),
               snn.LIF()]
    compound_layer = CompoundLayer(*layers)`

    Argument:
        `layers`: Sequence containing the equinox modules and snnax stateful layers
        `init_fn`: Initialization function for the state of the layer
    """

    layers: Sequence[eqx.Module]
    def __init__(self,
                *layers: Sequence[eqx.Module], 
                init_fn: Callable = default_forward_fn) -> None:
        self.layers = layers
        super().__init__(init_fn=init_fn)
        

    def init_state(self,
                   shape: StateShape,
                   key: Optional[PRNGKey] = None) -> Sequence[Array]:
        """
        Arguments:
            `shape`: Shape of the input data
            `key`: JAX PRNGKey
        """
        states = []
        outs = []           
        keys = jrand.split(key, len(self.layers))
        for ilayer, (key, layer) in enumerate(zip(keys, self.layers)):
            # Check if layer is a StatefulLayer
            if isinstance(layer, StatefulLayer):
                state = layer.init_state(shape = shape, key = key)
                out = layer.init_out(shape = shape, key = key)
                states.append(state)
                outs.append(out)
            # This allows the usage of modules from equinox
            # by calculating the output shape with a mock input
            elif isinstance(layer, RequiresStateLayer):
                mock_input = jnp.zeros(shape)
                out = layer(mock_input)
                states.append([out])
                outs.append(out)
            elif isinstance(layer, Parallel):
                out = layer([jnp.zeros(s) for s in shape], key=key)
                states.append([out])
                outs.append(out)
            elif isinstance(layer, eqx.Module):
                out = layer(jnp.zeros(shape), key=key)
                states.append([out])
                outs.append(out)
            else:
                raise ValueError(f"Layer of type {type(layer)} not supported!")
            shape = out.shape

        return states

    def __call__(self, 
                state: Union[Array, Sequence[Array]], 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> Tuple[Sequence, Sequence]:
        """
        Arguments:
            `state`: Sequence containing the state of the compound layer
            `synaptic_input`: Synaptic input to the compound layer
            `key`: JAX PRNGKey
        """
        h = synaptic_input
        keys = jrand.split(key, len(self.layers))
        new_states = []
        outs = []
        for key, old_state, layer in zip(keys, state, self.layers):
            if isinstance(layer, StatefulLayer):
                new_state, h = layer(state=old_state, synaptic_input=h, key=key) 
                new_states.append(new_state)
                outs.append(h)
            elif isinstance(layer, RequiresStateLayer):
                h = layer(synaptic_input=h, state=old_state, key=key) 
                new_states.append([h])
                outs.append(h)
            else:
                h = layer(h, key=key) 
                new_states.append([h])
                outs.append(h)
        return new_states, outs


class SequentialLocalFeedback(Sequential):
    """
    Convenience class to construct a feed-forward spiking neural network with
    self recurrent connections in a simple manner. It supports the defined
    StatefulLayer neuron types as well as equinox layers. Under the hood it
    constructs a connectivity graph with a feed-forward structure and local
    recurrent connections for each layer and feeds it to the StatefulModel class.

    Important: By default, when feedback_layers is None, only CompoundLayer are 
    recurrently connected to themselves. If you want to connect other layers to
    themselves, you need to provide a dictionary with the layer indices as keys
    and the feedback layer indices as values.

    Arguments:
        `layers`: Sequence containing the layers of the network in causal
        order.
        `forward_fn`: forward function used in the scan loop. default forward
        function default_forward_fn used if not provided
        `feedback_layers`: dictionary of which feedback connections to
        create.  If omitted, all CompoundLayers will be connected to themselves
        (= local feedback)
    """

    def __init__(self, 
                *layers: Sequence[eqx.Module],
                forward_fn: ForwardFn = default_forward_fn,
                feedback_layers: dict = None) -> None:

        num_layers = len(list(layers))
        graph_structure = gen_feed_forward_struct(num_layers)
        input_connectivity = [list(i) for i in graph_structure.input_connectivity]
        input_layer_ids = [list(i) for i in graph_structure.input_layer_ids ]



        # TODO this is shown to be dead-end code. Fix linting or replace...
        if feedback_layers is None:
            for i, l in enumerate(layers):
                if isinstance(l, CompoundLayer):
                    input_connectivity[i].append(i)
        else:
            for i, (k, l) in enumerate(feedback_layers.items()):
                input_connectivity[l].append(k)

        # Constructing the connectivity graph
        graph_structure_feedback = GraphStructure(
            num_layers = num_layers,
            input_layer_ids = input_layer_ids,
            input_connectivity = input_connectivity
        )

        StatefulModel.__init__(self,
                               graph_structure=graph_structure,
                               layers=list(layers),
                               forward_fn=forward_fn)

