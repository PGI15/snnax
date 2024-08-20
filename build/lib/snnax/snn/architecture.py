import functools as ft
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx

from chex import Array, PRNGKey
from jaxtyping import PyTree
from .layers.stateful import StatefulLayer, RequiresStateLayer


@dataclass(frozen=True)
class GraphStructure:
    """
    This class contains meta-information about the computational graph.
    It can be used in conjunction with the `StatefulModel` class to construct 
    a computational model.

    Arguments:
        `num_layers` (int): The number of layers we want to have in our model.
        `input_layer_ids` (Sequence[int]): Index of the layers are provided with 
            external input
        `input_connectivity` (Sequence[Sequence[int]]): Specifies how the layers 
            are connected to each other. 
    """
    num_layers: int
    input_layer_ids: Sequence[Sequence[int]]
    input_connectivity: Sequence[Sequence[int]]

    @property
    def output_layer_ids(self):
        return [self.num_layers-1]


ForwardFnOutput = Tuple[Sequence[Array], Sequence[Array]]
ForwardFn = Callable[[Sequence[eqx.Module], 
                    GraphStructure, 
                    PRNGKey, 
                    Sequence[Array], 
                    PyTree], ForwardFnOutput]


def default_forward_fn(layers: Sequence[eqx.Module], 
                        struct: GraphStructure, 
                        key: PRNGKey,
                        states: Sequence[Array], 
                        data: Sequence[Array]) -> ForwardFnOutput:
    """
    Computes the forward pass (via jax.lax.scan) through the layers in a 
    straight-through manner, i.e. every layer takes the input from the last 
    layer at the same time step. The layers are traversed in the order specified 
    by the connectivity graph.

    Arguments:
        `layers`: Specifies layers in our model.
        `struct`: Specifies graph structure
        `states`: States as returned by init_state 
        `data`: Input Sequence data of the model.
        `key`: Random key for the forward pass.
    """
    keys = jrand.split(key, len(layers))
    new_states, new_outs = [], []
    batch = batch if isinstance(data, Sequence) else [data]
    data = data if isinstance(data, Sequence) else [data]

    for ilayer, (key, state, layer) in enumerate(zip(keys, states, layers)):
        # Grab output from nodes for which the connectivity graph 
        # specifies a connection
        inputs, inputs_v = [], []

        # If the node is also a input layer, also append external input
        if ilayer in struct.input_layer_ids:
            inputs.append(batch)
            inputs_v.append(batch)

		# TODO suboptimal solution below, won't generalize to "deeper"" states

        for layer_id in struct.input_connectivity[ilayer]:
            if type(states[layer_id][-1]) == list:
                inputs.append(states[layer_id][-1][-1])
                inputs_v.append(states[layer_id][-1][0])
            else:
                inputs.append(states[layer_id][-1])
                inputs_v.append(states[layer_id][-1])

        # If the layer also gets external input append it as well
        external_inputs = [data[id] for id in struct.input_layer_ids[ilayer]]
        inputs += external_inputs
        inputs_v += external_inputs
        
        if len(inputs) == 1:
            inputs = jnp.concatenate(inputs, axis=0)
        inputs_v = jnp.concatenate(inputs_v, axis=0)

        # Check if layer is a StatefulLayer
        if isinstance(layer, StatefulLayer):
            new_state, new_out = layer(state, inputs, key=key)
            new_states.append(new_state)
            if ilayer == len(layers) - 1:
                new_outs.append(new_out)
        elif isinstance(layer, RequiresStateLayer):
            new_out = layer(inputs_v, key=key)
            new_states.append([new_out])
            if ilayer == len(layers)-1:
                new_outs.append(new_out)            
        else:
            new_out = layer(inputs, key=key)
            new_states.append([new_out])
            if ilayer == len(layers) - 1:
                new_outs.append(new_out)

    return new_states, new_outs 


def delayed_forward_fn(layers: Sequence[eqx.Module], 
                        struct: GraphStructure, 
                        key: PRNGKey,
                        carry: Tuple[Sequence[Array], Sequence[Array]], 
                        batch) -> ForwardFnOutput:
    """TODO add docstring
    This class contains meta-information about the computational graph.
    It can be used in conjunction with the StatefulModel class to construct 
    a computational model.

    Arguments:
        `num_layers`: Specifies the number of layers we want to have in our model.
        `input_layer_ids`: Specifies which layers are provided with external input
        `input_connectivity`: Specifies how the layers are connected to each other. 
    """
    keys = jrand.split(key, len(layers))
    new_states, new_outs = [], []
    snn_states, outs = carry

    batch = batch if isinstance(batch, Sequence) else [batch]

    for layer_id, (key, state, layer) in enumerate(zip(keys, snn_states, layers)):
        # Grab output from nodes for which 
        # the connectivity graph specifies a connection
        inputs = [outs[id] for id in struct.input_connectivity[layer_id]]

        # If the node is also a input layer, also append external input
        external_inputs = [batch[id] for id in struct.input_layer_ids[layer_id]]
        inputs += external_inputs

        inputs = jnp.concatenate(inputs, axis=-1)

        # Check if node is a StatefulLayer
        if isinstance(layer, StatefulLayer):
            state, out  = layer(state, inputs, key=key)
            new_states.append(state)
            new_outs.append(out)
        else:
            out = layer(inputs, key=key)
            new_states.append(None)
            new_outs.append(out)

    new_carry = (new_states, new_outs)
    return new_carry, new_outs


class StatefulModel(eqx.Module):
    """
    Class that allows the creation of custom SNNs with almost arbitrary 
    connectivity defined through a graph structure called the connectivity graph.
    Has to inherit from eqx.Module to be a callable pytree.
    
    Arguments:
        `graph_structure` (GraphStructure): GraphStructure object to specify 
            network connectivity.
        `layers` (Sequence[eqx.Module]): Computational building blocks of the model.
        `forward_fn` (ForwardFn): Evaluation procedure/loop for the model. 
                        Defaults to backprop through time using lax.scan().
    Output:
    """
    graph_structure: GraphStructure
    layers: Sequence[eqx.Module]
    forward_fn: ForwardFn
    loop_fn: Callable

    def __init__(self, 
                graph_structure: GraphStructure, 
                layers: Sequence[eqx.Module],
                forward_fn: ForwardFn = default_forward_fn,
                loop_fn: Callable = lax.scan) -> None:
        super().__init__()

        self.graph_structure = graph_structure
        self.layers = layers
        self.forward_fn = forward_fn
        self.loop_fn = loop_fn

        assert len(layers) == self.graph_structure.num_layers
        assert len(self.graph_structure.input_connectivity) == self.graph_structure.num_layers

    def init_state(self, 
                   in_shape: Union[Sequence[Tuple[int]], Tuple[int]], 
                   key: Optional[PRNGKey] = None) -> Sequence[Array]:
        """
        Init function that recursively calls the init functions of the stateful
        layers. Non-stateful layers are initialized as None and their output
        shape is computed using a mock input.
        
        Arguments:
            `in_shape`: GraphStructure object to specify network topology.
            `key`: Computational building blocks of the model.
        Returns:
        """
        keys = jrand.split(key, len(self.layers))
        states, outs = [], []
        struct = self.graph_structure

        if not isinstance(in_shape, list):
            in_shape_0 = [in_shape]
        else:
            in_shape_0 = in_shape

        for layer_id, (layer, key) in enumerate(zip(self.layers, keys)):
            # Grab output from nodes for which the connectivity graph 
            # specifies a connection
            inputs = [outs[id] for id in struct.input_connectivity[layer_id]]
            
            # If the node is also a input layer, also append external input
            external_inputs = [jnp.zeros(in_shape_0[id]) for id in struct.input_layer_ids[layer_id]]
            inputs += external_inputs

            inputs = jnp.concatenate(inputs, axis=0)

            # Check if layer is a StatefulLayer
            if isinstance(layer, StatefulLayer):
                state = layer.init_state(shape = in_shape, key = key)
                out = layer.init_out(shape = in_shape, key = key)
                in_shape = out.shape
                states.append(state)
                outs.append(out)
            # This allows the usage of modules from equinox
            # by calculating the output shape with a mock input
            # elif isinstance(layer, RequiresStateLayer):
            #     mock_input = jnp.zeros(in_shape)
            #     out = layer(mock_input)
            #     in_shape = out.shape
            #     states.append([out])
            #     outs.append(out)
            elif isinstance(layer, eqx.Module):
                out = layer(inputs, key=key)
                in_shape = out.shape
                states.append([out])
                outs.append(out)
            else:
                raise ValueError(f"Layer of type {type(layer)} not supported!")

        return states

    def __call__(self, 
                input_states: Sequence[Array], 
                input_batch,
                key: PRNGKey,
                burnin: int = 0) -> Tuple:
        # Partial initialization of the forward function
        forward_fn = ft.partial(self.forward_fn, 
                                self.layers, 
                                self.graph_structure,
                                key)       
        
        if burnin > 0:
            new_states, new_outs = self.loop_fn(forward_fn, 
                                                input_states, 
                                                input_batch[:burnin])

            # Performes the actual BPTT when differentiated
            new_states, new_outs = self.loop_fn(forward_fn, 
                                                jax.lax.stop_gradient(new_states), 
                                                input_batch[burnin:])
        else:
            new_states, new_outs = self.loop_fn(forward_fn, 
                                                input_states, 
                                                input_batch)
        return new_states, new_outs         

