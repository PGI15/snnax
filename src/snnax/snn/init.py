from typing import Optional, Sequence
import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jrand
import equinox as eqx

from jax.tree_util import tree_leaves
from jax.nn.initializers import zeros, orthogonal

from chex import Array, PRNGKey
from .layers.stateful import StatefulLayer
from ..utils.tree import apply_to_tree_leaf

def max_iters_not_reached(idx: int, max_iters: int) -> bool:
    if max_iters is not None:
        return idx < max_iters
    else:
        return True

def lsuv(model, 
        data: Sequence[Array], 
        key: PRNGKey, 
        tgt_mean: float=-.85, 
        tgt_var: float=1., 
        mean_tol: float=.1, 
        var_tol: float=.1, 
        max_iters: Optional[int] = None):
    """
    LSUV = Layer Sequential Unit Variance
    Initialization inspired from Mishkin D and Matas J. 
    All you need is a good init. arXiv:1511.06422 [cs], February 2016.
    This is a datadriven initialization scheme for possibly stateful 
    networks which adjusts the mean and variance of the membrane output to the 
    given values for each layer. The weights and biases are required to be 
    pytree leaves with attributes `weight` and `bias`.
    It operates on an entire minibatch.
    """
    # TODO maybe write JIT compiled version?
    init_key, key = jrand.split(key)
    shape = data[0,0].shape
    init_states = model.init_state(shape, init_key)

    weight_init_fn = orthogonal()
    w_key, b_key, key = jrand.split(key, 3)
    orthogonal_init_fn = lambda leaf: weight_init_fn(w_key, 
                                                    leaf.shape, 
                                                    leaf.dtype)
    zero_init_fn = lambda leaf: zeros(b_key, leaf.shape, leaf.dtype)

    # Initialize all layers with `weight` or `bias` attribute with 
    # random orthogonal matrices or zeros respectively 
    model = apply_to_tree_leaf(model, "weight", orthogonal_init_fn)
    model = apply_to_tree_leaf(model, "bias", zero_init_fn)

    adjust_var = lambda weight: weight *jnp.sqrt(tgt_var) / jnp.sqrt(jnp.maximum(mem_pot_var, 1e-2))
    adjust_mean_bias = lambda bias: bias -.2*(mem_pot_mean-tgt_mean) 
    # (1. - .2*(mem_pot_mean - tgt_mean)/norm) # TODO Further investigation!!!
    adjust_mean_weight = lambda eps, weight: weight*(1.-eps) 

    spike_layers = [i for i, layer in enumerate(model.layers) if isinstance(layer, StatefulLayer)]
    vmap_model = jax.vmap(model, in_axes=(None, 0, 0))
    alldone = False

    iters = 0
    while not alldone and max_iters_not_reached(iters, max_iters):
        alldone = True
        iters += 1

        keys = jrand.split(key, data.shape[0])
        states, outs = vmap_model(init_states, data, keys)
        spike_layer_idx = 0
        for ilayer, layer in enumerate(model.layers):
            # Filter for layers that have weight attribute

            if isinstance(layer, StatefulLayer) :
                spike_layer_idx += 1
                continue
                
            if not(isinstance(layer, eqx.nn.Conv2d) or isinstance(layer, eqx.nn.Linear)):
                continue


            #handles corner case where lif is not the final layer
            if spike_layer_idx >= len(spike_layers): 
                continue
                
            idx = spike_layers[spike_layer_idx]
            
            # Sum of spikes over entire time horizon
            spike_sum = jnp.array(states[idx][-1]).mean(axis=0).sum() #last column is the spike output
            mem_pot_var = jnp.var(states[idx][0]) 
            mem_pot_mean = jnp.mean(states[idx][0])
            spike_mean = jnp.mean(spike_sum)
            
            
            assert spike_mean>=0
            
            print(f"Layer: {idx}, Variance of U: {mem_pot_var:.3}, \
                    Mean of U: {mem_pot_mean:.3}, \
                    Mean of S: {spike_mean:.3}")
            
            if jnp.isnan(mem_pot_var) or jnp.isnan(mem_pot_mean):
                done = False
                raise ValueError("NaN encountered during init!")
        
            if jnp.abs(mem_pot_var-tgt_var) > var_tol:
                model.layers[ilayer] = apply_to_tree_leaf(layer, 
                                                        "weight", 
                                                        adjust_var)
                done = False
            else:
                done = True
            alldone *= done
            
                
            if jnp.abs(mem_pot_mean-tgt_mean) > mean_tol:
                _has_bias = lambda leaf: hasattr(leaf, "bias")
                # Initialization with or without bias
                # TODO simplify this expression and refine case w/o bias
                bias_list = [getattr(leaf, "bias") is not None for leaf in tree_leaves(layer, is_leaf=_has_bias) if _has_bias(leaf)]
                if all(bias_list):
                    model.layers[ilayer] = apply_to_tree_leaf(layer, 
                                                            "bias", 
                                                            adjust_mean_bias)
                else:
                    eps = -.05*jnp.sign(mem_pot_mean-tgt_mean)/jnp.abs(mem_pot_mean-tgt_mean)**2
                    adjust_weight = ft.partial(adjust_mean_weight, eps)
                    model.layers[ilayer] = apply_to_tree_leaf(layer, 
                                                            "weight", 
                                                            adjust_weight)
                done = False
            else:
                done = True
            alldone *= done
        iters += 1
            
    return model

