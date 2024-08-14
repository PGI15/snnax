#!/bin/python
#-----------------------------------------------------------------------------
# File Name : model_snnmlp.py
# Author: Emre Neftci
#
# Creation Date : Tue 28 Mar 2023 12:44:33 PM CEST
# Last Modified : Wed 14 Aug 2024 05:08:20 PM CEST
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from .common import *
import snnax.snn as snn
from snnax.functional.surrogate import superspike_surrogate as sr

# In[3]:
class SNNMLP(eqx.Module):
    cell: eqx.Module
    ro_int: int 
    burnin: int 
    shapes: List[int] = eqx.static_field()

    def __init__(self, 
                 in_channels: int = 32*32*2,
                 hid_channels: int = 64,
                 out_channels:int = 11, 
                 dt : float = 0.001,
                 alpha = None,
                 beta = None,
                 tau_m: float = .02,
                 tau_s: float = .006,
                 size_factor: float = 4,
                 ro_int: int = -1,
                 burnin: int = 20,
                 norm: bool = True,
                 num_hid_layers: int = 1,
                 use_bias: bool = True,
                 neuron_model: str = 'snnax.snn.LIFSoftReset',
                 key = jrandom.PRNGKey(0),                 
                 **kwargs):
        
        ckey, lkey = jrandom.split(key)
        #conn,inp,out = snn.composed.gen_feed_forward_struct(6)
        self.ro_int = ro_int
        self.burnin = burnin
        #graph = snn.GraphStructure(6,inp,out,conn)

        if tau_m is None:
            tau_m = dt/np.log(alpha)
        else:
            assert alpha is None, "Only one of alpha or tau_m can be specified"
            alpha = np.exp(-dt/tau_m)

        if tau_s is None:
            tau_s = dt/np.log(beta)
        else:
            assert beta is None, "Only one of beta or tau_s can be specified"
            beta = np.exp(-dt/tau_s)

        layers, self.shapes = make_layers(in_channels,
                         hid_channels,
                         out_channels,
                         key = key,
                         neuron_model = get_method(neuron_model),
                         alpha = alpha,
                         beta = beta,
                         size_factor=size_factor, 
                         use_bias = use_bias,
                         norm = norm, num_hid_layers=num_hid_layers)
        
        self.cell = snn.composed.SequentialLocalFeedback(
            *layers,
            forward_fn = snn.architecture.default_forward_fn,
            feedback_layers = None,) 

    def __call__(self, x, key=None, seqlen=None):
        if seqlen is not None:
            assert self.ro_int is None, "readout interval not yet supported with seqlen"
            return self.get_final_states(x, key, seqlen)[-1]
        else:    
            state, out = self.get_final_states(x, key, seqlen)
            return self.multiloss(out)
        
    def multiloss(self, out):
        if self.ro_int is None:
            return out[-1][-1]
        elif self.ro_int == -1:
            ro = out[-1].shape[0]
            return out[-1][::-ro]
        else:
            ro = np.minimum(self.ro_int, out[-1].shape[0])
            return out[-1][::-ro]

    def get_cumsum(self,x,key, seqlen=None):
        state = self.cell.init_state([ x[0,:].shape ], shapes=self.shapes, key=key)
        state, out = self.cell(state, x, key, burnin=self.burnin)
        seq = out[-1]
        f = lambda n: jnp.tile(jnp.arange(out[-1].shape[0])<(n-self.burnin), (out[-1].shape[1],1)).T
        return (f(seqlen)*seq).sum(axis=0)

    def embed(self, x, key):
        state = self.cell.init_state([ x[0,:].shape ], shapes=self.shapes, key = key)
        state, out = self.cell(state, x, key, burnin=self.burnin)
        return out[-1][-1]
    
    def get_final_states(self, x, key, seqlen=None):
        state = self.cell.init_state([ x[0,:].shape ], shapes=self.shapes, key = key)

        states, out = self.cell(state, x, key, burnin=self.burnin)
        if seqlen is None:
            return states, out
        else:
            return states, out[-1][seqlen-self.burnin,:]

class LinearNonDiag(eqx.nn.Linear):
    def __call__(self, x: Array, *, key: Optional[Sequence[PRNGKey]] = None) -> Array:
        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = (self.weight-jnp.diag(self.weight)) @ x
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x

def make_layers(in_channels, hid_channels, out_channels, key, neuron_model, size_factor=1, use_bias=True, num_hid_layers=2, alpha=0.95, beta=.85, norm=False):
    surr = sr(beta = 10.0)
    shapes = []
    layers = [snn.Flatten()]
    shapes.append(in_channels)
    for i in range(num_hid_layers):
        m = []
        init_key, key = jrandom.split(key,2)
        _out_s = hid_channels*size_factor
        _ff = eqx.nn.Linear(in_channels, _out_s, key=init_key, use_bias=use_bias)
        _rec = LinearNonDiag(_out_s, _out_s, key=init_key, use_bias=False)
        m.append(snn.composed.Parallel([_ff,_rec]))
        shapes.append([in_channels,_out_s])
        if norm:
            m.append(eqx.nn.LayerNorm(shape=(_out_s,), elementwise_affine=False, eps=1e-4))
        m.append(neuron_model(decay_constants = [alpha,beta], 
                              spike_fn=surr, 
                              reset_val=1,
                              shape=[_out_s],
                              key=init_key
                              ))
        
        #layers += m
        layers .append( snn.composed.CompoundLayer(*m))
        in_channels = hid_channels*size_factor

    init_key, key = jrandom.split(key,2)
    layers.append(eqx.nn.Linear(hid_channels*size_factor, out_channels, key=key, use_bias=use_bias))
    shapes.append(hid_channels*size_factor)
    return layers, shapes

def _model_init(model):
    ## Custom code ensures that only  conv layers are trained
    import jax.tree_util as jtu
    from utils import apply_to_tree_leaf_bytype,apply_to_tree_leaf

    filter_spec = jtu.tree_map(lambda _: False, model)

    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.Conv2d, 'weight', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.Conv2d, 'bias', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.Linear, 'weight', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.Linear, 'bias', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.LayerNorm, 'weight', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.LayerNorm, 'bias', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.GroupNorm, 'weight', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.GroupNorm, 'bias', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, 'requires_grad', 'data', lambda _: True)
    
    return model, filter_spec

def snn_mlp(input_size=[2,32,32], out_channels=10, key = jax.random.PRNGKey(0), **kwargs):
    return _model_init(SNNMLP(in_channels = np.prod(input_size), out_channels=out_channels, key = key, **kwargs))


if __name__ == "__main__":
    key = jrandom.PRNGKey(0)
    batch_key = jrandom.split(key, 36)
    model, filter_spec = snn_mlp(out_channels=10, norm=True, num_hid_layers=3, key = key)

    x = jnp.zeros((36,50,2*32*32)) #batch, time, channels, height, width
    out = jax.vmap(model)(x, batch_key)

