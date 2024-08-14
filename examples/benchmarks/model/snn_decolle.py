#!/bin/python
#-----------------------------------------------------------------------------
# File Name : model_snndecolle.py
# Author: Emre Neftci
#
# Creation Date : Tue 28 Mar 2023 12:21:33 PM CEST
# Last Modified : Wed 14 Aug 2024 04:38:15 PM CEST
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 

'''
SNN Decolle model as implemented in the paper: Synaptic Plasticity Dynamics for
Deep Continuous Local Learning
(DECOLLE) [doi: 10.3389/fnins.2020.00424]
'''


from .common import *
import snnax.snn as snn
from snnax.functional.surrogate import superspike_surrogate as sr


class SNNDECOLLE(eqx.Module):
    '''
    This model only implements the network, not the local learning rule
    '''
    cell: eqx.Module
    ro_int: int 
    burnin: int 

    def __init__(self, 
                 num_classes:int = 11, 
                 dt : float = 0.001,
                 alpha = None,
                 beta = None,
                 tau_m: float = .02,
                 tau_s: float = .006,
                 skip_connections: bool = False,
                 size_factor: float = 2,
                 ro_int: int = -1,
                 burnin: int = 20,
                 key: jrandom.PRNGKey = None,                 
                 **kwargs):
        
        ckey, lkey = jrandom.split(key)
        conn,inp,out = snn.composed.gen_feed_forward_struct(19)
        self.ro_int = ro_int
        self.burnin = burnin
        graph = snn.GraphStructure(19, inp,out,conn)

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

        if skip_connections:
            conn[6]=[3,5]
            conn[12]=[11,9]
        
        key1, key2, key3, key4, key = jrandom.split(key, 5)
        surr = sr(beta = 10.0)
        self.cell = snn.StatefulModel(
        graph_structure = graph,
        layers=[
            eqx.nn.LayerNorm([2,32,32], eps=1e-5, elementwise_affine=False),
            eqx.nn.Conv2d(2, 8*size_factor, 5, 1, padding=2, key=key1, use_bias=True),
            snn.LIFSoftReset([alpha,beta], spike_fn=surr, reset_val=1),
            snn.SpikingMaxPool2d(2,2,0, spike_fn = surr),

            eqx.nn.LayerNorm([8*size_factor,16,16], eps=1e-5, elementwise_affine=False),
            eqx.nn.Conv2d(8*size_factor, 8*size_factor, 5, 1, padding=2, key=key2, use_bias=True),
            snn.LIFSoftReset([alpha,beta], spike_fn=surr, reset_val=1),
            eqx.nn.Conv2d(8*(1+int(skip_connections))*size_factor, 16*size_factor, 5, 1, padding=2, key=key3, use_bias=True),
            snn.LIFSoftReset([alpha,beta], spike_fn=surr, reset_val=1),
            snn.SpikingMaxPool2d(2,2,0, spike_fn = surr, threshold=1),    

            eqx.nn.LayerNorm((16*size_factor,8,8), eps=1e-5, elementwise_affine=False),
            eqx.nn.Conv2d(16*size_factor, 16*size_factor, 5, 1, padding=2, key=key2, use_bias=True),
            snn.LIFSoftReset([alpha,beta], spike_fn=surr, reset_val=1),
            eqx.nn.LayerNorm((16*size_factor,8,8), eps=1e-5, elementwise_affine=False),
            eqx.nn.Conv2d(16*(1+int(skip_connections))*size_factor, 32*size_factor, 5, 1, padding=2, key=key3, use_bias=True),
            snn.LIFSoftReset([alpha,beta], spike_fn=surr, reset_val=1),
            snn.SpikingMaxPool2d(2,2,0, spike_fn = surr, threshold=1), 

            snn.Flatten(),
            eqx.nn.Linear(512*size_factor, num_classes, key=key4, use_bias=True)],
        )

    def __call__(self, x, key=None):
        state = self.cell.init_state(x[0,:].shape, key)

        state, out = self.cell(state, x, key, burnin=self.burnin)
        if self.ro_int == -1:
            ro = out[-1].shape[0]
        else:
            ro = np.minimum(self.ro_int, out[-1].shape[0])
        return out[-1][::-ro]
    
    def get_final_states(self, x, key):
        state = self.cell.init_state(x[0,:].shape, key)

        states, out = self.cell(state, x, key, burnin=self.burnin)
        return states, out

def _model_init(model):
    ## Custom code ensures that only conv layers are trained
    from snnax.snn.layers.stateful import StatefulLayer
    import jax.tree_util as jtu

    filter_spec = jtu.tree_map(lambda _: False, model)

    # trainable_layers = [i for i, layer in enumerate(model.cell.layers) if hasattr(layer, 'weight')]
    ## or  isinstance(layer, eqx.nn.LayerNorm)
    trainable_layers = [i for i, layer in enumerate(model.cell.layers) if isinstance(layer, eqx.nn.Conv2d)]

    for idx in trainable_layers:
        filter_spec = eqx.tree_at(
            lambda tree: (tree.cell.layers[idx].weight, tree.cell.layers[idx].bias),
            filter_spec,
            replace=(True,True),
        )
    return model, filter_spec

def snn_decolle(key = jax.random.PRNGKey(0), input_size = (2,32,32), **kwargs):
    assert input_size == (2,32,32), 'SNN Decoll is hard coded for input (2,32,32)'
    return _model_init(SNNDECOLLE(key = key, **kwargs))


if __name__ == "__main__":
    key = jrandom.PRNGKey(0)
    batch_key = jrandom.split(key, 36)
    model, filter_spec = snn_decolle(key = key)

    x = jnp.zeros((36,50,2,32,32)) #batch, time, channels, height, width
    out = jax.vmap(model)(x, batch_key)

