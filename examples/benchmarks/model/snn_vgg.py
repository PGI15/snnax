#!/bin/python
#-----------------------------------------------------------------------------
# File Name : model_snnvgg.py
# Author: Emre Neftci
#
# Creation Date : Tue 28 Mar 2023 03:15:53 PM CEST
# Last Modified : Tue 23 Apr 2024 10:00:52 PM CEST
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from .common import *
import snnax.snn as snn
from snnax.functional.surrogate import superspike_surrogate as sr

class spVGG(eqx.Module):
    cell: eqx.Module
    #avgpool: eqx.Module
    classifier: eqx.Module
    ro_int: int 
    burnin: int 

    def __init__(self, 
                 cfg: List[Union[str, int]], 
                 input_size : List[int] = [2,32,32],
                 out_channels:int = 11, 
                 dt: float = 0.001, 
                 alpha = None,
                 beta = None,
                 tau_m: float = .02,
                 tau_s: float = .006,
                 dropout_rate: float = 0.0,
                 ro_int: int = -1,
                 burnin: int = 20,
                 norm: bool = True,
                 key: jrandom.PRNGKey = None,                 
                 neuron_model: str = 'snnax.snn.LIFSoftReset',
                 **kwargs):

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

        
        ckey, lkey = jrandom.split(key)
        self.ro_int = ro_int
        self.burnin = burnin

        
        key1, key2, key3, key4, key = jrandom.split(key, 5)
        surr = sr(beta = 10.0)
        layers, channels, output_size = make_layers(input_size,
                                                    alpha,
                                                    beta,
                                                    surr,
                                                    cfg,
                                                    norm = norm,
                                                    dropout_rate = dropout_rate,
                                                    neuron_model = get_method(neuron_model),
                                                    key=key)
        self.cell = snn.Sequential(*layers)

        init_key1, init_key2 = jax.random.split(key,2)

        self.classifier = eqx.nn.Sequential([
            eqx.nn.Linear(int(np.prod(output_size)),out_channels, key = init_key1),
            ])

    def __call__(self, x, key=None):
        feat = self.embed(x, key = key)
        if self.ro_int is None:
            return self.classifier(feat.reshape(-1), key = key)
        else:
            _key = jax.random.split(key, feat.shape[0])
            return jax.vmap(self.classifier)(feat.reshape(feat.shape[0],-1), key = _key)

    def embed(self, x, key=None):
        state = self.cell.init_state(x[0,:].shape, key)
        state, out = self.cell(state, x, key, burnin=self.burnin)

        if self.ro_int is None:
            return out[-1][-1]
        elif self.ro_int == -1:
            ro = out[-1].shape[0]
            return out[-1][::-ro]
        else:
            ro = np.minimum(self.ro_int, out[-1].shape[0])
            return out[-1][::-ro]
    
    def get_final_states(self, x, key):
        #For the moment only return the output of cell, bypassing the avg adaptive pooling
        state = self.cell.init_state(x[0,:].shape, key)

        states, out = self.cell(state, x, key, burnin=self.burnin)
        return states, out

def make_layers(input_size,
                alpha, beta, surr,
                cfg: List[Union[
str, int]],
                norm: bool = False,
                dropout_rate: float = 0.0,
                neuron_model = eqx.Module,
                key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
    layers = []
    input_size = np.array(input_size)
    input_size = input_size.transpose()
    in_channels = input_size[0]
    for v in cfg:
        shape = np.concatenate([[v], input_size[1:]])
        if v == "M":
            layers += [snn.MaxPool2d(kernel_size=2,stride=2,padding=0, spike_fn = surr, threshold = 1)]
            input_size[1:] = (input_size[1:]-2)//2 + 1
        elif v == "AL":
            layers += [eqx.nn.AdaptiveAvgPool2d(target_shape=(4,4)),
                       eqx.nn.Lambda(lambda x: x.reshape(-1)),
                       eqx.nn.Linear(4*4*in_channels, 4*4*in_channels, key=init_key),
                       eqx.nn.LayerNorm(shape=shape,elementwise_affine=False, eps=1e-4)]
        elif type(v)==int:
            v = int(v)
            init_key, key = jax.random.split(key,2)
            dropout = eqx.nn.Dropout(dropout_rate)
            conv2d = eqx.nn.Conv2d(in_channels = in_channels,
                               out_channels = v,
                               kernel_size=(5, 5),
                               padding=2,
                               key = init_key)
            if norm:
                layers += [dropout,
                        conv2d,
                        eqx.nn.LayerNorm(shape=shape,elementwise_affine=False, eps=1e-4)]
            else:
                layers += [dropout,
                           conv2d]
            layers += [neuron_model([alpha,beta], shape=shape, spike_fn=surr, reset_val=1, key=init_key)]
            in_channels = v
        elif v[0] == "A":
            if len(v) == 1: #backward compatibility
                v = "A4"
            layers += [eqx.nn.AdaptiveAvgPool2d(target_shape=(int(v[1:]),int(v[1:])))]
            input_size[1],input_size[2] = int(v[1:]),int(v[1:])
        elif v[0] == "L":
            ch = int(v[1:])
            init_key, key = jax.random.split(key,2)
            layers += [eqx.nn.Conv2d(in_channels, ch, kernel_size=(1,1),key=init_key)]
        else:
            raise ValueError(f"Unrecognized layer type {v}")
        input_size[0] = in_channels 
    return layers, in_channels, input_size.tolist()
# "A" average pool
# "M" max pool
# int conv2d with relu
# "AL" adaptive average pool + linear
# TODO: "ML" max pool + linear

cfgs = {
    "S": [32, "M", 64, 64, "M", 128, 128, "M", "A4"],
    "M": [16, "M", 32, "M", 64, 64, "M", 128, 128, "M", "A4"],
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M", "A4"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M", "A4"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M", "A4"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M", "A4"],
}

def snn_vgg(input_size, cfg, **kwargs):
    model = spVGG(cfg = cfg, input_size = input_size, **kwargs)
    return default_init(model)


def snn_vgg4_hybrid(input_size, cfg = cfgs['S'], **kwargs):
    model = spVGG(cfg = cfg, input_size = input_size, **kwargs)
    return default_init(model)

def snn_vgg6_hybrid(input_size, cfg = cfgs['M'], **kwargs):
    model = spVGG(cfg = cfg, input_size = input_size, **kwargs)
    return default_init(model)

def snn_vgg11_hybrid(input_size, cfg = cfgs['A'], **kwargs):
    model = spVGG(cfg = cfg, input_size = input_size, **kwargs)
    return default_init(model)


if __name__ == "__main__":
    key = jrandom.PRNGKey(0)
    bs=16
    batch_key = jrandom.split(key, bs)
    model = spVGG(cfg=cfgs['S'], key = key)

    x = jnp.zeros((bs,50,2,150,130)) #batch, time, channels, height, width
    out = jax.vmap(model)(x, batch_key)

