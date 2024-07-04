#!/bin/python
#-----------------------------------------------------------------------------
# File Name : model_vgg.py
# Author: Emre Neftci
#
# Creation Date : Tue 21 Mar 2023 04:25:25 PM CET
# Last Modified : Thu 04 Apr 2024 12:25:18 PM CEST
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 
'''
VGG implemention translated from PyTorch to Equinox using ch
'''
from .common import *

#Convenience
_conv3x3 = ft.partial(eqx.nn.Conv2d, kernel_size=3, use_bias=False)
_conv1x1 = ft.partial(eqx.nn.Conv2d, kernel_size=1, use_bias=False)
_bn = ft.partial(eqx.nn.LayerNorm, elementwise_affine=False)

class VGG(eqx.Module):
    features: eqx.Module
    avgpool: eqx.Module
    classifier: eqx.Module
    num_classes: int = 1000
    dropout: float = 0.5

    def __init__(self,
                 input_size,
                 cfg: List[Union[str, int]],
                 num_classes: int = 1000,
                 dropout: float = 0.5,
                 classifier_width = 4096,
                 act_fun = jax.nn.relu,
                 key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        super().__init__()
        init_key0, init_key1, init_key2, init_key3 = jax.random.split(key,4)
        self.features, channels, conv_size = make_layers(input_size, cfg, norm=True, key = init_key0, act_fun= act_fun)
        self.num_classes = num_classes
        self.dropout = dropout

        self.avgpool = nn.AdaptiveAvgPool2d(target_shape=conv_size[1:])
        self.classifier = nn.Sequential([
            nn.Linear(channels*np.prod(conv_size), classifier_width, key = init_key1),
            nn.Lambda(act_fun),
            nn.Dropout(self.dropout),
            nn.Linear(classifier_width, classifier_width, key = init_key2),
            nn.Lambda(act_fun),
            nn.Dropout(self.dropout),
            nn.Linear(classifier_width, self.num_classes, key = init_key3)])

    def __call__(self, x, key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        dropout_key, key = jax.random.split(key,2)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(-1)
        x = self.classifier(x, key = dropout_key)
        return x

def make_layers(input_size,
                cfg: List[Union[str, int]],
                norm: bool = False,
                act_fun = jax.nn.relu,
                key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
    layers = []
    input_size = np.array(input_size)
    in_channels = input_size[0]
    for v in cfg:
        if v == "M":
            layers += [eqx.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
            input_size[1:] = (input_size[1:]-2)//2 + 1
        else:
            v = int(v)
            init_key, key = jax.random.split(key,2)
            conv2d = eqx.nn.Conv2d(in_channels = in_channels,
                               out_channels = v,
                               kernel_size=(3, 3),
                               padding=1,
                               key = init_key)
            if norm:
                init_key, key = jax.random.split(key,2)
                shape = np.concatenate([[v], input_size[1:]])
                layers += [conv2d, 
                           eqx.nn.LayerNorm(shape, elementwise_affine=False),
                           nn.Lambda(act_fun)]
            else:
                layers += [conv2d,
                           nn.Lambda(act_fun)]
            in_channels = v
    input_size[0] = in_channels
    return nn.Sequential(layers), in_channels, input_size.tolist()

cfgs = {
    "debug28": [16, "M", 32, "M"],
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "A32": [64, "M", 128, "M", 256, 256, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
    
vgg_init = default_init

def vgg(out_channels=10, input_size=(3, 32, 32), cfg = 'A', dropout = 0.5, key = jax.random.PRNGKey(0), dt=None, **kwargs):
    return default_init(VGG(input_size, cfg = cfgs[cfg], num_classes = out_channels, dropout = dropout, key = key, **kwargs))

if __name__ == "__main__":
    batch_key = jax.random.split(jax.random.PRNGKey(0), 64)
    x = jnp.ones((64, 3, 32, 32 ))
    model = VGG(cfg = cfgs['A'], num_classes = 1000, dropout = 0.5, input_size = x.shape[1:])
    jax.vmap(model)(x, batch_key)
