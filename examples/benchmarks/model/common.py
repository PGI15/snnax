import sys
sys.path.append('../')
import equinox as eqx
from equinox import nn
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import einops


import math
import warnings
from functools import partial
from typing import *
from typing import Callable,Type
from typing import Sequence
from jaxtyping import Array, Bool, Float, PyTree
from typing import List, Union
from jax.random import PRNGKey
import functools as ft
from utils import apply_to_tree_leaf_bytype, default_init, get_method
from utils import prng_batch, TrainableArray

class CustomTensorTrainable(eqx.Module):
    data: Array

    def __init__(self,*dims):
        self.data = jnp.zeros(dims)



def dpfp(x, nu=1):
    r = jax.nn.relu
    x = jnp.concatenate([r(x), r(-x)], axis=-1)
    x_rolled = jnp.concatenate([jnp.roll(x, shift=j, axis=-1) for j in range(0,nu)], axis=-1)
    x_repeat = jnp.concatenate([x] * nu, axis=-1)
    return x_repeat * x_rolled


