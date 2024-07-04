from typing import Callable
import jax
import jax.numpy as jnp
from jax import custom_jvp

from .functional import sigmoid


# TODO set default to 1. or 10. as in 
# 'The remarkable robustness of surrogate gradient learning 
# for instilling complex function in spiking neural networks'
# by Zenke and Vogels (https://www.biorxiv.org/content/10.1101/2020.06.29.176925v1) 
def superspike_surrogate(beta=10.): # 

    @custom_jvp
    def heaviside_with_super_spike_surrogate(x):
        return jnp.heaviside(x, 1)

    @heaviside_with_super_spike_surrogate.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = heaviside_with_super_spike_surrogate(x)
        tangent_out = 1./(beta*jnp.abs(x)+1.) * x_dot
        return primal_out, tangent_out
    
    return heaviside_with_super_spike_surrogate


# TODO set default to 1. or 10. as in 
# 'The remarkable robustness of surrogate gradient learning 
# for instilling complex function in spiking neural networks'
# by Zenke and Vogels (https://www.biorxiv.org/content/10.1101/2020.06.29.176925v1) 
def sigmoid_surrogate(beta=1.): # set default to 1. or 10. ?
    assert float(beta) == 1., "Currently only beta = 1.0 is supported for numerical stability."
    @custom_jvp
    def heaviside_with_sigmoid_surrogate(x):
        return jnp.heaviside(x, 1.)

    @heaviside_with_sigmoid_surrogate.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = heaviside_with_sigmoid_surrogate(x)
        # TODO multiplication by beta correct here?
        tangent_out = sigmoid(x, beta) * (1.-beta*sigmoid(x, beta)) * x_dot 
        return primal_out, tangent_out

    return heaviside_with_sigmoid_surrogate

def piecewise_surrogate(beta=.5): # set default to 1. or 10. ?
    @custom_jvp
    def heaviside_with_sigmoid_surrogate(x):
        return jnp.heaviside(x, 1.)

    @heaviside_with_sigmoid_surrogate.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = heaviside_with_sigmoid_surrogate(x)
        # TODO multiplication by beta correct here?
        tangent_out = jnp.where((x>-beta)*(x<beta), x_dot, 0)
        return primal_out, tangent_out

    return heaviside_with_sigmoid_surrogate

