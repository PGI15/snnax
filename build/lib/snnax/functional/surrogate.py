from typing import Callable
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import custom_jvp
from chex import Array


SpikeFn = Callable[[Array], Array]


def superspike_surrogate(beta: float = 10.) -> SpikeFn: 
    """
    Implementation of the superspike surrogate gradient function as described in
    'The remarkable robustness of surrogate gradient learning 
    for instilling complex function in spiking neural networks' by Zenke and 
    Vogels: (https://www.biorxiv.org/content/10.1101/2020.06.29.176925v1) 

    Arguments:
        `beta` (float): Parameter to control the steepness of the surrogate 
            gradient. Default is 10.

    Returns:
        A function that returns the surrogate gradient of the heaviside function.
    """
    @custom_jvp
    def heaviside_with_superspike_surrogate(x):
        return jnp.heaviside(x, 1.)

    @heaviside_with_superspike_surrogate.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = heaviside_with_superspike_surrogate(x)
        tangent_out = x_dot / (1. + beta * jnp.abs(x))
        return primal_out, tangent_out
    
    return heaviside_with_superspike_surrogate


def sigmoid_surrogate(beta: float = 1.) -> SpikeFn:
    """
    Implementation of the sigmoidal surrogate gradient function as described in
    'The remarkable robustness of surrogate gradient learning 
    for instilling complex function in spiking neural networks' by Zenke and 
    Vogels: (https://www.biorxiv.org/content/10.1101/2020.06.29.176925v1) 

    Arguments:
        `beta` (float): Parameter to control the steepness of the surrogate 
            gradient. Default is 10.

    Returns:
        A function that returns the surrogate gradient of the heaviside function.
    """
    @custom_jvp
    def heaviside_with_sigmoid_surrogate(x):
        return jnp.heaviside(x, 1.)

    @heaviside_with_sigmoid_surrogate.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = heaviside_with_sigmoid_surrogate(x)
        # TODO multiplication by beta correct here?
        tangent_out = jnn.sigmoid(x, beta) 
        tangent_out *= (1. - beta * jnn.sigmoid(x, beta)) * x_dot 
        return primal_out, tangent_out

    return heaviside_with_sigmoid_surrogate

def piecewise_surrogate(beta: float = .5) -> SpikeFn:
    """
    Implementation of the sigmoidal surrogate gradient function as described in
    'The remarkable robustness of surrogate gradient learning 
    for instilling complex function in spiking neural networks' by Zenke and 
    Vogels: (https://www.biorxiv.org/content/10.1101/2020.06.29.176925v1) 

    Arguments:
        `beta` (float): Parameter to control the steepness of the surrogate 
            gradient. Default is .5.

    Returns:
        A function that returns the surrogate gradient of the heaviside function.
    """
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

