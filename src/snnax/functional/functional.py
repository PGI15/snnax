import jax
import jax.numpy as jnp

# Taken from the default implementation in Jax
sigmoid = jax.nn.sigmoid
softplus = jax.nn.softplus


def inverse_softplus(x): #, beta: float = 1.0):
    return jnp.log(jnp.exp(x)-1.) # / beta


def divergence(P, Q):
    """
    Calculate the divergence between two probability distributions.
    """
    return -jnp.sum(Q*jax.nn.log_softmax(P)) / len(Q)


def one_hot_cross_entropy(prediction, target):
    """
    Calculate one-hot cross-entropy for a single prediction and target.
    This will be vectorized using vmap.
    """
    return divergence(prediction, target)

