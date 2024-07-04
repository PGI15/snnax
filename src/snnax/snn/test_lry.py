import layers.lru as lru
import jax.numpy as jnp
import jax
_lru = lru.LRU(64,4)

d_model = 4 
L = 1000

inputs = jax.random.normal(key= jax.random.PRNGKey(0), shape = (L,4))*1e-3+1
layer = lru.LRULayer(d_hidden=64,d_model=d_model)

out = layer(inputs)




