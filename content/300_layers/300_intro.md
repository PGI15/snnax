# Introduction

To finalize the model architecture, we need to define the layers that will be used in the model. SNNAX provides a simple way to define these layers. The layers can be from the `equinox` library, from our custom layers in the SNNAX library, or you can create your own custom layers.

SNNAX provides two ways to define the layers of the model:

- [`snnax.snn.layers.StatefulLayer`](./300_intro.md#statefullayer): to create custom layers.
- [`snnax.snn.layers`](./301_predefined.md): provides classes of predefined layers that can be used to build your model.

## StatefulLayer

The `StatefulLayer` class allows the creation of custom layers with specific properties which enables you to create highly customizable layers, adapting to specific needs of your neural network models. It inherits from `eqx.Module` to be a callable pytree.

### Properties

- `init_fn (Callable)`: A function for initializing the state of the layer. If not provided, it defaults to initializing the state with zeros.

### Methods

- `init_parameters(parameters, shape, requires_grad)`: A static method to initialize the parameters of the layer.
- `init_state(shape, key, *args, **kwargs)`: Initializes the state of the layer, defaults to zeros.
- `init_out(shape, key)`: Initializes the output of the layer, defaults to zeros.
- `__call__(state, synaptic_input, key)`: Defines the computation performed at every call of the layer.

#### Example

You can create a custom layer by subclassing the `StatefulLayer` class and defining the `__init__`, `__call__`, `init_out`, or `init_state` methods.

```python
from snnax import snn
import equinox as eqx
import jax.numpy as jnp
from jax.random import PRNGKey

class CustomLayer(snn.layers.StatefulLayer):

    def __init__(self,
                alpha: float,
                beta: float,
                init_fn: Optional[Callable] = None) -> None:

        # Custom initialization function
        super().__init__(init_fn)
        self.alpha = self.init_parameters(alpha, (1,), requires_grad=False)
        self.beta = self.init_parameters(beta, (1,), requires_grad=True)

    def __call__(self,
                state: Array,
                synaptic_input: Array, *,
                key: Optional[PRNGKey] = None) -> Sequence[Array]:

        # Custom neuron dynamics
        alpha = self.alpha.data[0]
        beta = self.beta.data[0]
        mem_pot = state
        mem_pot = alpha*mem_pot + (1.-alpha)*beta*synaptic_input

        output = mem_pot
        state = mem_pot
        return state, output


custom_layer = CustomLayer(alpha=0.9, beta=0.1)
```
