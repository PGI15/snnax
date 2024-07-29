# Predefined Layers

The `snnax.snn.layers` module provides a range of predefined layers that simplify the construction of spiking neural networks (SNNs). These layers offer common functionalities and neuron models, allowing for easier and more efficient network design.
Each of these predefined layers can be combined to build complex and tailored SNN architectures, leveraging the built-in functionalities for efficient and effective network design. Below is an overview of the available predefined layers:

## BatchNormLayer

The `BatchNormLayer` subclass of `eqx.Module`, applies batch normalization to the input, normalizing the features to stabilize learning and enhance convergence. This layer is useful for mitigating internal covariate shift during training.

#### Arguments

- `eps (float)`: A small value added to the variance to avoid division by zero.
- `forget_weight (float)`: The weight of the previous mean and variance in the current batch normalization.
- `gamma (float)`: The scaling factor for the normalized input, default is 0.8.

## Flatten

The `Flatten` subclass of `eqx.Module`, reshapes the input tensor into a 1D tensor. This is particularly useful for transitioning between different types of layers, such as moving from convolutional to fully connected layers.

## SimpleIAF

The `SimpleIAF` subclass of [`snn.StatefulLayer`](./300_intro.md#statefullayer), implements a basic integrate-and-fire (IAF) neuron model. It integrates raw synaptic input without explicit modeling of synaptic currents and requires one constant to simulate a constant leak in membrane potential.

State update function is given by the following equation:

$$
\begin{aligned}
\text{membrane\_potential}(t) &=  (\text{membrane\_potential}(t-1) - \text{reset\_val} \cdot synaptic\_input - \text{leak}) + \text{synaptic\_input} \\
\text{spike\_output} &= \text{spike\_fn}(\text{membrane\_potential}(t) - \text{threshold})
\end{aligned}
$$

#### Arguments

- `spike_fn (Callable)`: The surrogate function for the spike operation, default is [`superspike_surrogate(10.)`](../400_functions.md#superspike_surrogate).
- `leak (float)`: The leak constant for the membrane potential, default is 0.
- `threshold (float)`: The threshold for the membrane potential to spike, default is 1.
- `stop_reset_grad (bool)`: Whether to stop the gradient during the reset operation, default is True.
- `reset_val (Optional[float])`: The reset value for the membrane potential, default is None. If None, the reset value is set to 0.
- `init_fn (Optional[Callable])`: The initialization function for the layer, default is None. If None, the layer is initialized with zeros.

## IAF

The `IAF` subclass of [`snn.StatefulLayer`](./300_intro.md#statefullayer), models an integrate-and-fire neuron with an optional constant leak. By default, no leak is applied, but it can be configured to include a leak, providing flexibility for different neuron dynamics.

State update function is given by the following equation:

$$
\begin{aligned}
\text{membrane\_potential}(t) &= \alpha \cdot (\text{membrane\_potential}(t-1)  - \text{reset\_val} \cdot synaptic\_input - \text{leak}) + (1 - \alpha) \cdot \text{synaptic\_current}(t-1) \\
\text{synaptic\_current}(t) &= \beta \cdot \text{synaptic\_current}(t-1) + (1 - \beta) \cdot \text{synaptic\_input} \\
\text{spike\_output} &= \text{spike\_fn}(\text{membrane\_potential}(t) - \text{threshold})
\end{aligned}
$$

#### Arguments

- `decay_constants (Union[Sequence[float], Array])`: The decay constants for the membrane potential and synaptic current. Index 0 describes the decay constant of the membrane potential $\alpha$, Index 1 describes the decay constant of the synaptic current $\beta$.
- `spike_fn (Callable)`: The surrogate function for the spike operation, default is [`superspike_surrogate(10.)`](../400_functions.md#superspike_surrogate).
- `leak (float)`: The leak constant for the membrane potential, default is 0.
- `threshold (float)`: The threshold for the membrane potential to spike, default is 1.
- `stop_reset_grad (bool)`: Whether to stop the gradient during the reset operation, default is True.
- `reset_val (Optional[float])`: The reset value for the membrane potential, default is None. If None, the reset value is set to 0.
- `init_fn (Optional[Callable])`: The initialization function for the layer, default is None. If None, the layer is initialized with zeros.

## LI

The `LI` subclass of [`snn.StatefulLayer`](./300_intro.md#statefullayer), represents a Leaky Integrator (LI) neuron, integrating over synaptic inputs with a leaky mechanism. This layer is suitable for tasks requiring leaky integration without complex dynamics.

State update function is given by the following equation:

$$
\begin{aligned}
\text{membrane\_potential}(t) &= \alpha \cdot \text{membrane\_potential}(t-1) + (1 - \alpha) \cdot \text{synaptic\_input} \\
\text{spike\_output} &= \text{membrane\_potential}(t)
\end{aligned}
$$

#### Arguments

- `decay_constants (float)`: The decay constant for the membrane potential $\alpha$.
- `init_fn (Optional[Callable])`: The initialization function for the layer, default is None. If None, the layer is initialized with zeros.

## SimpleLIF

The `SimpleLIF` subclass of [`snn.StatefulLayer`](./300_intro.md#statefullayer), a straightforward implementation of a leaky integrate-and-fire (LIF) neuron that does not explicitly model synaptic currents. It requires a decay constant to simulate membrane potential leak.

State update function is given by the following equation:

$$
\begin{aligned}
\text{membrane\_potential}(t) &= \alpha \cdot (\text{membrane\_potential}(t-1)  - \text{reset\_val} \cdot synaptic\_input) + (1 - \alpha) \cdot \text{synaptic\_input} \\
\text{spike\_output} &= \text{spike\_fn}(\text{membrane\_potential}(t) - \text{threshold})
\end{aligned}
$$

#### Arguments

- `decay_constants (float)`: Initial value of the trainable decay constant for the membrane potential $\alpha$.
- `spike_fn (SpikeFn)`: The surrogate function for the spike operation, default is [`superspike_surrogate(10.)`](../400_functions.md#superspike_surrogate).
- `threshold (float)`: The threshold for the membrane potential to spike, default is 1.
- `stop_reset_grad (bool)`: Whether to stop the gradient during the reset operation, default is True.
- `reset_val (Optional[float])`: The reset value for the membrane potential, default is None. If None, the reset value is set to 0.
- `init_fn (Optional[Callable])`: The initialization function for the layer, default is None. If None, the layer is initialized with zeros.
- `shape (Optional[StateShape])`: If given, the parameters will be expanded into vectors and initialized accordingly.
- `key (Optional[PRNGKey])`: Used to initialize the parameters.

## LIF

The `LIF` subclass of [`snn.StatefulLayer`](./300_intro.md#statefullayer), models a leaky integrate-and-fire neuron with synaptic currents. It uses two decay constants: one for the membrane potential and another for the synaptic current, providing a detailed representation of neuron dynamics.

State update function is given by the following equation:

$$
\begin{aligned}
\text{membrane\_potential}(t) &= \alpha \cdot (\text{membrane\_potential}(t-1)  - (\text{membrane\_potential}(t-1) - \text{reset\_val}) \cdot synaptic\_input) + (1 - \alpha) \cdot \text{synaptic\_current}(t-1) \\
\text{synaptic\_current}(t) &= \beta \cdot \text{synaptic\_current}(t-1) + (1 - \beta) \cdot \text{synaptic\_input} \\
\text{spike\_output} &= \text{spike\_fn}(\text{membrane\_potential}(t) - \text{threshold})
\end{aligned}
$$

#### Arguments

- `decay_constants (Union[Sequence[float], Array])`: Initial value of the trainable decay constants for the membrane potential and synaptic current. Index 0 describes the decay constant of the membrane potential $\alpha$, Index 1 describes the decay constant of the synaptic current $\beta$.
- `spike_fn (Callable)`: The surrogate function for the spike operation, default is [`superspike_surrogate(10.)`](../400_functions.md#superspike_surrogate).
- `threshold (float)`: The threshold for the membrane potential to spike, default is 1.
- `stop_reset_grad (bool)`: Whether to stop the gradient during the reset operation, default is True.
- `reset_val (Optional[float])`: The reset value for the membrane potential, default is None. If None, the reset value is set to 0.
- `init_fn (Optional[Callable])`: The initialization function for the layer, default is None. If None, the layer is initialized with zeros.
- `shape (Optional[StateShape])`: If given, the parameters will be expanded into vectors and initialized accordingly.
- `key (Optional[PRNGKey])`: Used to initialize the parameters.

## LIFSoftReset

The `LIFSoftReset` subclass of `snn.LIF` similar to the `LIF` layer but applies an additive (relative) reset when a neuron spikes. Instead of setting the membrane potential to a fixed reset value, it adds a reset value to the current membrane potential.
it has the same arguments as `LIF` layer.

State update function is given by the following equation:

$$
\begin{aligned}
\text{membrane\_potential}(t) &= \alpha \cdot (\text{membrane\_potential}(t-1)  - \text{reset\_val} \cdot synaptic\_input) + (1 - \alpha) \cdot \text{synaptic\_current}(t-1) \\
\text{synaptic\_current}(t) &= \beta \cdot \text{synaptic\_current}(t-1) + (1 - \beta) \cdot \text{synaptic\_input} \\
\text{spike\_output} &= \text{spike\_fn}(\text{membrane\_potential}(t) - \text{threshold})
\end{aligned}
$$

## AdaptiveLIF

The `AdaptiveLIF` subclass of [`snn.StatefulLayer`](./300_intro.md#statefullayer), implements an adaptive exponential leaky integrate-and-fire neuron.

State update function is given by the following equation:

$$
\begin{aligned}
\text{adaptive\_state}(t) &=  \beta \cdot \text{adaptive\_state}(t-1) + (1 - \beta) \cdot a \cdot \text{membrane\_potential}(t-1) - b \cdot synaptic\_input\\
\text{membrane\_potential}(t) &= \alpha \cdot (\text{membrane\_potential}(t-1)  - \text{reset\_val} \cdot synaptic\_input) + (1 - \alpha) \cdot (synaptic\_input - \text{adaptive\_state}(t))\\
\text{spike\_output} &= \text{spike\_fn}(\text{membrane\_potential}(t) - \text{threshold})
\end{aligned}
$$

#### Arguments

- `decay_constants (float)`: Initial value of the trainable decay constant for the membrane potential $\alpha$.
- `ada_decay_constant (float)`: Initial value of the trainable decay constant for the adaptive threshold $\beta$, default is 0.8.
- `ada_step_val (float)`: Initial value of the trainable step value for the adaptive threshold $b$, default is 1.0.
- `ada_coupling_var (float)`: Initial value of the trainable coupling variable for the adaptive threshold $a$, default is 0.5.
- `spike_fn (Callable)`: The surrogate function for the spike operation, default is [`superspike_surrogate(10.)`](../400_functions.md#superspike_surrogate).
- `threshold (float)`: The threshold for the membrane potential to spike, default is 1.
- `stop_reset_grad (bool)`: Whether to stop the gradient during the reset operation, default is True.
- `reset_val (Optional[float])`: The reset value for the membrane potential, default is None. If None, the reset value is set to 0.
- `init_fn (Optional[Callable])`: The initialization function for the layer, default is None. If None, the layer is initialized with zeros.
- `shape (Optional[StateShape])`: If given, the parameters will be expanded into vectors and initialized accordingly.
- `key (Optional[PRNGKey])`: Used to initialize the parameters.

## MaxPool1d

The `MaxPool1d` subclass of `eqx.nn.MaxPool1d` and [`snn.StatefulLayer`](./300_intro.md#statefullayer), performs 1D max pooling on the input tensor and applies a spike operation to the output.

#### Arguments

it has default arguments similar to [`eqx.nn.MaxPool1d`](https://docs.kidger.site/equinox/api/nn/pool/#equinox.nn.MaxPool1d) layer, in addition to the following arguments:

- `spike_fn (Callable)`: The surrogate function for the spike operation, default is [`superspike_surrogate(10.)`](../400_functions.md#superspike_surrogate).
- `threshold (float)`: The threshold for the membrane potential to spike, default is 1.

## MaxPool2d

The `MaxPool2d` subclass of `eqx.nn.MaxPool2d`,[`snn.StatefulLayer`](./300_intro.md#statefullayer), performs 2D max pooling on the input tensor and applies a spike operation to the output.

#### Arguments

it has default arguments similar to [`eqx.nn.MaxPool2d`](https://docs.kidger.site/equinox/api/nn/pool/#equinox.nn.MaxPool2d) layer, in addition to the following arguments:

- `spike_fn (Callable)`: The surrogate function for the spike operation, default is [`superspike_surrogate(10.)`](../400_functions.md#superspike_surrogate).
- `threshold (float)`: The threshold for the membrane potential to spike, default is 1.

## ResNetBlock

The `ResNetBlock` subclass of `eqx.Module`, implements a residual block used in ResNet architectures. It includes two convolutional layers with a skip connection that bypasses the second convolutional layer, allowing for deeper networks with improved training dynamics.

#### Arguments

- `layer_order (Sequence)`: Represents the order of layers in the block, which can be 'c' for convolutional layers or 'l' for linear layers.
- `layer_params (Sequence)`: The parameters for each layer in the block. which represents [in_channels, out_channels, kernel_size, stride, padding] for convolutional layers and [in_features, out_features] for linear layers.
- `stateful_layer_type (str)`: The type of stateful layer to use in the block, which can be 'LIF' or 'SigmaDelta', default is 'LIF'.
- `key (Optional[PRNGKey])`: The random key for initialization, default is None.

## SigmaDelta

The `SigmaDelta` subclass of [`snn.StatefulLayer`](./300_intro.md#statefullayer), models a Sigma-Delta neuron, providing another variant of spiking neuron models with specific integration and firing properties.

#### Arguments

- `threshold (float)`: The threshold for the membrane potential to spike, default is 1.
- `spike_fn (Callable)`: The surrogate function for the spike operation, default is [`superspike_surrogate(10.)`](../400_functions.md#superspike_surrogate).
- `init_fn (Optional[Callable])`: The initialization function for the layer, default is None. If None, the layer is initialized with zeros.

## SRM

The `SRM` (Spike Response Model) layer simulates the postsynaptic response to spike input. It models the synaptic response of a neuron to incoming spikes, useful for studying spike-timing dependent plasticity and other synaptic effects.

#### Arguments

- `decay_constants (Union[Sequence[float], jnp.ndarray, TrainableArray])`: Initial value of the trainable decay constants for the membrane potential and synaptic current. Index 0 describes the decay constant of the membrane potential, Index 1 describes the decay constant of the synaptic current.
- `r_decay_constants (Union[Sequence[float], jnp.ndarray, TrainableArray])`: Initial value of the trainable decay constants for the refractory period, default is 0.9.
- `spike_fn (Callable)`: The surrogate function for the spike operation, default is [`superspike_surrogate(10.)`](../400_functions.md#superspike_surrogate).
- `threshold (Union[float, jnp.ndarray])`: The threshold for the membrane potential to spike, default is 1.
- `reset_val (Optional[Union[float, jnp.ndarray, TrainableArray])`: Initial value of the trainable reset value for the membrane potential, default is None. If None, the reset value is set to 0.
- `stop_reset_grad (Optional[bool])`: Whether to stop the gradient from propagating through the refectory potential, default is True.
- `init_fn (Optional[Callable])`: The initialization function for layer states, default is None. If None, the layer is initialized with zeros.
- `input_shape (Union[Sequence[int],int,None])`: The shape of the neuron layer.
- `shape (Union[Sequence[int],int,None])`: The shape of the layer.
- `key (jrand.PRNGKey)`: Random key for initialization.
  $$
