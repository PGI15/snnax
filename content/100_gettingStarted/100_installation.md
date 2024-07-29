# Installation

You can install SNNAX from PyPI using pip:

```bash
pip install snnax
```

Or you can install the latest version from GitHub using pip:

```bash
pip install git+https://github.com/PGI15/snnax.git
```

:::note
This project has been set up using PyScaffold 4.3.1. For details and usage
information on **PyScaffold**, see the [docs](https://pyscaffold.org/en/stable/).
:::

## Requirements

The following packages need to be installed to use SNNAX:

- Python 3.9+
- JAX 0.4.13+
- Equinox 0.11.1+.

These dependencies are automatically installed if SNNAX is installed using the pip command.

## Validate installation

To confirm that snnax has been installed, you can use:

```bash
pip list | grep snnax
```

the output will be similar to:

```bash
snnax       0.0.1.18
```

## Example

```python
import equinox as eqx
import snnax.snn as snn

# Define the network layers
layers = [
    eqx.nn.Linear(16, 16),
    snn.LIF([.95, .85]),

    eqx.nn.Linear(16, 2),
    snn.LIF([.95, .85])
]

# Define the graph structure
graph = snn.GraphStructure(
    4, [[0],[],[],[]], [3], [[], [0], [1], [2]]
)

# Create the stateful model
model = snn.StatefulModel(graph, layers)

# Print the model to confirm it's been created
print(model)
```
