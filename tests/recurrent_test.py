import os
import functools as ft

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.random as jrandom

import optax
import snnax as snx
import snnax.snn as snn
import equinox as eqx

from tqdm import tqdm
from jax.tree_util import tree_map
from snnax.snn.architecture import SNNGraphStructure
from snnax.utils.data import DataLoader
from randman import RandmanDataset


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

N_CLASSES = 10 # Number of classes
N = [16, 32, N_CLASSES] # List of number of neurons per layer
N_EPOCHS = 25 # Number of training epochs
T = 100 # Number of timesteps per epoch
NUM_SAMPLES = N_CLASSES*1000
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 48
INPUT_SHAPE = 16

SEED = 42 
rng = np.random.default_rng(SEED)
key = jax.random.PRNGKey(SEED)

randman = RandmanDataset(rng, num_classes=N_CLASSES, num_units=N[0], num_steps=T, 
                        dim_manifold=2, num_samples=NUM_SAMPLES, alpha=2.0, shuffle=True)

dataloader_train = DataLoader(randman, batch_size=BATCH_SIZE, drop_last=True, num_workers=8, prefetch_batches=1)

def one_hot_cross_entropy(prediction, target_one_hot):
    """
    Calculate one-hot cross-entropy for a single prediction and target.
    This will be vectorized using vmap.
    """
    return -jnp.sum(target_one_hot*jax.nn.log_softmax(prediction)) / len(target_one_hot)

def calc_loss(model, init_state, data, target, loss_fn):
    """
    Here we define how the loss is exacly calculated, i.e. whether
    we use a sum of spikes or spike-timing for the calculation of
    the cross-entropy.
    """
    states, outs = model(init_state, data)
    final_layer_out = outs[-1] # TODO adjust to new API
    pred = tree_map(lambda x: jnp.sum(x, axis=0), final_layer_out) # sum over all spikes

    loss_val = loss_fn(pred, target)
    return loss_val

# Vectorization of the loss function calculation
vmap_calc_loss = jax.vmap(calc_loss, in_axes=(None, None, 0, 0, None))

def calc_batch_loss(model, init_state, input_batch, target, loss_fn):
    """
    The vectorized version of calc_loss is used to 
    to get the batch loss.
    """
    loss_vals = vmap_calc_loss(model, init_state, input_batch, target, loss_fn)
    loss_val = loss_vals.sum() # summing over all losses of the batch
    return loss_val

def calc_loss_and_grads(model, init_state, input_batch, target, loss_fn, key):
    """
    This function uses the filter_value_and_grad feature of equinox to calculate the
    value of the batch loss as well as it's gradients w.r.t. the models parameters.
    """
    loss_val, grad = eqx.filter_value_and_grad(calc_batch_loss)(model, init_state, input_batch, target, loss_fn)
    print(grad)
    return loss_val, grad

def update(calc_loss_and_grads, optim, loss_fn, model, opt_state, input_batch, target, key):
    """
    Function to calculate the update of the model and the optimizer based
    on the calculated updates.
    """
    init_key, grad_key = jax.random.split(key)
    model_states = model.init_state(INPUT_SHAPE, init_key)
    loss_value, grads = calc_loss_and_grads(model, model_states, input_batch, target, loss_fn, grad_key) 

    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

### Functions for calculating the accuracy
@ft.partial(eqx.filter_jit, filter_spec=eqx.is_array)
@ft.partial(jax.vmap, in_axes=(None, None, 0))
def predict(model, state, data):
    return model(state, data)

@ft.partial(eqx.filter_jit, filter_spec=eqx.is_array)
def calc_accuracy(model, state, data, target, normalized=True):
    _, outs = predict(model, state, data)
    pred = tree_map(lambda x: jnp.sum(x, axis=1), outs[-1]).argmax(axis=-1) # sum over spikes

    if normalized:
        return (pred == target).mean()
    else:
        return (pred == target).sum()

key1, key2, key3, init_key, key = jrandom.split(key, 5)

layers = [
    eqx.nn.Linear(16, 16, key=key1, use_bias=False),
    snn.SimpleLIF([.95], "superspike"),

    eqx.nn.Linear(16, 16, key=key2, use_bias=False),
    snn.SimpleLIF([.95], "superspike"),

    eqx.nn.Linear(16, N_CLASSES, key=key3, use_bias=False),
    snn.SimpleLIF([.95], "superspike")
]

graph_structure = SNNGraphStructure(
    6, [0], [5], 
    [[], [0], [1], [2], [3], [4]], 
)

model = snn.SNNModel(graph_structure, layers)

loss_vals, accuracies, accuracies_train = [], [], []
    
optim = optax.adam(1e-2)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

loss_fn = one_hot_cross_entropy
update_method = ft.partial(update, calc_loss_and_grads, optim, loss_fn)

nbar = tqdm(range(N_EPOCHS))
for epoch in nbar:
    pbar = tqdm(dataloader_train, leave=False)
    batch_key, init_key = jax.random.split(init_key)
    for input_batch, target_batch in pbar:
        input_batch = jnp.asarray(input_batch, dtype=jnp.float32)
        target_batch = jnp.asarray(nn.one_hot(target_batch, N_CLASSES), dtype=jnp.float32)
        
        model, opt_state, loss_value = eqx.filter_jit(update_method)(model, opt_state, input_batch, target_batch, tree_map(jnp.asarray, batch_key))
        loss_vals.append(loss_value/BATCH_SIZE)
        
        pbar.set_description(f"Loss: {loss_value/BATCH_SIZE}")

    acc_val = calc_accuracy(model, model.init_state(INPUT_SHAPE, batch_key), jnp.asarray(randman.test_data, dtype=jnp.float32), jnp.asarray(randman.test_labels, dtype=jnp.float32))
    acc_train = calc_accuracy(model, model.init_state(INPUT_SHAPE, batch_key), jnp.asarray(randman.train_data, dtype=jnp.float32), jnp.asarray(randman.train_labels, dtype=jnp.float32))

    nbar.set_description(f"epoch: {epoch}, loss = {np.mean(loss_vals)}, accuracy_train = {acc_train}, accuracy_val = {acc_val}")
    accuracies.append(acc_val)
    accuracies_train.append(acc_train)

