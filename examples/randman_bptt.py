import os
import argparse
import functools as ft

from tqdm import tqdm
import numpy as np
import wandb

import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.random as jrand

import optax
import snnax as snx
import snnax.snn as snn
import equinox as eqx

from jax.tree_util import tree_map
from torch.utils.data import DataLoader
from snnax.functional import one_hot_cross_entropy
from randman import RandmanGenerator, RandmanDataset
from utils import calc_accuracy

# Program arguments
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", help="Index of GPU. Check with nvidia-smi.", 
                    type=str, default="0")
parser.add_argument("--epochs", help="Number of epochs for training.", 
                    type=int, default=100)
parser.add_argument("--name", 
                    help="Folder where tensorboard stores the event files.", 
                    type=str, default="test")
parser.add_argument("--batchsize", help="Define batchsize for learning.", 
                    type=int, default=256)
parser.add_argument("--lr", help="Learning rate of the optimizer.", 
                    type=float, default=5e-3)
parser.add_argument("--timesteps", 
                    help="Number of BPTT steps.", 
                    type=int, default=100)
parser.add_argument("--seed", 
                    help="Seeding of random number generator.", 
                    type=int, default=123)
args = parser.parse_args()


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NCLASSES = 10 # Number of classes
EPOCHS = args.epochs # Number of training epochs
T = args.timesteps # Number of timesteps per epoch
NSAMPLES = NCLASSES*1000
TRAIN_TEST_SPLIT = .8
BATCH_SIZE = args.batchsize
INPUT_SHAPE = 16
SEED = args.seed 

rng = np.random.default_rng(SEED)
key = jax.random.PRNGKey(SEED)

randman = RandmanGenerator(rng, 
                            num_classes=NCLASSES, 
                            num_units=INPUT_SHAPE, 
                            num_steps=T, 
                            dim_manifold=2, 
                            num_samples=NSAMPLES, 
                            alpha=2., 
                            shuffle=True)

train_dataset = RandmanDataset(randman.train_data, randman.train_labels)
train_dataloader = DataLoader(train_dataset, 
                            batch_size=BATCH_SIZE, 
                            drop_last=False, 
                            num_workers=8)

test_dataset = RandmanDataset(randman.test_data, randman.test_labels)
test_dataloader = DataLoader(test_dataset, 
                            batch_size=BATCH_SIZE, 
                            drop_last=False, 
                            num_workers=8)


# Parameters stored for wandb
print("Saving model as", args.name)
wandb.init(project="jax-gestures")
wandb.run.name = args.name
wandb.config = {
    "epochs": EPOCHS,
    "batchsize": BATCH_SIZE,
    "learning_rate": args.lr
}


def calc_loss(model, init_state, data, target, loss_fn, key):
    """
    Here we define how the loss is exacly calculated, i.e. whether
    we use a sum of spikes or spike-timing for the calculation of
    the cross-entropy.
    """
    states, outs = model(init_state, data, key=key)
    final_layer_out = outs[-1]
    
     # sum over all spikes
    prediction = tree_map(lambda x: jnp.sum(x, axis=0), final_layer_out)

    loss_val = loss_fn(prediction, target)
    return loss_val


# Vectorization of the loss function calculation
vmap_calc_loss = jax.vmap(calc_loss, in_axes=(None, None, 0, 0, None, 0))


def calc_batch_loss(model, 
                    init_state, 
                    input_batch, 
                    target, 
                    loss_fn,
                    key):
    """
    The vectorized version of calc_loss is used to 
    to get the batch loss.
    """
    keys = jrand.split(key, input_batch.shape[0])
    loss_batch = vmap_calc_loss(model, 
                                init_state, 
                                input_batch, 
                                target, 
                                loss_fn, 
                                keys)
    loss = loss_batch.sum() # summing over all losses of the batch
    return loss


def calc_loss_and_grads(model, 
                        init_state, 
                        input_batch, 
                        target, 
                        loss_fn, 
                        key):
    """
    This function uses the filter_value_and_grad feature of equinox to 
    calculate the value of the batch loss as well as it's 
    gradients w.r.t. the models parameters.
    """
    loss_val, grad = eqx.filter_value_and_grad(calc_batch_loss)(model, 
                                                                init_state, 
                                                                input_batch, 
                                                                target, 
                                                                loss_fn,
                                                                key)
    return loss_val, grad


def update(calc_loss_and_grads, 
           optim, 
           loss_fn, 
           model, 
           opt_state, 
           input_batch, 
           target, 
           key):
    """
    Function to calculate the update of the model and the optimizer based
    on the calculated updates.
    """
    init_key, grad_key = jax.random.split(key)
    states = model.init_state(INPUT_SHAPE, init_key)
    loss, grad = calc_loss_and_grads(model, 
                                        states, 
                                        input_batch, 
                                        target, 
                                        loss_fn, 
                                        grad_key)    

    updates, opt_state = optim.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


key1, key2, key3, init_key, key = jrand.split(key, 5)
# Bias is detrimental to model performance
model = snn.Sequential(
    eqx.nn.Linear(16, 64, key=key1, use_bias=False),
    snn.LIF([.95, .85]),
    
    eqx.nn.Linear(64, 64, key=key2, use_bias=False),
    snn.LIF([.95, .85]),
    
    eqx.nn.Linear(64, NCLASSES, key=key3, use_bias=False),
    snn.LIF([.95, .85]))

init_batch = jnp.asarray(next(iter(train_dataloader))[0], dtype=jnp.float32)
model = snn.init.lsuv(model, init_batch, init_key, max_iters=125)

optim = optax.adam(args.lr)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

loss_fn = one_hot_cross_entropy
update_method = ft.partial(update, calc_loss_and_grads, optim, loss_fn)

losses = []
nbar = tqdm(range(EPOCHS))
for epoch in nbar:
    pbar = tqdm(train_dataloader, leave=False)
    batch_key, init_key, key = jax.random.split(key, 3)
    for input_batch, target_batch in pbar:
        input_batch = jnp.asarray(input_batch.numpy(), 
                                    dtype=jnp.float32)
        target_batch = jnp.asarray(nn.one_hot(target_batch.numpy(), NCLASSES), 
                                    dtype=jnp.float32)
        
        model, opt_state, loss = eqx.filter_jit(update_method)(model, 
                                                                opt_state, 
                                                                input_batch, 
                                                                target_batch, 
                                                                batch_key)
        losses.append(loss/BATCH_SIZE)
        wandb.log({"loss": loss/BATCH_SIZE})
        
        pbar.set_description(f"Loss: {loss/BATCH_SIZE}")

    test_key, train_key, key = jrand.split(key, 3)
    test_acc = calc_accuracy(model, 
                            model.init_state(INPUT_SHAPE, batch_key), 
                            jnp.asarray(randman.test_data, dtype=jnp.float32), 
                            jnp.asarray(randman.test_labels, dtype=jnp.float32),
                            test_key)
    train_acc = calc_accuracy(model, 
                              model.init_state(INPUT_SHAPE, batch_key), 
                              jnp.asarray(randman.train_data, dtype=jnp.float32), 
                              jnp.asarray(randman.train_labels, dtype=jnp.float32),
                              train_key)
    
    wandb.log({"train_accuracy": test_acc})
    wandb.log({"test_accuracy": train_acc})

    nbar.set_description(f"epoch: {epoch}, loss: {np.mean(losses)}, train accuracy: {train_acc}, test accuracy: {test_acc}")

