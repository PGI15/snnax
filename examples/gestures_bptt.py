import os
import argparse
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.random as jrand
from jax.tree_util import tree_map

import optax
import snnax.snn as snn
from torch.utils.data import DataLoader
from snnax.functional import one_hot_cross_entropy
import equinox as eqx

from tonic.transforms import Compose, Downsample, ToFrame

from utils import calc_accuracy, DVSGestures, RandomSlice

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
                    type=int, default=48)
parser.add_argument("--lr", help="Learning rate of the optimizer.", 
                    type=float, default=1e-3)
parser.add_argument("--save", 
                    help="Set this flag to save model.", 
                    action='store_true')
parser.add_argument("--timesteps", 
                    help="Number of BPTT steps.", 
                    type=int, default=1000)
parser.add_argument("--seed", 
                    help="Rng seed.", 
                    type=int, default=123)
args = parser.parse_args()

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Define training dataset and test dataset
EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
T = args.timesteps
TEST_T = 1798 # This is the duration of the shortest sample 
T_SCALE = 1 # Time scale in ms
SENSOR_WIDTH = 32
SENSOR_HEIGHT = 32
SCALING = .25 # .5
SENSOR_SIZE = (2, SENSOR_WIDTH, SENSOR_HEIGHT)
NCLASSES = 11
TEST_ACC = 0.
LABELS = ["hand clap",
        "right hand wave",
        "left hand wave",
        "right arm clockwise",
        "right arm counterclockwise",
        "left arm clockwise",
        "left arm counterclockwise",
        "arm roll",
        "air drums",
        "air guitar",
        "other gestures"]


# Keys for random state management
SEED = args.seed
key = jrand.PRNGKey(SEED)
init_key, key = jrand.split(key, 2)

# Downsample and ToFrames have to be applied last!
# Initial dataset size is 128x128
train_transform = Compose([Downsample(time_factor=1., 
                                        spatial_factor=SCALING),
                            ToFrame(sensor_size=(SENSOR_HEIGHT, SENSOR_WIDTH, 2), 
                                    n_time_bins=T//T_SCALE)])

train_dataset = DVSGestures("data_tonic/DVSGesture/ibmGestureTrain", 
                            sample_duration=T,
                            transform=train_transform)

train_dataloader = DataLoader(train_dataset, 
                            shuffle=True, 
                            batch_size=BATCH_SIZE, 
                            num_workers=8)

# Test data loading                             
test_transform = Compose([RandomSlice(TEST_T, seed=SEED),
                        Downsample(time_factor=1., 
                                    spatial_factor=SCALING),
                        ToFrame(sensor_size=(SENSOR_HEIGHT, SENSOR_WIDTH, 2), 
                                n_time_bins=TEST_T//T_SCALE)])

test_dataset = DVSGestures("data_tonic/DVSGesture/ibmGestureTest", 
                            transform=test_transform)

test_dataloader = DataLoader(test_dataset, 
                            shuffle=True, 
                            batch_size=BATCH_SIZE, 
                            num_workers=8)


key1, key2, key3, key4, key = jrand.split(key, 5)

model = snn.Sequential(
    eqx.nn.Conv2d(2, 32, 7, 2, key=key1, use_bias=False),
    snn.LIF([.95, .85]),
    eqx.nn.Dropout(p=.5),

    eqx.nn.Conv2d(32, 64, 7, 1, key=key2, use_bias=False),
    snn.LIF([.95, .85]),
    eqx.nn.Dropout(p=.5),

    eqx.nn.Conv2d(64, 64, 7, 1, key=key3, use_bias=False),
    snn.LIF([.95, .85]),
    eqx.nn.Dropout(p=.5),

    snn.Flatten(),
    eqx.nn.Linear(64, 11, key=key4, use_bias=False),
    snn.LIF([.95, .9]))

def calc_loss(model, init_state, data, target, loss_fn, key):
    """
    Here we define how the loss is exacly calculated, i.e. whether
    we use a sum of spikes or spike-timing for the calculation of
    the cross-entropy.
    """
    states, outs = model(init_state, data, key=key)
    # output of last layer
    final_layer_out = outs[-1]
    # sum over all spikes
    pred = tree_map(lambda x: jnp.sum(x, axis=0), final_layer_out)

    loss = loss_fn(pred, target)
    return loss

# Vectorization of the loss function calculation
vmap_calc_loss = jax.vmap(calc_loss, in_axes=(None, None, 0, 0, None, 0))

def calc_batch_loss(model, init_state, input_batch, target, loss_fn, key):
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
    loss = loss_batch.sum()
    return loss

def calc_loss_and_grads(model, init_state, input_batch, target, loss_fn, key):
    """
    This function uses the filter_value_and_grad feature of equinox to 
    calculate the value of the batch loss as well as it's gradients w.r.t. 
    the models parameters.
    """
    loss, grad = eqx.filter_value_and_grad(calc_batch_loss)(model, 
                                                            init_state, 
                                                            input_batch, 
                                                            target, 
                                                            loss_fn, 
                                                            key)
    return loss, grad

def update(model,
            optim, 
            opt_state, 
            input_batch, 
            target_batch, 
            loss_fn, 
            key):
    """
    Function to calculate the update of the model and the optimizer based
    on the calculated updates.
    """
    init_key, grad_key = jax.random.split(key)
    states = model.init_state(SENSOR_SIZE, init_key)
    loss_value, grads = calc_loss_and_grads(model, 
                                            states, 
                                            input_batch, 
                                            target_batch, 
                                            loss_fn, 
                                            grad_key)    

    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


# Initialization is important for model performance
optim = optax.adam(args.lr)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

nbar = tqdm(range(EPOCHS))
for epoch in nbar:
    losses, test_accuracies, train_accuracies = [], [], []
    
    pbar = tqdm(train_dataloader, leave=False)
    for input_batch, target_batch in pbar:
        model_key, batch_key, key = jrand.split(key, 3)
        input_batch = jnp.asarray(input_batch.numpy(), dtype=jnp.float32)
        target_batch = jnp.asarray(target_batch.numpy(), dtype=jnp.float32)
        one_hot_target_batch = jnp.asarray(nn.one_hot(target_batch, NCLASSES), 
                                            dtype=jnp.float32)

        model, opt_state, loss = \
            eqx.filter_jit(update)(model, 
                                    optim,
                                    opt_state,  
                                    input_batch,
                                    one_hot_target_batch,
                                    one_hot_cross_entropy, 
                                    model_key)
            
        losses.append(loss/BATCH_SIZE)
        
        pbar.set_description(f"loss: {loss/BATCH_SIZE}")

    # Turn off dropout for model tests
    model = eqx.tree_inference(model, True)
    pbar = tqdm(train_dataloader, leave=False)
    for input_batch, target_batch in pbar:
        batch_key, key = jrand.split(key, 2)
        input_batch = jnp.asarray(input_batch.numpy(), dtype=jnp.float32)
        target_batch = jnp.asarray(target_batch.numpy(), dtype=jnp.float32)
        train_acc = calc_accuracy(model, 
                                model.init_state(SENSOR_SIZE, batch_key), 
                                input_batch, 
                                target_batch,
                                key)

        train_accuracies.append(train_acc)

    tbar = tqdm(test_dataloader, leave=False)    
    for input_test, target_test in tbar:
        batch_key, key = jrand.split(key, 2)
        input_batch = jnp.asarray(input_test.numpy(), dtype=jnp.float32)
        target_batch = jnp.asarray(target_test.numpy(), dtype=jnp.float32)
        test_acc = calc_accuracy(model, 
                                model.init_state(SENSOR_SIZE, batch_key), 
                                input_batch, 
                                target_batch,
                                key)
        test_accuracies.append(test_acc)

    model = eqx.tree_inference(model, False)
    nbar.set_description(f"epoch: {epoch}, loss = {np.mean(losses)}, train_accuracy = {np.mean(train_accuracies):.2f}, test_accuracy = {np.mean(test_accuracies):.2f}")

