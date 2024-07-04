#!/bin/python
#-----------------------------------------------------------------------------
# File Name : utils_training.py
# Author: Emre Neftci
#
# Creation Date : Wed 24 Apr 2024 10:15:25 AM CEST
# Last Modified : Fri 26 Apr 2024 09:24:53 PM CEST
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import equinox as eqx
import numpy as np
from typing import *
import jax
import jax.numpy as jnp

def xent(y_pred, y_hat, num_classes=10):
    """
    y_pred: jnp array of predictions
    y_hat: jnp array of targets (categorical)
    """
    y1h = jax.nn.one_hot(y_hat, num_classes=num_classes)
    return  -jnp.sum(y1h*jax.nn.log_softmax(y_pred)) / len(y_pred)

def mse(y_pred, y_hat):
    """
    y_pred: jnp array of predictions
    y_hat: jnp array of targets (dense)
    """
    return jnp.mean((y_hat-y_pred)**2)

def mse_cat(y_pred, y_hat, num_classes=10):
    """
    y_pred: jnp array of predictions
    y_hat: jnp array of targets (categorical)
    """
    y1h = jax.nn.one_hot(y_hat, num_classes=num_classes)
    return jnp.mean((y1h-y_pred)**2)

def create_cls_func_xent(model, optim, filter_spec, num_classes):
    '''
    Default classification functions using the cross entropy loss for classical neural networks (no time component)
    Y 
    ^ 
    X 
    '''
    import equinox as eqxac
    import jax
    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, y, *z = batch
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x, key = rng_key)
        y1h = jax.nn.one_hot(y, num_classes=num_classes)
        return -jnp.mean(y1h*jax.nn.log_softmax(pred_y))

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        #Needed because counter is not jittable
        x, y, *z = batch
        return jax.vmap(model)(x, rng_key) 

    def accuracy(model, batch, rng_key):
        x, y, *z = batch
        pred_y = _run_accuracy(model, batch, rng_key).argmax(axis=-1)
        return np.mean(np.array(pred_y) == y)
    return compute_loss, make_step, accuracy

def create_cls_func_multixent(model, optim, filter_spec, num_classes, sparsity_loss = False):
    '''
    Sequence classification functions using the cross entropy loss for classical neural networks, loss applied every output
     Y  Y
     ^  ^
    X1 X2  
    '''
    import equinox as eqx
    import jax
    from collections import Counter, defaultdict 
    
    
    if sparsity_loss:
        @eqx.filter_value_and_grad
        def compute_loss(diff_model, static_model, batch, rng_key):
            x, y, *z = batch
            model = eqx.combine(diff_model, static_model)
            state, out = jax.vmap(model.get_final_states)(x, key = rng_key)
            pred_y = jax.vmap(model.multiloss)(out)
            y1h = jax.nn.one_hot(y, num_classes=num_classes)
            y11h = jnp.tile(y1h,(pred_y.shape[1],1,1)).swapaxes(0,1) 
            act = [o[-1].mean() for o in out[:-1]]
            vact = [jnp.mean((o[-2])**2) for o in out[:-1]]
            xent_loss = -jnp.mean(y11h*jax.nn.log_softmax(pred_y))
            return xent_loss +sum(act)+.2*sum(vact)

    else:
        @eqx.filter_value_and_grad
        def compute_loss(diff_model, static_model, batch, rng_key):
            x, y, *z = batch
            model = eqx.combine(diff_model, static_model)
            pred_y = jax.vmap(model)(x, key = rng_key)
            y1h = jax.nn.one_hot(y, num_classes=num_classes)
            y11h = jnp.tile(y1h,(pred_y.shape[1],1,1)).swapaxes(0,1) 
            xent_loss = -jnp.mean(y11h*jax.nn.log_softmax(pred_y)) 
            return xent_loss

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        x, y, *z = batch
        #Needed because counter is not jittable
        return jax.vmap(model)(x, rng_key) 

    def accuracy(model, batch, rng_key):
        x, y, *z = batch
        pred_y = _run_accuracy(model, batch, rng_key)
        maxs = pred_y.argmax(axis=-1) 
        pred = [] 
        for m in maxs: 
            pred.append(Counter(np.array(m)).most_common(1)[0][0])
        return np.mean(np.array(pred) == y)
    return compute_loss, make_step, accuracy

def create_func_seq2seq_xent(model, optim, filter_spec, num_classes):
    import equinox as eqx
    import jax
    from collections import Counter, defaultdict 
    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, y, *z = batch
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x, key = rng_key)
        y1h = jax.nn.one_hot(y, num_classes=num_classes)
        xent_loss = -jnp.mean(y1h*jax.nn.log_softmax(pred_y)) 
        return xent_loss

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        x, y, *z = batch
        #Needed because counter is not jittable
        return jax.vmap(model)(x, rng_key) 

    def accuracy(model, batch, rng_key):
        x, y, *z = batch
        pred_y = _run_accuracy(model, batch, rng_key)
        maxs = pred_y.argmax(axis=-1) 
        
        return np.mean(maxs == y)
    return compute_loss, make_step, accuracy

def create_func_seq2seq_mse(model, optim, filter_spec, output_size):
    import equinox as eqx
    import jax
    from collections import Counter, defaultdict 
    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, y, *z = batch
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x, key = rng_key)
        mse_loss = jnp.mean((y-pred_y)**2) 
        return mse_loss

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        x, y, *z = batch
        #Needed because counter is not jittable
        return jax.vmap(model)(x, rng_key) 

    def accuracy(model, batch, rng_key):
        x, y, *z = batch
        pred_y = _run_accuracy(model, batch, rng_key)
        errors = (pred_y-y)**2 
        
        return np.mean(errors)
    return compute_loss, make_step, accuracy

def create_func_seq2seq_mse_lastpointacc(model, optim, filter_spec, output_size):
    import equinox as eqx
    import jax
    from collections import Counter, defaultdict 
    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, y, *z = batch
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x, key = rng_key)
        mse_loss = jnp.mean((y-pred_y)**2) 
        return mse_loss

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        x, y, *z = batch
        #Needed because counter is not jittable
        return jax.vmap(model)(x, rng_key) 

    def accuracy(model, batch, rng_key):
        x, y, *z = batch
        pred_y = _run_accuracy(model, batch, rng_key)
        errors = (pred_y[:,-1]-y[:,-1])**2 
        return np.mean(errors)
    return compute_loss, make_step, accuracy

def create_func_seq2seq_xent(model, optim, filter_spec, num_classes):
    '''
    Sequence classification functions using the cross entropy loss for neural networks
    TODO: draw diagram
    '''
    import equinox as eqx
    import jax
    from collections import Counter, defaultdict 
    from optax import softmax_cross_entropy_with_integer_labels as xent

    def _loss_fn(model, x, labels, rng_key):
        logits = model(x, key=rng_key)
        return xent(logits, labels, num_classes=num_classes).sum(), logits
    
    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, labels, *z = batch
        model = eqx.combine(diff_model, static_model)
        losses, outs = jax.vmap(_loss_fn, in_axes=(None, 0, 0, 0))(model, x, labels, rng_key)
        return losses.mean()

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def accuracy(model, batch, rng_key, normalize=True):
        x, labels, *z = batch
        logits = jax.vmap(model)(x, key = rng_key)

        if logits.ndim == 3:
            assert labels.ndim == 2
            logits = logits.reshape((-1, logits.shape[-1]))
            labels = labels.reshape((-1,))
        reduce_fn = jnp.mean if normalize else jnp.sum
        return reduce_fn(jnp.argmax(logits, axis=-1) == labels) 

    return compute_loss, make_step, accuracy

def create_cls_func_xent_lastpoint(model, optim, filter_spec, num_classes):
    '''
    Sequence classification functions using the cross entropy loss for classical neural networks
        Y
     ^  ^
    X1 X2
    '''
    import equinox as eqxac
    import jax
    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, y, *z = batch
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x, key = rng_key)
        y1h = jax.nn.one_hot(y, num_classes=num_classes)
        return -jnp.mean(y1h*jax.nn.log_softmax(pred_y[:,-1]))

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        #Needed because counter is not jittable
        x, y, *z = batch
        return jax.vmap(model)(x, rng_key)[:,-1] 

    def accuracy(model, batch, rng_key):
        x, y, *z = batch
        pred_y = _run_accuracy(model, batch, rng_key).argmax(axis=-1)
        return np.mean(np.array(pred_y) == y)
    return compute_loss, make_step, accuracy

def create_cls_func_xent_variable_seqlen(model, optim, filter_spec, num_classes):
    '''
    Sequence classification functions using the cross entropy loss for neural networks, where the sequence length is variable and the datset returns the sequence length in Z
               Y
     ^  ^      ^
    X1 X2 ... XZ

    assumes first element of z is the sequence length
    assumes model will return element at seqlen, by passing seqlen
    '''
    import equinox as eqxac
    import jax

    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, y, *z = batch
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model.get_cumsum)(x, key = rng_key, seqlen=z[0])
        y1h = jax.nn.one_hot(y, num_classes=num_classes)
        return -jnp.mean(y1h*jax.nn.log_softmax(pred_y))

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        #Needed because counter is not jittable
        x, y, *z = batch
        return jax.vmap(model.get_cumsum)(x, key = rng_key, seqlen=z[0]) 

    def accuracy(model, batch, rng_key):
        pred_y = _run_accuracy(model, batch, rng_key).argmax(axis=-1)
        x, y, *z = batch
        return np.mean(np.array(pred_y) == y)
    return compute_loss, make_step, accuracy


