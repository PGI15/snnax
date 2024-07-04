import math, tqdm, time
import jax
import jax.lax as lax
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import optax  # https://github.com/deepmind/optax
import functools as ft
import equinox as eqx
import snnax.snn as snn
from snnax.functional.surrogate import superspike_surrogate as sr
import argparse 

parser = argparse.ArgumentParser(description='DVS Gesture SNN trained with BPTT')
parser.add_argument('-c','--config', help='YAML config file', required=True)
args = vars(parser.parse_args())


# In[3]:
class SNN(eqx.Module):
    cell: eqx.Module
    ro_int: int 
    burnin: int 

    def __init__(self, 
                 num_classes:int , 
                 alpha: float = 0.95,
                 beta: float = .85,
                 size_factor: float = 2,
                 ro_int: int = -1,
                 burnin: int = 400,
                 key = jrandom.PRNGKey,                 
                 **kwargs):
        
        ckey, lkey = jrandom.split(key)
        conn,inp,out = snn.composed.gen_feed_forward_struct(6)
        self.ro_int = ro_int
        self.burnin = burnin
        graph = snn.GraphStructure(6,inp,out,conn)

        
        key1, key2, key3, key4, key = jrandom.split(key, 5)
        surr = sr(beta = 10.0)
        self.cell = snn.StatefulModel(
        graph_structure = graph,
        layers=[
            snn.Flatten(),
            eqx.nn.Linear(32*32*2, 64*size_factor, key=key1, use_bias=True),
            snn.LIFSoftReset([alpha,beta], spike_fn=surr, reset_val=1),

            eqx.nn.Linear(64*size_factor, 64*size_factor, key=key1, use_bias=True),
            snn.LIFSoftReset([alpha,beta], spike_fn=surr, reset_val=1),
            
            eqx.nn.Linear(64*size_factor, num_classes, key=key4, use_bias=True)],
        )

    def __call__(self, x, key=None):
        state = self.cell.init_state(x[0,:].shape, key)

        state, out = self.cell(state, x, key, burnin=self.burnin)
        if self.ro_int == -1:
            ro = out[-1].shape[0]
        else:
            ro = np.minimum(self.ro_int, out[-1].shape[0])
        return out[-1][::-ro]
    
    def get_final_states(self, x, key):
        state = self.cell.init_state(x[0,:].shape, key)

        states, out = self.cell(state, x, key, burnin=self.burnin)
        return states, out


# ### Training functions

# In[4]:

def init_LSUV_actrate(act_rate, threshold=0., var=1.0):
    from scipy.stats import norm
    import scipy.optimize
    return scipy.optimize.fmin(lambda loc: (act_rate-(1-norm.cdf(threshold,loc,var)))**2, x0=0.)[0]


def get_filter_spec(model):
    ## Custom code ensures that only  conv layers are trained
    from snnax.snn.layers.stateful import StatefulLayer
    import jax.tree_util as jtu

    filter_spec = jtu.tree_map(lambda _: False, model)

    # trainable_layers = [i for i, layer in enumerate(model.cell.layers) if hasattr(layer, 'weight')]
    ## or  isinstance(layer, eqx.nn.LayerNorm)
    trainable_layers = [i for i, layer in enumerate(model.cell.layers) if isinstance(layer, eqx.nn.Linear)]

    for idx in trainable_layers:
        filter_spec = eqx.tree_at(
            lambda tree: (tree.cell.layers[idx].weight, tree.cell.layers[idx].bias),
            filter_spec,
            replace=(True,True),
        )
    return filter_spec


# In[5]:


def create_functions(model, optim, filter_spec, num_classes):
    @ft.partial(eqx.filter_value_and_grad, arg=filter_spec)
    def compute_loss(model, x, y, rng_key):
        from snnax.functional import one_hot_cross_entropy
        pred_y = jax.vmap(model)(x, key = rng_key)
        y1h = jax.nn.one_hot(y, num_classes=num_classes)
        y11h = jnp.tile(y1h,(pred_y.shape[1],1,1)).swapaxes(0,1) 
        return jax.vmap(one_hot_cross_entropy, in_axes=1)(pred_y,y11h).mean()
        # return jax.vmap(one_hot_cross_entropy)(pred_y,y1h).sum()

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(model, x, y, opt_state, rng_key):    
        loss, grads = compute_loss(model, x, y, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def run_accuracy(model, x, y, rng_key):
        #Needed because counter is not jittable
        return jax.vmap(model)(x, rng_key) 

    def accuracy(model, x, y, rng_key):
        pred_y = run_accuracy(model, x, y, rng_key)
        from collections import Counter, defaultdict 
        maxs = pred_y.argmax(axis=-1) 
        pred = [] 
        for m in maxs: 
            pred.append(Counter(np.array(m)).most_common(1)[0][0])
        return np.mean(np.array(pred) == y)
    return compute_loss, make_step, accuracy


# ## Main
if __name__ == "__main__":

    # ### Dataset

    # In[6]:
    seed=int(time.time()*100)
    
    import wandb
    wandb.init(project="nmnist_multiloss", config=args['config']) 
    
    #Following the result of hyperparameter search
    w_c = wandb.config
    learning_rate = w_c['learning_rate']
    epochs = w_c['epochs']
    
    #Script is hard-coded for DVS Gestures
    num_classes = 10
    key, model_key = jrandom.split(jrandom.PRNGKey(seed), 2)
    
    
    import torchneuromorphic.nmnist.nmnist_dataloaders as create_dataloader
    dataloader_train, _ = create_dataloader.create_dataloader(                                             
                                                                chunk_size_train=300,
                                                                chunk_size_test=300,
                                                                dt=1000, num_workers=8,
                                                                ds=1,
                                                                batch_size=256,
                                                                #channel_first = True,
                                                                target_transform_train =  lambda x:x,
                                                                target_transform_test  =  lambda x:x,
                                                                drop_last = True)
    _, dataloader_test  = create_dataloader.create_dataloader(                                             
                                                                chunk_size_train=300,
                                                                chunk_size_test=300,
                                                                dt=1000, num_workers=8,
                                                                ds=1,
                                                                batch_size=256,
                                                                #channel_first = True,
                                                                target_transform_train =  lambda x:x,
                                                                target_transform_test  =  lambda x:x, 
                                                                drop_last = True,
                                                                )
    
    data, labels = next(iter(dataloader_train))
    xs = jnp.array(data)
    ys = jnp.array(labels)

    model = SNN(num_classes = num_classes, key = model_key, **wandb.config)
    filter_spec = get_filter_spec(model)

    surr_grad_fn = lambda x: 1./(5*jnp.abs(x)+1.) 
    x0 = np.linspace(-5,5, 1000)
    delta = x0[np.where(surr_grad_fn(x0)>.2)][-1]-x0[np.where(surr_grad_fn(x0)>.2)][0]
    tgt_std = delta/2
    tgt_var = tgt_std**2
    tgt_mu = init_LSUV_actrate(w_c['act_rate'], threshold=1., var=tgt_var)


    init_key, key = jrandom.split(key, 2)
    cell = snn.init.lsuv(model.cell, xs, init_key, var_tol=0.1, mean_tol=0.1, tgt_mean=tgt_mu, tgt_var=tgt_var, max_iters=100)
    from snnax.utils.tree import apply_to_tree_leaf
    model = apply_to_tree_leaf(model,'cell',lambda x: cell);

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate/5,
        peak_value=learning_rate,
        warmup_steps=10,
        decay_steps=int(epochs * len(dataloader_train)),
        end_value=2e-2*learning_rate
    )

    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule, b1=0, b2=0.95))
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


    compute_loss, make_step, accuracy = create_functions(model, optim, filter_spec, num_classes)

    for e in tqdm.tqdm(range(epochs)):
        loss_sum = 0
        ms_key, key = jrandom.split(key,2)
        model = eqx.tree_inference(model, value=False)
        for x, y in iter(dataloader_train):
            batch_key = jrandom.split(key, len(x))
            _, key = jrandom.split(batch_key[-1], 2)
            loss, model, opt_state, grads, updates = make_step(model, jnp.array(x), jnp.array(y), opt_state, batch_key)
            loss_sum += loss.item()

        wandb.log({'train/loss':loss_sum, "epoch":e})

        if e % 1 == 0:
            acc_ = []
            model = eqx.tree_inference(model, value=True)

            for x, y in iter(dataloader_test):
                batch_key = jrandom.split(key, len(x))
                _, key = jrandom.split(batch_key[-1], 2)
                acc_.append(accuracy(model, jnp.array(x), jnp.array(y), batch_key).item())


            acc_test = np.array(acc_).mean()
            wandb.log({'val/acc':acc_test, "epoch":e})
        

