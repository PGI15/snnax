import os
import functools as ft

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import equinox as eqx

from jax.tree_util import tree_map

### Functions for calculating the accuracy
@eqx.filter_jit
@ft.partial(jax.vmap, in_axes=(None, None, 0, 0))
def predict(model, state, data, key):
    return model(state, data, key=key)

@eqx.filter_jit
def calc_accuracy(model, state, data, target, key, normalized=True):
    keys = jrand.split(key, data.shape[0])
    _, out = predict(model, state, data, keys)
    # sum over spikes
    pred = tree_map(lambda x: jnp.sum(x, axis=1), out[-1]).argmax(axis=-1)

    if normalized:
        return (pred == target).mean()
    else:
        return (pred == target).sum()

class DVSGestures(object):
    def __init__(self, 
                path, 
                transform=None, 
                target_transform=None, 
                sample_duration=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        self.sample_duration = sample_duration

        if self.sample_duration is not None:
            self.sample_duration *= 1000 # Convert to microseconds

        self.file_list = []
        self.time_stamps = []
        self.dtype = np.dtype([("x", np.uint16), ("y", np.uint16), \
                                ("p", "u1"), ("t", np.uint32)])

        for usrdir in os.listdir(self.path):
            path = os.path.join(self.path, os.fsdecode(usrdir))
            for fname in os.listdir(path):
                events = np.load(os.path.join(path, fname))
                # Convert time to microseconds and cast to structured array
                events[:,3] *= 1000
                events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
                    
                duration = events["t"][-1] - events["t"][0]
                t_stamp = 0
                if self.sample_duration is not None:
                    n_slices = duration // self.sample_duration
                    padding_value = (duration % self.sample_duration) // 2
                    t_stamp = padding_value
                else:
                    n_slices = 1
                    self.sample_duration = duration

                npy_file = os.path.join(usrdir, fname)
                for i in range(n_slices):
                    self.file_list.append(npy_file)
                    self.time_stamps.append(t_stamp)
                    t_stamp += self.sample_duration

    def __getitem__(self, idx):
        events = np.load(os.path.join(self.path, self.file_list[idx]))
        # Convert timestamps to microseconds and cast to structured array
        events[:,3] *= 1000
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
        
        t_begin = self.time_stamps[idx]
        t_end = t_begin + self.sample_duration
        events_slice = events[(events["t"] >= t_begin)*(events["t"] <= t_end)]

        label = "".join(filter(str.isdigit, self.file_list[idx][-7:]))
        target = int(label)

        if self.transform:
            events_slice = self.transform(events_slice)

        if self.target_transform:
            target = self.target_transform(target)

        return events_slice, target

    def __len__(self):
        return len(self.time_stamps)
    
    
class RandomSlice(object):
    def __init__(self, length, seed):
        self.length = length * 1000 # convert to microseconds
        self.rng = np.random.default_rng(seed)

    def __call__(self, events):
        """
        Extract a random slice of length `length` milliseconds from the data.
        """
        t_0 = events[0]["t"]
        t_final = events[-1]["t"]
        t_begin = self.rng.integers(low=t_0,high=t_final-self.length, size=1)[0]
        t_end = t_begin + self.length

        timeslice = events[(events["t"] >= t_begin) * (events["t"] <= t_end)]
        return timeslice

    
    def __repr__(self):
        return self.__class__.__name__ + "()"
    
    
