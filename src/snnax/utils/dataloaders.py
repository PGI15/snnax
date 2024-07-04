#!/bin/python
#-----------------------------------------------------------------------------
# File Name : 
# Author: Jamie Lohoff, Jan Finkbeiner, Emre Neftci
#
# Creation Date : Sun 21 Apr 2024 10:16:53 AM CEST
# Last Modified : Sun 21 Apr 2024 10:32:00 AM CEST
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 
'''
This is a simple implementation of a dataloader for jax, with a design similar to pytorch dataloader
'''

from typing import Any, Callable, Iterable, Sequence, Dict

import numpy as np
import jax.numpy as jnp
import math

import queue
from itertools import cycle
from multiprocessing import Process, Queue

def default_collate(batch):
    """
    TODO better element checking
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], (int, float)):
        return np.array(batch)
    if isinstance(batch[0], (list, tuple)):
        return tuple(default_collate(var) for var in zip(*batch))

class AbstractDataLoader(object):
    """
    Abstract class definition for compatibility
    and to avoid boilerplate code
    """
    dataset: Any
    batch_size: int
    prefetch_batches: int
    num_workers: int

    collate_fn: default_collate
    drop_last: bool
    shuffle: bool
    
    _state: int
    _prefetch_state: int
    _length: int

    indexes: Sequence[int]
    cache: dict

    def __init__(self,
                dataset: Any, *,
                batch_size: int = 1,
                prefetch_batches: int = 2,
                num_workers: int = 0,
                collate_fn:Callable = default_collate,
                drop_last: bool = False,
                shuffle: bool = True,
                **kwargs
                ) -> None:

        """
        **Arguments:**
        dataset: Any, the dataset to be loaded
        batch_size: int, default is 1
        collate_fn: Callable, default is default_collate which stacks numpy arrays
        drop_last: bool, default is False, if True the last batch will be dropped if it is smaller than batch_size
        shuffle: bool, default is True, if True the dataset is shuffled
        **kwargs: additional arguments passed to the dataloader (not used in abstract class)

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.num_workers = num_workers

        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self._state = 0
        self._prefetch_state = 0
        self.cache = {}

        if len(self.dataset) % self.batch_size != 0 and drop_last:
            self._length = math.floor(len(self.dataset) / self.batch_size) * self.batch_size
        else:
            self._length = len(self.dataset)

        self.indexes = list(range(self._length))
        if shuffle:
            np.random.shuffle(self.indexes)

    def __iter__(self) -> Iterable:
        self._state = 0
        self.cache = {}
        self._prefetch_state = 0
        self.prefetch()
        return self

    def __len__(self):
        return self._length // self.batch_size

    def __next__(self) -> Any:
        """
        Returns the next batch of data.
        Arguments:
        """
        if self._state >= self._length:
            raise StopIteration
        _batch_size = self.batch_size if self.drop_last else min(len(self.dataset) - self._state, self.batch_size) # last batch might be smaller
        return self.collate_fn([self.get() for _ in range(_batch_size)])

    def prefetch(self) -> None:
        pass

    def get(self) -> Any:
        raise NotImplementedError

    def __del__(self) -> None:
        pass 
        

class SimpleDataLoader(AbstractDataLoader):
    """
    Simple dataloader implementation without multiprocessing
    """
    def __init__(self,
                dataset: Any, *,
                batch_size: int = 1,
                collate_fn: Callable = default_collate,
                drop_last: bool = False,
                shuffle: bool = True,
                **kwargs
                ) -> None:


        super().__init__(dataset, 
                        batch_size=batch_size,
                        prefetch_batches=0, 
                        num_workers=0,
                        collate_fn=collate_fn, 
                        drop_last=drop_last,
                        shuffle=shuffle,
                        **kwargs)


    def get(self) -> Any:
        """
        Returns the next item in the dataset, increments the state
        """
        idx = self.indexes[self._state]
        item = self.dataset[idx]
        self._state += 1
        return item


def worker_fn(dataset, index_queue, output_queue) -> None:
    """
    Worker function, simply reads indices from state_queue, and adds the
    dataset element to the output_queue
    """
    while True:
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break
        output_queue.put((index, dataset[index]))


class DataLoader(AbstractDataLoader):
    """
    Dataloader with multiprocessing
    """
    output_queue: Queue
    index_queues: Sequence[Queue]
    workers: Sequence[Process]
    worker_cycle: cycle

    def __init__(self,
                dataset: Any, *,
                batch_size:int = 1,
                prefetch_batches: int = 2,
                num_workers: int = 1,
                collate_fn: Callable = default_collate,
                drop_last: bool = False,
                shuffle: bool = True,
                **kwargs
                ) -> None:
        """
        **Arguments:**
        dataset: Any, the dataset to be loaded
        batch_size: int, default is 1
        collate_fn: Callable, default is default_collate which stacks numpy arrays
        drop_last: bool, default is False, if True the last batch will be dropped if it is smaller than batch_size
        shuffle: bool, default is True, if True the dataset is shuffled
        num_workers: int, default is 1, number of worker processes
        prefetch_batches: int, default is 2, number of batches to prefetch
        **kwargs: additional arguments passed to the dataloader
        """

        super().__init__(dataset, 
                        batch_size=batch_size,
                        prefetch_batches=prefetch_batches, 
                        num_workers=num_workers,
                        collate_fn=collate_fn, 
                        drop_last=drop_last,
                        shuffle=shuffle)

        self.output_queue = Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = cycle(range(num_workers))

        for _ in range(self.num_workers):
            index_queue = Queue()
            worker = Process(target=worker_fn, 
                            args=(self.dataset, index_queue, self.output_queue))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

        self.prefetch()

    def prefetch(self) -> None:
        """
        Prefetches the next batch of data
        """
        while (self._prefetch_state < self._length and 
                self._prefetch_state < self._state + self.prefetch_batches * self.num_workers * self.batch_size):
            # if the prefetch_index hasn't reached the end of the dataset
            # and it is not ´prefetch_batches´ batches ahead, add indexes to the index queues
            idx = self.indexes[self._prefetch_state]
            self.index_queues[next(self.worker_cycle)].put(idx)
            self._prefetch_state += 1

    def get(self) -> Any:
        """
        Returns the next item in the dataset, increments the state
        """
        self.prefetch()
        idx = self.indexes[self._state]
        if idx in self.cache:
            item = self.cache[idx]
            del self.cache[idx]
        else:
            while True:
                try:
                    (index, data) = self.output_queue.get(timeout=0)
                except queue.Empty:  # output queue empty, keep trying
                    continue
                if index == idx:  # found our item, ready to return
                    item = data
                    break
                else:  # item isn't the one we want, cache for later
                    self.cache[index] = data

        self._state += 1
        return item

    def __del__(self) -> None:
        """
        Function for cleanup, closes the queues and terminates the workers
        """
        try:
            for i, worker in enumerate(self.workers):
                self.index_queues[i].put(None)
                worker.join(timeout=5.0)
            for queue in self.index_queues:
                queue.cancel_join_thread()
                queue.close()
            self.output_queue.cancel_join_thread()
            self.output_queue.close()
        finally:
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()

