#!/bin/python
#-----------------------------------------------------------------------------
# File Name : datasets.py
# Author: Emre Neftci
#
# Creation Date : Thu 01 Jun 2023 01:05:51 PM CEST
# Last Modified : Tue 23 Apr 2024 03:54:16 PM CEST
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import warnings

'''
This file contains functions that return dataloaders for various datasets, in a format that is compatible with the train_dataset_model.py script.

The functions should have the following signature:
    def dataset_name(batch_size, dt=None, root="./data", **dl_kwargs):
        #define the dataset and dataloaders
        return dataloader_train, dataloader_test, dataloader_val, input_size, output_size

Note that batch_size argumetn may not be necessary in certain datasets
'''

def dvs_gestures(batch_size=72, dt=1e-3, steps_per_dt = None, ds=4, n_events_attention=None, seqlen_train=500, seqlen_test=1800, num_workers=8, root="./data", **dl_kwargs):
    '''
    This function returns the dataloaders for the DVS Gestures dataset, using the torchneuromorphic library. 
    The dataset is described in:
    "A dataset and benchmark for large-scale event-based gesture recognition"

    **Arguments:**
    - batch_size: int, the batch size for the dataloaders
    - dt: float, the time step for the dataset in ms
    - steps_per_dt: int, the number of steps per dt
    - ds: int, the spatial downsampling factor for the dataset
    - n_events_attention: int, the number of events to use for the attention mechanism
    - seqlen_train: int, the sequence length for the training set
    - seqlen_test: int, the sequence length for the test set
    - num_workers: int, the number of workers for the dataloaders
    - root: str, the root directory for the dataset
    - dl_kwargs: dict, additional keyword arguments for the neuromorphic dataloaders (as using in the torchneuromorphic library)
    '''
    if steps_per_dt is None: steps_per_dt = int(dt*1e3)
    if n_events_attention is not None:
        print('using attention')

    import torchneuromorphic.dvs_gestures.dvsgestures_dataloaders as create_dataloader
    dataloader_train, _ = create_dataloader.create_dataloader(  chunk_size_train=seqlen_train//steps_per_dt,
                                                                chunk_size_test=seqlen_test//steps_per_dt,
                                                                dt=int(dt*1e6), num_workers=num_workers,
                                                                ds=ds,
                                                                batch_size=batch_size,
                                                                n_events_attention = n_events_attention,
                                                                target_transform_train =  lambda x:x,
                                                                target_transform_test  =  lambda x:x,
                                                                drop_last = True,
                                                                root=root,
                                                                **dl_kwargs)
    _, dataloader_test  = create_dataloader.create_dataloader(  chunk_size_train=seqlen_train//steps_per_dt,
                                                                chunk_size_test=seqlen_test//steps_per_dt,
                                                                dt=int(dt*1e6), num_workers=num_workers,                                                          
                                                                ds=ds,
                                                                batch_size = batch_size,
                                                                n_events_attention = n_events_attention,
                                                                target_transform_train =  lambda x:x,
                                                                target_transform_test  =  lambda x:x, 
                                                                drop_last = True,
                                                                time_shuffle = False,
                                                                root=root,
                                                                **dl_kwargs)
    if n_events_attention is not None:
        input_size = (2, 64//ds, 64//ds) #64 due to attention window
    else: 
        input_size = (2, 128//ds, 128//ds)
    output_size = 11
    return dataloader_train, dataloader_test, None, input_size, output_size

def dvs_gestures_attn(n_events_attention=1000, *args, **kwargs):
    '''
    This function is a wrapper for the dvs_gestures function, with the n_events_attention argument set to a default value.
    '''
    return dvs_gestures(n_events_attention=n_events_attention, *args, **kwargs)

def nmnist(batch_size=256, dt=1e-3, steps_per_dt = None, ds=1, n_events_attention=None, seqlen_train=300, seqlen_test=300, num_workers=8, root="./data/nmnist/n_mnist.hdf5", **dl_kwargs):
    '''
    This function returns the dataloaders for the NMNIST dataset, using the torchneuromorphic library.

    **Arguments:**
    - batch_size: int, the batch size for the dataloaders
    - dt: float, the time step for the dataset in ms
    - steps_per_dt: int, the number of steps per dt
    - ds: int, the spatial downsampling factor for the dataset
    - n_events_attention: int, the number of events to use for the attention mechanism
    - seqlen_train: int, the sequence length used for the training set
    - seqlen_test: int, the sequence length used for the test set
    - num_workers: int, the number of workers for the dataloaders
    - root: str, the root directory for the dataset
    - dl_kwargs: dict, additional keyword arguments for the neuromorphic dataloaders (as using in the torchneuromorphic library)
    '''

    if steps_per_dt is None:
        steps_per_dt = int(dt*1e3)

    if n_events_attention is not None:
        print('using attention')

    if seqlen_train > 300:
        seq_len_train = 300
        warnings.warn('seqlen_train > 300, setting to 300')

    if seqlen_test > 300:
        seq_len_test = 300
        warnings.warn('seqlen_test > 300, setting to 300')
    
    import torchneuromorphic.nmnist.nmnist_dataloaders as create_dataloader
    dataloader_train, _ = create_dataloader.create_dataloader(chunk_size_train=seqlen_train//steps_per_dt,
                                                              chunk_size_test=seqlen_test//steps_per_dt,
                                                              dt=int(dt*1e6), num_workers=num_workers,
                                                              ds=ds,
                                                              batch_size=batch_size,
                                                              target_transform_train =  lambda x:x,
                                                              target_transform_test  =  lambda x:x,
                                                              drop_last = True,
                                                              root=root,
                                                              )
    _, dataloader_test  = create_dataloader.create_dataloader(chunk_size_train=seqlen_train//steps_per_dt,
                                                              chunk_size_test=seqlen_test//steps_per_dt,
                                                              dt=int(dt*1e6), num_workers=num_workers,
                                                              ds=ds,
                                                              batch_size=batch_size,
                                                              target_transform_train =  lambda x:x,
                                                              target_transform_test  =  lambda x:x, 
                                                              drop_last = True,
                                                              root=root,
                                                              )
    input_size = (2, 32//ds, 32//ds) 
    output_size = 10
    return dataloader_train, dataloader_test, None, input_size, output_size

def double_nmnist(num_tasks=32,
                  num_ways=5,
                  num_shots=1,
                  num_shots_test=5,
                  dt=1e-3,
                  steps_per_dt=None,
                  ds=2,
                  seqlen_train=300,
                  seqlen_test=300,
                  num_workers=8,
                  root="./data/nmnist/n_mnist.hdf5",
                  **dl_kwargs):

    '''
    Meta learning used in the paper Stewart and Neftci, 2022: "Meta-learning spiking neural networks with surrogate gradient descent"
    DOI 10.1088/2634-4386/ac8828
    
    Usage:
    >>> tr,te,val, inp, out = double_nmnist()
    >>> tr[(3,5,56,21,2)]['train'][0] # return a single sample of a 5-way task with classes (3,5,56,21,2) from the training set

    Arguments:
    - num_ways: int, number of classes in the task
    - num_shots: int, number of samples per class in the training set
    - num_shots_test: int, number of samples per class in the test set
    - dt: float, the time step for the dataset in ms
    - steps_per_dt: int, the number of steps per dt
    - ds: int, the spatial downsampling factor for the dataset
    - seqlen_train: int, the sequence length used for the training set
    - seqlen_test: int, the sequence length used for the test set
    - num_workers: int, the number of workers for the dataloaders
    - root: str, the root directory for the dataset
    - dl_kwargs: dict, additional keyword arguments for the neuromorphic dataloaders (as using in the torchneuromorphic library)

    '''
    from torchneuromorphic.doublenmnist_torchmeta.doublenmnist_dataloaders import DoubleNMNIST,Compose,ClassNMNISTDataset,CropDims,Downsample,ToCountFrame,ToTensor,ToEventSum,Repeat,toOneHot, ToJNPTensor
    from torchmeta.transforms import ClassSplitter, Categorical, Rotation
    from torchmeta.utils.data import BatchMetaDataLoader

    if steps_per_dt is None:
        steps_per_dt = int(dt*1e3)

    bin_size_train = seqlen_train//steps_per_dt
    bin_size_test = seqlen_test//steps_per_dt

    batch_size = num_ways
    
    ds = 2
    transform = None
    target_transform = None
    input_size = [2, (32//ds), (32//ds)]
    output_size = [2, 2*(32//ds), (32//ds)]
        
    num_ways_train = num_ways
    num_ways_val = num_ways
    num_ways_test = num_ways

        
    transform_train = Compose([
        CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
        Downsample(factor=[int(dt*1e6),1,ds,ds]),
        ToCountFrame(T = bin_size_train, size = input_size),
        ToTensor()])

    transform_test = Compose([
        CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
        Downsample(factor=[int(dt*1e6),1,ds,ds]),
        ToCountFrame(T = bin_size_test, size = input_size),
        ToTensor()])
   
    meta_train_dataset = ClassSplitter(DoubleNMNIST(root = root,
                                                    meta_train=True,
                                                    dt=int(dt*1e6),
                                                    transform = transform_train,
                                                    target_transform = Categorical(num_ways),
                                                    chunk_size=bin_size_train,
                                                    num_classes_per_task=num_ways_train), 
                                       num_train_per_class = num_shots, 
                                       num_test_per_class = num_shots_test)
    
    meta_val_dataset = ClassSplitter(DoubleNMNIST(root = root,
                                                  meta_val=True,
                                                  dt=int(dt*1e6),
                                                  transform = transform_test,
                                                  target_transform = Categorical(num_ways_test),
                                                  chunk_size=bin_size_test,
                                                  num_classes_per_task=num_ways_val),
                                     num_train_per_class = num_shots,
                                     num_test_per_class = num_shots_test)
    
    meta_test_dataset = ClassSplitter(DoubleNMNIST(root = root,
                                                   meta_test=True,
                                                   dt=int(dt*1e6),
                                                   transform = transform_test,
                                                   target_transform = Categorical(num_ways_val),
                                                   chunk_size=bin_size_test,
                                                   num_classes_per_task=num_ways_val), 
                                      num_train_per_class = num_shots, 
                                      num_test_per_class = num_shots_test)
    
    meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True, **dl_kwargs)

    meta_test_dataloader = BatchMetaDataLoader(meta_test_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=True, **dl_kwargs)

    meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=True, **dl_kwargs)

    return meta_train_dataloader, meta_test_dataloader, meta_val_dataloader, output_size, num_ways


def heidelberg_spoken_digits(batch_size, dt=1e-3, steps_per_dt = None, ds=1, num_workers=8, root="./data/SHD/", use_augm=False,  **dl_kwargs):
    if steps_per_dt is None:
        steps_per_dt = ds = int(dt*1e3) 
    if use_augm:
        print("using augmentations")
    from custom_datasets.datasets_bittar_garner_hd_sc import load_hd_or_sc
    dt_tr = load_hd_or_sc('hd','data/SHD','train', batch_size=batch_size, ds=ds, workers=num_workers, use_augm = use_augm)
    dt_te = load_hd_or_sc('hd','data/SHD','test',  batch_size=batch_size, ds=ds, workers=num_workers, use_augm = use_augm)
    return dt_tr, dt_te, None, [40], 20

def spiking_heidelberg_spoken_digits(batch_size, dt=1e-3, steps_per_dt = None, ds=1, num_workers=8, root="./data/SHD/",  **dl_kwargs):
    if steps_per_dt is None:
        steps_per_dt = ds = int(dt*1e3) 
    from custom_datasets.datasets_bittar_garner_shd_ssc import load_shd_or_ssc
    dt_tr = load_shd_or_ssc('shd','data/SHD','train', batch_size=batch_size, ds=ds, workers=num_workers)
    dt_te = load_shd_or_ssc('shd','data/SHD','test',  batch_size=batch_size, ds=ds, workers=num_workers)
    return dt_tr, dt_te, None, [700], 20




