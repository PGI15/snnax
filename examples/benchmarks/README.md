# Benchmarks using the SNNAX Library
This repository contains the benchmarks used in the paper [inal paper title here].

## Installation
This project uses the (SNNAX library)[] for the implementation of the benchmarks, which itself depends on the Equinox librariy. To install the library, run the following commands:

```
pip3 install -r requirements.txt
```

## Usage
These benchmarks expect the data to be stored in the local directory `data/`. The default root directory can be changed in the parameters files under paramets/ directory. To run the benchmarks, use the following command:

```
python3 train_dataset_model.py --dataset [dataset_path] --model [model_path] --parameters [parameters_file] --funcs [path to training functions generator]
```

By default, these scripts will run on all available GPUs. To run on a specific GPU, use the following command:

```
CUDA_VISIBLE_DEVICES=[gpu_id] python3 train_dataset_model.py --dataset [dataset_path] --model [model_path] --parameters [parameters_file] --funcs [path to training functions generator] --no_parallel

```

The scripts will use wandb to log the results. To disable wandb, add the `--nowb` flag

### Examples to Train various SNN models on the DVS Gestures dataset using the SNNAX library and multi-xent loss
#### SNN MLP with Attention
```
XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_dataset_model.py -m model.snn_mlp.snn_mlp -d benchmark_datasets.dvs_gestures -c parameters/config_snn_dvsgestureattn.yaml -l utils.create_cls_func_multixent
```
#### SNN VGG4 Hybrid without Attention
```
XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_dataset_model.py -m model.snn_vgg.snn_vgg4_hybrid -d benchmark_datasets.dvs_gestures -c parameters/config_snn_dvsgesture.yaml -l utils.create_cls_func_multixent
```
#### SNN DECOLLE with Attention
```
XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_dataset_model.py -m model.snn_decolle.snn_decolle -d benchmark_datasets.dvs_gestures -c parameters/config_snn_dvsgestureattn.yaml -l utils.create_cls_func_multixent
```

### Examples to train SNN models with various neuron models on the HD dataset, using variable sequence length
#### SNN MLP with Adaptive LIF 
```
XLA_PYTHON_CLIENT_PREALLOCATE=false python3  train_dataset_model.py -m model.snn_mlp.snn_mlp -d benchmark_datasets.heidelberg_spoken_digits -c parameters/config_snnmlp_shd.yaml -l utils.create_cls_func_xent_variable_seqlen  
```
#### SNN MLP with Standard LIF with soft reset
```
# Different model arguments can be usd to tweak the yaml configuration file, like this:
XLA_PYTHON_CLIENT_PREALLOCATE=false python3  train_dataset_model.py -m model.snn_mlp.snn_mlp -d benchmark_datasets.heidelberg_spoken_digits -c parameters/config_snnmlp_shd.yaml -l utils.create_cls_func_xent_variable_seqlen  --model_kwargs neuron_model='snnax.snn.LIFSoftReset'
```

### Same on the SHD dataset
```
XLA_PYTHON_CLIENT_PREALLOCATE=false python3  train_dataset_model.py -m model.snn_mlp.snn_mlp -d benchmark_datasets.spiking_heidelberg_spoken_digits -c parameters/config_snnmlp_shd.yaml -l utils.create_cls_func_xent_variable_seqlen  
```

