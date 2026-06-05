# Abstract

Spiking Neural Networks (SNNs) have emerged as an energy-efficient and biologically plausible paradigm. However, the inherent binary and sparse nature of spike representations limits their expressive capacity, resulting in ambiguous and entangled features that introduce spurious correlations between object semantics and background interference. To address these issues, we propose the Spiking Causal Deconfounded Network (SCD-Net), a novel causal deconfounding framework for energy-efficient SNNs. Specifically, SCD-Net introduce a Spiking Causal Intervention (SCI) Module that performs explicit causal intervention on spatiotemporal spike representations, suppressing confounding effects while preserving authentic object semantics. Furthermore, a Causal Necessity and Sufficiency Loss (CNSL) is proposed to optimize the intervention process, facilitating the learning of deconfounded representations. By jointly optimizing the SCI Module and the CNSL, SCD-Net mitigates spurious correlations and captures intrinsic causal representations. Notably, SCD-Net achieves state-of-the-art (SOTA) performance 97.19\% on CIFAR10 while significantly reducing energy consumption by 15.8\%. Moreover, our approach exhibits 66.56\% ($\textbf{+0.63\%}$) accuracy on Tiny-ImageNet with 24\% energy reduction and 86.60\% ($\textbf{+0.76\%}$) on ImageNet-100.

# SCD-Net

This repository contains the code for SCD-Net and QK-SCD-Net experiments on static image datasets and event-based datasets. 

## Environment

```bash
conda create -n scd-net python=3.9 -y
conda activate scd-net
pip install -r requirements.txt
```

Key packages include PyTorch 2.1.0 with CUDA 11.8, SpikingJelly 0.0.0.0.14, timm 0.6.13, CuPy, TensorBoard, and common scientific Python packages.

If your CUDA version is different, install the matching PyTorch build from the official PyTorch index before installing the remaining packages.

## Folder Structure

```text
.
|-- QK-SCD-Net/
|   |-- cifar10/                         # QK-SCD-Net CIFAR-10 causal training
|   |-- cifar100/causal/                 # QK-SCD-Net CIFAR-100 causal training
|   |-- cifar10-dvs/causal/              # QK-SCD-Net CIFAR10-DVS causal training
|   |-- dvs-128/                         # QK-SCD-Net DVS128 Gesture training
|   `-- imagenet/
|       |-- qkformer_causal_imagenet100/ # QK-SCD-Net ImageNet-100 causal training
|       `-- qkformer_causal_tinyimagenet/# QK-SCD-Net Tiny-ImageNet causal training
|-- SCD-Net/
|   |-- cifar10/causal/                  # SCD-Net CIFAR-10 causal training
|   |-- cifar10-dvs/                     # SCD-Net CIFAR10-DVS causal training
|   |-- dvs-128/                         # SCD-Net DVS128 Gesture causal training
|   `-- imagenet/
|       |-- maxformer_causal_imagenet100/# SCD-Net ImageNet-100 causal training
|       `-- maxformer_causal_tinyimagenet/# SCD-Net Tiny-ImageNet causal training
`-- requirements.txt
```


## Data Preparation

Update the dataset paths in the YAML files or override them from the command line.

For ImageNet-style scripts, set `NPROC_PER_NODE` according to the number of GPUs available.

## Causal Training Commands

Only the causal main training commands are listed here.

### SCD-Net

#### CIFAR-10

```bash
cd SCD-Net/cifar10/causal
python SCD_Net_train_cifar10.py -c cifar10.yaml --data-path /path/to/cifar10 
```

#### CIFAR-100

```bash
cd SCD-Net/cifar100
python SCD_Net_train_cifar100.py -c cifar100.yaml --data-path /path/to/cifar100 
```

#### CIFAR10-DVS

```bash
cd SCD-Net/cifar10-dvs
python SCD_Net_train_cifar10dvs.py -c cifar10dvs.yaml --data-path /path/to/cifar10dvs 
```

#### DVS128 Gesture

```bash
cd SCD-Net/dvs-128
python SCD_Net_train_dvs128.py -c dvsgesture.yaml --data-path /path/to/DVS128Gesture 
```

#### ImageNet-100

```bash
cd SCD-Net/imagenet/maxformer_causal_imagenet100
NPROC_PER_NODE=2 torchrun --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train.py \
  -c ./conf/10_512_t4.yml \
  --dataset imagenet100 \
  --data_path /path/to/imagenet100 \
  --exp train_imagenet100_10_512_t4 \
  --output_dir ./output/train_imagenet100_10_512_t4 \
  --log_dir ./log/train_imagenet100_10_512_t4 \
```

#### Tiny-ImageNet

```bash
cd SCD-Net/imagenet/maxformer_causal_tinyimagenet
NPROC_PER_NODE=2 torchrun --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train.py \
  -c ./conf/10_512_t4.yml \
  --dataset tinyimagenet \
  --data_path /path/to/tiny-imagenet-200 \
  --exp train_tinyimagenet_10_512_t4 \
  --output_dir ./output/train_tinyimagenet_10_512_t4 \
  --log_dir ./log/train_tinyimagenet_10_512_t4 \
```

### QK-SCD-Net

#### CIFAR-10

```bash
cd QK-SCD-Net/cifar10
python QK_SCR_Net_train_cifar10.py -c cifar10.yml --data-path /path/to/cifar10dvs
```

#### CIFAR-100

```bash
cd QK-SCD-Net/cifar100/causal
python QK_SCR_Net_train_causal_cifar100.py -c cifar100.yml --data-path /path/to/cifar10dvs
```

#### CIFAR10-DVS

```bash
cd QK-SCD-Net/cifar10-dvs/causal
python QK_SCR_Net_train_cifar10dvs.py --data-path /path/to/cifar10dvs
```

#### DVS128 Gesture

```bash
cd QK-SCD-Net/dvs-128
python QK_SCR_Net_train_dvs128.py --data-path /path/to/DVS128Gesture 
```

#### ImageNet-100

```bash
cd QK-SCD-Net/imagenet/qkformer_causal_imagenet100
NPROC_PER_NODE=2 torchrun --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train.py \
  --config ./conf/10-512-t4.yml \
  --data_path /path/to/imagenet100 \
```

#### Tiny-ImageNet

```bash
cd QK-SCD-Net/imagenet/qkformer_causal_tinyimagenet
NPROC_PER_NODE=2 torchrun --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train1.py \
  --config ./conf/10-512-t4.yml \
  --data_path /path/to/tiny-imagenet-200 \
```



## Notes

- Replace all `/path/to/...` placeholders with local dataset paths.
- The number of GPUs can be changed by editing `NPROC_PER_NODE`.
