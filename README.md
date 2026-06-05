# Abstract

Spiking Neural Networks (SNNs) have emerged as an energy-efficient and biologically plausible paradigm. However, the inherent binary and sparse nature of spike representations limits their expressive capacity, resulting in ambiguous and entangled features that introduce spurious correlations between object semantics and background interference. To address these issues, we propose the Spiking Causal Deconfounded Network (SCD-Net), a novel causal deconfounding framework for energy-efficient SNNs. Specifically, SCD-Net introduce a Spiking Causal Intervention (SCI) Module that performs explicit causal intervention on spatiotemporal spike representations, suppressing confounding effects while preserving authentic object semantics. Furthermore, a Causal Necessity and Sufficiency Loss (CNSL) is proposed to optimize the intervention process, facilitating the learning of deconfounded representations. By jointly optimizing the SCI Module and the CNSL, SCD-Net mitigates spurious correlations and captures intrinsic causal representations. Notably, SCD-Net achieves state-of-the-art (SOTA) performance 97.19\% on CIFAR10 while significantly reducing energy consumption by 15.8\%. Moreover, our approach exhibits 66.56\% ($\textbf{+0.63\%}$) accuracy on Tiny-ImageNet with 24\% energy reduction and 86.60\% ($\textbf{+0.76\%}$) on ImageNet-100.

# SCD-Net

This repository contains the code for SCD-Net and QK-SCD-Net experiments on static image datasets and event-based datasets. The checkpoint files (`*.pth`, `*.pth.tar`) and TensorBoard event files are intentionally excluded from the repository.

## Environment

The dependency list was prepared from the `zcy` conda environment, focusing on SNN and training-related packages.

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

The `output/`, `log/`, and `checkpoint/` folders may contain experiment records or placeholders. Model weight files are not included.

## Data Preparation

Update the dataset paths in the YAML files or override them from the command line. The most common command-line arguments are:

- CIFAR-style scripts: `--data-path` or `-data-dir`
- DVS scripts: `--data-path`
- ImageNet-style scripts: `--data_path`

For ImageNet-style scripts, set `NPROC_PER_NODE` according to the number of GPUs available.

## Causal Training Commands

Only the causal main training commands are listed here. Ablation scripts, metric-only scripts, heatmap scripts, RIE scripts, firing-rate scripts, and baseline commands are intentionally omitted.

### QK-SCD-Net

#### CIFAR-10

```bash
cd QK-SCD-Net/cifar10
python train_causal.py -c cifar10.yml -data-dir /path/to/cifar10
```

#### CIFAR-100

```bash
cd QK-SCD-Net/cifar100/causal
python train_causal.py -c cifar100.yml -data-dir /path/to/cifar100
```

#### CIFAR10-DVS

```bash
cd QK-SCD-Net/cifar10-dvs/causal
python train_causal.py --data-path /path/to/cifar10dvs --output-dir ./output/train_causal_cifar10dvs
```

#### DVS128 Gesture

```bash
cd QK-SCD-Net/dvs-128
python train.py --data-path /path/to/DVS128Gesture --output-dir ./output/train_causal_dvs128
```

#### ImageNet-100

```bash
cd QK-SCD-Net/imagenet/qkformer_causal_imagenet100
NPROC_PER_NODE=2 torchrun --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train.py \
  --config ./conf/10-512-t4.yml \
  --data_path /path/to/imagenet100 \
  --output_dir ./output/imagenet100_qkformer_10_512_T4 \
  --log_dir ./log/imagenet100_qkformer_10_512_T4 \
  --dist_eval --pin_mem
```

#### Tiny-ImageNet

```bash
cd QK-SCD-Net/imagenet/qkformer_causal_tinyimagenet
NPROC_PER_NODE=2 torchrun --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train1.py \
  --config ./conf/10-512-t4.yml \
  --data_path /path/to/tiny-imagenet-200 \
  --output_dir ./output/tiny_imagenet_qkformer_10_512_T4 \
  --log_dir ./log/tiny_imagenet_qkformer_10_512_T4 \
  --dist_eval --pin_mem
```

### SCD-Net

#### CIFAR-10

```bash
cd SCD-Net/cifar10/causal
python train_causal_10.py -c cifar10.yaml --data-path /path/to/cifar10 --dataset torch/cifar10
```

#### CIFAR10-DVS

```bash
cd SCD-Net/cifar10-dvs
python train_causal_cifar10dvs.py -c cifar10dvs.yaml --data-path /path/to/cifar10dvs --model max_former_causal
```

#### DVS128 Gesture

```bash
cd SCD-Net/dvs-128
python train_causal_dvs128.py -c dvsgesture.yaml --data-path /path/to/DVS128Gesture --dataset dvsgesture --num-classes 11 --model max_former
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
  --dist_eval --pin_mem
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
  --dist_eval --pin_mem
```

## Notes

- Replace all `/path/to/...` placeholders with local dataset paths.
- The number of GPUs can be changed by editing `NPROC_PER_NODE`.
- Checkpoints should be stored locally and are ignored by Git.
