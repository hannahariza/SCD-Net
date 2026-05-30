# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import glob
import io
import math
import os
import sys
from typing import Iterable, Optional

import PIL
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import model.maxformer_baseline_imagenet100.misc as misc
from spikingjelly.clock_driven import functional
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import accuracy

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


################## LR ######################
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup."""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=None, layer_decay=.75):
    if no_weight_decay_list is None:
        no_weight_decay_list = []

    param_group_names = {}
    param_groups = {}

    num_layers = len(model.stage3) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id(name, num_layers):
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed1'):
        return 0
    elif name.startswith('patch_embed2'):
        return 0
    elif name.startswith('patch_embed3'):
        return 0
    elif name.startswith('stage1'):
        return 0
    elif name.startswith('stage2'):
        return 0
    elif name.startswith('stage3'):
        return num_layers
    else:
        return num_layers


########################### Datasets #############################

def normalize_dataset_name(dataset_name):
    aliases = {
        "imagenet": "imagenet100",
        "imagenet100": "imagenet100",
        "imagenet-100": "imagenet100",
        "in100": "imagenet100",
        "imagenet1k": "imagenet1k",
        "imagenet-1k": "imagenet1k",
        "imagenet_1k": "imagenet1k",
        "in1k": "imagenet1k",
        "tinyimagenet": "tinyimagenet",
        "tiny-imagenet": "tinyimagenet",
        "tiny_imagenet": "tinyimagenet",
        "tiny-imagenet-200": "tinyimagenet",
        "tiny_imagenet_200": "tinyimagenet",
        "tinyimagenet200": "tinyimagenet",
    }
    key = str(dataset_name).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of: {sorted(aliases)}")
    return aliases[key]


def infer_num_classes(dataset_name):
    normalized = normalize_dataset_name(dataset_name)
    if normalized == "imagenet1k":
        return 1000
    if normalized == "tinyimagenet":
        return 200
    return 100


def _resolve_imagenet1k_data_dir(data_path):
    if os.path.isdir(os.path.join(data_path, "data")):
        return os.path.join(data_path, "data")
    return data_path


def _imagenet1k_split_pattern(is_train):
    return "train-*.parquet" if is_train else "validation-*.parquet"


def is_tiny_imagenet_layout(data_path):
    return (
        os.path.isdir(os.path.join(data_path, 'train'))
        and os.path.isdir(os.path.join(data_path, 'val', 'images'))
        and os.path.isfile(os.path.join(data_path, 'val', 'val_annotations.txt'))
    )


class HuggingFaceImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]
        image = item["image"]
        label = int(item["label"])

        if isinstance(image, dict):
            if image.get("bytes") is not None:
                image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
            elif image.get("path"):
                image = Image.open(image["path"]).convert("RGB")
            else:
                raise ValueError("Unsupported image record in parquet dataset.")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type from parquet dataset: {type(image)}")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_imagenet1k_parquet_dataset(is_train, args, transform):
    if load_dataset is None:
        raise ImportError(
            "ImageNet-1K parquet loading requires the 'datasets' package and its parquet backend. "
            "Please install: pip install datasets pyarrow"
        )

    data_dir = _resolve_imagenet1k_data_dir(args.data_path)
    pattern = _imagenet1k_split_pattern(is_train)
    parquet_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files matched '{pattern}' under '{data_dir}'. "
            "Expected a local ImageNet-1K parquet dataset."
        )

    hf_dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    dataset = HuggingFaceImageDataset(hf_dataset=hf_dataset, transform=transform)
    print(f"Loaded ImageNet-1K parquet split={'train' if is_train else 'validation'} with {len(dataset)} samples")
    return dataset


class TinyImageNetValDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        val_root = os.path.join(data_path, 'val')
        images_root = os.path.join(val_root, 'images')
        annotations_path = os.path.join(val_root, 'val_annotations.txt')
        train_root = os.path.join(data_path, 'train')

        if not os.path.isdir(images_root):
            raise FileNotFoundError(f'Tiny-ImageNet val images directory not found: {images_root}')
        if not os.path.isfile(annotations_path):
            raise FileNotFoundError(f'Tiny-ImageNet val annotations file not found: {annotations_path}')
        if not os.path.isdir(train_root):
            raise FileNotFoundError(f'Tiny-ImageNet train directory not found: {train_root}')

        classes = sorted(
            entry for entry in os.listdir(train_root)
            if os.path.isdir(os.path.join(train_root, entry)) and not entry.startswith('.')
        )
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

        self.samples = []
        with open(annotations_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                image_name, class_name = parts[0], parts[1]
                if class_name not in self.class_to_idx:
                    continue
                image_path = os.path.join(images_root, image_name)
                if os.path.isfile(image_path):
                    self.samples.append((image_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f'No labeled Tiny-ImageNet validation samples found under {val_root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = PIL.Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    dataset_name = normalize_dataset_name(getattr(args, 'dataset', 'imagenet100'))
    data_path = os.path.abspath(args.data_path)

    if dataset_name == 'imagenet1k':
        imagefolder_root = os.path.join(data_path, 'train' if is_train else 'val')
        if os.path.isdir(imagefolder_root):
            dataset = datasets.ImageFolder(imagefolder_root, transform=transform)
            print(f"Loaded ImageFolder ImageNet-1K from {imagefolder_root}, samples: {len(dataset)}")
            return dataset
        else:
            dataset = build_imagenet1k_parquet_dataset(is_train=is_train, args=args, transform=transform)
            return dataset

    if dataset_name == 'tinyimagenet' and not is_train and is_tiny_imagenet_layout(data_path):
        dataset = TinyImageNetValDataset(data_path, transform=transform)
        print(f'Loaded Tiny-ImageNet validation split from {data_path} with {len(dataset)} samples')
        return dataset

    root = os.path.join(data_path, 'train' if is_train else 'val')
    if not os.path.isdir(root):
        raise FileNotFoundError(f"找不到指定的目录: {root}。请检查 args.data_path 是否配置正确。")

    print(f"正在从 {root} 加载 {dataset_name} 数据集...")
    dataset = datasets.ImageFolder(root, transform=transform)
    print(f"数据集加载完成，样本总数: {len(dataset)}")
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0

    size = int(args.input_size / crop_pct)
    t.append(transforms.Resize(size, interpolation=PIL.Image.BICUBIC))
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


########################### Train and Eval ############################

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, model_ema=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000

    accum_iter = getattr(args, 'accum_iter', 1)
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        update_grad = ((data_iter_step + 1) % accum_iter == 0) or ((data_iter_step + 1) == len(data_loader))

        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=update_grad,
        )

        if update_grad:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        functional.reset_net(model)

        metric_logger.update(loss=loss_value)

        max_lr = 0.
        min_lr = 10.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and update_grad:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None, epoch=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 200, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        functional.reset_net(model)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}