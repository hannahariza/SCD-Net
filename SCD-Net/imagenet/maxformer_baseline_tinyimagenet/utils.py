# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import bisect
import io
import math
import os
import sys
import warnings
from glob import glob
from typing import Iterable, Optional

import PIL
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import model.maxformer_baseline_tinyimagenet.misc as misc
from spikingjelly.clock_driven import functional
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import accuracy

################## LR ######################
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
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

class ParquetImageNetDataset(Dataset):
    def __init__(self, data_path, split, transform):
        self.transform = transform
        shard_pattern = os.path.join(data_path, f'{split}-*.parquet')
        self.shards = sorted(glob(shard_pattern))
        if not self.shards:
            raise FileNotFoundError(f'No parquet shards found for split "{split}" under {data_path}')

        self._shard_infos = []
        self._shard_offsets = []
        total_rows = 0
        skipped_shards = []
        for shard_path in self.shards:
            try:
                parquet_file = pq.ParquetFile(shard_path)
            except Exception as exc:
                skipped_shards.append((shard_path, exc))
                continue

            row_group_offsets = []
            row_total = 0
            for row_group_index in range(parquet_file.num_row_groups):
                row_group_offsets.append(row_total)
                row_total += parquet_file.metadata.row_group(row_group_index).num_rows
            self._shard_infos.append({
                'path': shard_path,
                'num_rows': row_total,
                'row_group_offsets': row_group_offsets,
            })
            self._shard_offsets.append(total_rows)
            total_rows += row_total

        if skipped_shards:
            for shard_path, exc in skipped_shards:
                warnings.warn(f'Skipping invalid parquet shard: {shard_path} ({exc})')

        if not self._shard_infos:
            raise RuntimeError(f'No readable parquet shards found for split "{split}" under {data_path}')

        self._total_rows = total_rows
        self._cache = {'shard_index': None, 'row_group_index': None, 'rows': None}

    def __len__(self):
        return self._total_rows

    def __getitem__(self, index):
        if index < 0:
            index += self._total_rows
        if index < 0 or index >= self._total_rows:
            raise IndexError(index)

        shard_index = bisect.bisect_right(self._shard_offsets, index) - 1
        shard_info = self._shard_infos[shard_index]
        shard_local_index = index - self._shard_offsets[shard_index]

        row_group_index = bisect.bisect_right(shard_info['row_group_offsets'], shard_local_index) - 1
        row_group_start = shard_info['row_group_offsets'][row_group_index]
        row_index = shard_local_index - row_group_start

        if self._cache['shard_index'] != shard_index or self._cache['row_group_index'] != row_group_index:
            parquet_file = pq.ParquetFile(shard_info['path'])
            row_group = parquet_file.read_row_group(row_group_index, columns=['image', 'label'])
            self._cache = {
                'shard_index': shard_index,
                'row_group_index': row_group_index,
                'rows': row_group.to_pylist(),
            }

        sample = self._cache['rows'][row_index]
        image = sample['image']
        target = int(sample['label'])

        image_bytes = image.get('bytes') if isinstance(image, dict) else None
        image_path = image.get('path') if isinstance(image, dict) else None
        if image_bytes is not None:
            image = PIL.Image.open(io.BytesIO(image_bytes)).convert('RGB')
        elif image_path:
            image = PIL.Image.open(image_path).convert('RGB')
        else:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, target


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
                parts = line.strip().split('	')
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


# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)

#     if not is_train:
#         tiny_val_images = os.path.join(args.data_path, 'val', 'images')
#         tiny_val_annotations = os.path.join(args.data_path, 'val', 'val_annotations.txt')
#         if os.path.isdir(tiny_val_images) and os.path.isfile(tiny_val_annotations):
#             dataset = TinyImageNetValDataset(args.data_path, transform=transform)
#             print(f'Loaded Tiny-ImageNet validation split from {args.data_path} with {len(dataset)} samples')
#             return dataset

#     imagefolder_root = os.path.join(args.data_path, 'train' if is_train else 'val')
#     if os.path.isdir(imagefolder_root):
#         dataset = datasets.ImageFolder(imagefolder_root, transform=transform)
#         print(dataset)
#         return dataset

#     parquet_split = 'train' if is_train else 'validation'
#     parquet_pattern = os.path.join(args.data_path, f'{parquet_split}-*.parquet')
#     if glob(parquet_pattern):
#         dataset = ParquetImageNetDataset(args.data_path, parquet_split, transform=transform)
#         print(f'Loaded parquet {parquet_split} split from {args.data_path} with {len(dataset)} samples')
#         return dataset

#     raise FileNotFoundError(
#         f'Could not find an ImageFolder split at {imagefolder_root} or parquet shards matching {parquet_pattern}'
#     )

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    dataset_name = normalize_dataset_name(getattr(args, 'dataset', 'imagenet100'))

    if dataset_name == 'imagenet1k':
        # 保持原有的 ImageNet-1K 逻辑
        imagefolder_root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if os.path.isdir(imagefolder_root):
            dataset = datasets.ImageFolder(imagefolder_root, transform=transform)
        else:
            dataset = build_imagenet1k_parquet_dataset(is_train=is_train, args=args, transform=transform)
        return dataset

    # 针对 ImageNet-100 的多文件夹结构进行代码适配
    if is_train:
        # 定义可能的训练子文件夹列表
        train_sub_folders = ['train.X1', 'train.X2', 'train.X3', 'train.X4']
        datasets_list = []
        
        # 遍历检查并加载存在的文件夹
        for folder in train_sub_folders:
            path = os.path.join(args.data_path, folder)
            if os.path.isdir(path):
                datasets_list.append(datasets.ImageFolder(path, transform=transform))
        
        if datasets_list:
            # 使用 ConcatDataset 在逻辑上合并多个文件夹，无需移动物理文件
            dataset = torch.utils.data.ConcatDataset(datasets_list)
            print(f"成功合并并加载了 {len(datasets_list)} 个训练分片文件夹。")
        else:
            # 如果没找到分片，尝试寻找标准 train 文件夹
            root = os.path.join(args.data_path, 'train')
            dataset = datasets.ImageFolder(root, transform=transform)
    else:
        # 验证集逻辑：优先寻找 val.X1，否则寻找标准 val
        val_path = os.path.join(args.data_path, 'val.X1')
        if not os.path.isdir(val_path):
            val_path = os.path.join(args.data_path, 'val')
        
        dataset = datasets.ImageFolder(val_path, transform=transform)
        print(f"验证集加载路径: {val_path}")

    print(f"数据集 {dataset_name} 加载完成，样本总数: {len(dataset)}")
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
    print_freq = 2000

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    updates_per_epoch = (len(data_loader) + accum_iter - 1) // accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        update_grad = (data_iter_step + 1) % accum_iter == 0
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
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 200, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
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
