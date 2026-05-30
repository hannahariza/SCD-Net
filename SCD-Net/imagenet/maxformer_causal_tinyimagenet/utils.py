# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------


import glob
import io
import os
import sys
from typing import Iterable, Optional

import torch
import math

from timm.data import Mixup
from timm.utils import accuracy

import model.maxformer_causal_tiny_imagenet.misc as misc
from spikingjelly.clock_driven import functional

import PIL
from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch.nn.functional as F
import time
import datetime
import numpy as np

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

def get_annealed_lambdas(epoch, max_epochs, warmup_epochs=15, anneal_epochs=75):
    """动态计算因果损失(lambda_c)和稀疏损失(lambda_s)的权重"""
    max_lambda_c = 5.0
    max_lambda_s = 5e-4

    if epoch is None:
        return max_lambda_c, max_lambda_s

    if epoch < warmup_epochs:
        return 0.0, 0.0
    elif epoch < warmup_epochs + anneal_epochs:
        progress = (epoch - warmup_epochs) / anneal_epochs
        return max_lambda_c * progress, max_lambda_s * progress
    else:
        return max_lambda_c, max_lambda_s

def get_elapsed_time(start_time):
    """计算自起始时间以来的相对耗时"""
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def get_causal_progress(epoch, warmup_epochs=15, anneal_epochs=75):
    if epoch is None or epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + anneal_epochs:
        return (epoch - warmup_epochs) / anneal_epochs
    else:
        return 1.0


def fuse_logits_by_firing(output, firing_num_t, eps=1e-6):
    """基于各时刻放电数量动态融合时间步结果"""
    if output.dim() == 2:
        return output
    assert output.dim() == 3, f"Expect output dim=2 or 3, got {output.dim()}"
    T, N, C = output.shape

    # CIFAR-10 默认传入 None，因此降级为 mean(0)
    if firing_num_t is None or firing_num_t.dim() == 1:
        return output.mean(dim=0)

    denom = firing_num_t.sum(dim=0, keepdim=True) + eps
    w = firing_num_t / denom
    return (output * w.unsqueeze(-1)).sum(dim=0)

################## LR ######################
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
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

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.stage3) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
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

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

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
            "In your zql environment install: pip install datasets pyarrow"
        )

    data_dir = _resolve_imagenet1k_data_dir(args.data_path)
    pattern = _imagenet1k_split_pattern(is_train)
    parquet_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files matched '{pattern}' under '{data_dir}'. "
            "Expected a local ImageNet-1K dataset like /root/lanyun-pub/imagenet-1k/data."
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

    if dataset_name == 'imagenet1k':
        imagefolder_root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if os.path.isdir(imagefolder_root):
            dataset = datasets.ImageFolder(imagefolder_root, transform=transform)
        else:
            dataset = build_imagenet1k_parquet_dataset(is_train=is_train, args=args, transform=transform)
    else:
        if not is_train:
            tiny_val_images = os.path.join(args.data_path, 'val', 'images')
            tiny_val_annotations = os.path.join(args.data_path, 'val', 'val_annotations.txt')
            if os.path.isdir(tiny_val_images) and os.path.isfile(tiny_val_annotations):
                dataset = TinyImageNetValDataset(args.data_path, transform=transform)
                print(f'Loaded Tiny-ImageNet validation split from {args.data_path} with {len(dataset)} samples')
                return dataset

        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
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

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
        #crop_pct = 0.95
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


########################### Train and Eval ############################

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                    max_norm=0, mixup_fn=None, log_writer=None, model_ema=None, args=None):
    model.train(True)
    import model.maxformer_causal_tiny_imagenet.misc as misc
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # 增加细分 loss 的追踪器
    metric_logger.add_meter('loss_full', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_masked', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_causal', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_sparse', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000     # 可根据需要调整日志打印频率

    # 获取当前 epoch 的退火 lambda 参数
    total_epochs = getattr(args, 'epochs', 200)
    lambda_c, lambda_s = get_annealed_lambdas(epoch, total_epochs)
    causal_progress = get_causal_progress(epoch)
    causal_active = causal_progress > 0
    
    # 安全获取 accum_iter (默认 1)
    accum_iter = getattr(args, 'accum_iter', 1)

    # ====== 确保在每个 epoch 开始前，梯度处于清空状态 ======
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # ================= 关键修复：加入学习率动态调整 =================
        # 根据当前 iteration 进度计算精确的小数 epoch，实现平滑的 Warmup 和 Cosine 衰减
        # 在使用梯度累加时，我们通常在真正 update_grad 或者每个 batch 时平滑更新 LR
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        # ================================================================
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=True):
            output_full, output_masked, mask_prob, fr_full, fr_masked, firing_rate = model(
                samples, epoch, enable_causal=causal_active
            )

            if firing_rate is None:
                step_firing_rates = torch.zeros(1, device=device)
            elif firing_rate.dim() > 1:
                step_firing_rates = firing_rate.mean(dim=1)
            else:
                step_firing_rates = firing_rate
            fr_np = step_firing_rates.detach().cpu().numpy()

            output_full_fused = fuse_logits_by_firing(output_full, None, eps=1e-6)
            loss_full = criterion(output_full_fused, targets)
            loss_masked = torch.tensor(0.0, device=device)
            loss_causal = torch.tensor(0.0, device=device)
            loss_sparse = torch.tensor(0.0, device=device)

            if causal_active:
                output_masked_fused = fuse_logits_by_firing(output_masked, None, eps=1e-6)
                loss_masked = criterion(output_masked_fused, targets)

            if causal_active and lambda_c > 0:
                T_kd = 3.0
                import torch.nn.functional as F
                log_prob_masked = F.log_softmax(output_masked_fused / T_kd, dim=-1)
                prob_full = F.softmax(output_full_fused.detach() / T_kd, dim=-1)
                loss_causal = F.kl_div(log_prob_masked, prob_full, reduction='batchmean') * (T_kd * T_kd)

            if causal_active and lambda_s > 0:
                loss_sparse = mask_prob.mean()

            loss = loss_full + causal_progress * loss_masked + lambda_c * loss_causal + lambda_s * loss_sparse

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            import sys
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # ================= 核心修复：真实的梯度累加逻辑 =================
        # a. 缩放 loss：因为要累加 accum_iter 次，每次的梯度值必须等比例缩小
        loss = loss / accum_iter

        # b. 判断当前 iteration 是否达到了需要更新权重的边界，或者是否是本 Epoch 的最后一步
        update_grad = ((data_iter_step + 1) % accum_iter == 0) or ((data_iter_step + 1) == len(data_loader))

        # c. 反向传播 (此时绝不要调用 zero_grad，让梯度累加起来)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        if loss_scaler is not None:
            # 将 update_grad 信号传递给 scaler，告诉它什么时候该 step()
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order,
                        update_grad=update_grad)
        else:
            loss.backward(create_graph=is_second_order)
            if update_grad:
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

        # d. 只有在真正执行了参数更新之后，才清空梯度！并同步更新 EMA 影子权重
        if update_grad:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
        # ================================================================

        # SNN 状态重置 (注意：无论是否更新了梯度，只要过完了一次数据，电位就必须重置)
        from spikingjelly.clock_driven import functional
        functional.reset_net(model)
        torch.cuda.synchronize()

        # 6. 更新 MetricLogger 统计信息 (记录原始未缩放的 loss_value 以保证控制台显示准确)
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_full=loss_full.item())
        metric_logger.update(loss_masked=loss_masked.item())
        metric_logger.update(loss_causal=loss_causal.item())
        metric_logger.update(loss_sparse=loss_sparse.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 生成 CIFAR-10 风格的 FR 字符串并动态附加到本次终端打印
        if isinstance(fr_np, np.ndarray):
            fr_str = "[" + ", ".join([f"{x:.4f}" for x in fr_np]) + "]"
        else:
            fr_str = f"{fr_np:.4f}"

        # 仅在主进程打印
        if data_iter_step % print_freq == 0 and misc.is_main_process():
            print(f"Train: [{epoch}] [{data_iter_step}/{len(data_loader)}]  "
                  f"L_Tot: {metric_logger.meters['loss'].global_avg:>6.4f}  "
                  f"L_Full: {metric_logger.meters['loss_full'].global_avg:>6.4f}  "
                  f"L_Mask: {metric_logger.meters['loss_masked'].global_avg:>6.4f}  "
                  f"L_Caus: {metric_logger.meters['loss_causal'].global_avg:>6.4f}  "
                  f"L_Spar: {metric_logger.meters['loss_sparse'].global_avg:>6.4f}  "
                  f"FR: {fr_str}")

    # 聚合所有进程的统计数据
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None, epoch=None): # <-- 这里的参数签名必须包含 args 和 epoch
    criterion = torch.nn.CrossEntropyLoss()
    
    # 1. 严格按照 CIFAR-10 追踪所有 loss
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # 增加细分 loss 的追踪器
    metric_logger.add_meter('loss_full', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_masked', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_causal', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_sparse', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))

    model.eval()

    # 2. 获取退火 lambda 参数 (CIFAR-10 验证阶段也会计算辅助损失)
    total_epochs = getattr(args, 'epochs', 200) if args else 200
    current_epoch = epoch if epoch is not None else total_epochs
    lambda_c, lambda_s = get_annealed_lambdas(current_epoch, total_epochs)
    causal_progress = get_causal_progress(current_epoch)
    causal_active = causal_progress > 0

    from timm.utils import accuracy
    for images, target in metric_logger.log_every(data_loader, 50, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            output_full, output_masked, mask_prob, fr_full, fr_masked, firing_rate = model(
                images, epoch, enable_causal=causal_active
            )
            
            output_full_fused = fuse_logits_by_firing(output_full, None, eps=1e-6)
            loss_full = criterion(output_full_fused, target)
            loss_masked = torch.tensor(0.0, device=device)
            loss_causal = torch.tensor(0.0, device=device)
            loss_sparse = torch.tensor(0.0, device=device)

            if causal_active:
                output_masked_fused = fuse_logits_by_firing(output_masked, None, eps=1e-6)
                loss_masked = criterion(output_masked_fused, target)

            if causal_active and lambda_c > 0:
                T_kd = 3.0
                import torch.nn.functional as F
                log_prob_masked = F.log_softmax(output_masked_fused / T_kd, dim=-1)
                prob_full = F.softmax(output_full_fused.detach() / T_kd, dim=-1)
                loss_causal = F.kl_div(log_prob_masked, prob_full, reduction='batchmean') * (T_kd * T_kd)

            if causal_active and lambda_s > 0:
                loss_sparse = mask_prob.mean()

            loss_total = loss_full + causal_progress * loss_masked + lambda_c * loss_causal + lambda_s * loss_sparse

        output = output_masked_fused if causal_active else output_full_fused

        # SNN 状态重置
        from spikingjelly.clock_driven import functional
        functional.reset_net(model)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss_total.item())
        metric_logger.update(loss_full=loss_full.item())
        metric_logger.update(loss_masked=loss_masked.item())
        metric_logger.update(loss_causal=loss_causal.item())
        metric_logger.update(loss_sparse=loss_sparse.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # 聚合多卡验证结果
    metric_logger.synchronize_between_processes()
    
    # 6. 严格复刻 CIFAR-10 的验证输出格式，包含 Causal 和 Sparse Loss
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} L_Tot {losses.global_avg:.4f} L_Caus {l_caus.global_avg:.4f} L_Spar {l_spar.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, 
                  l_caus=metric_logger.loss_causal, l_spar=metric_logger.loss_sparse))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}