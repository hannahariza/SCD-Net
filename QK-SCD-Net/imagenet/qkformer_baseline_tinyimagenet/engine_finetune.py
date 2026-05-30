# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import model.qkformer_baseline_tinyimagenet.util.misc as misc
import model.qkformer_baseline_tinyimagenet.util.lr_sched as lr_sched
from spikingjelly.clock_driven import functional
import torch.nn.functional as F


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

    # 如果没有传入 firing_num_t，降级为均值融合
    if firing_num_t is None or firing_num_t.dim() == 1:
        return output.mean(dim=0)

    denom = firing_num_t.sum(dim=0, keepdim=True) + eps
    w = firing_num_t / denom
    return (output * w.unsqueeze(-1)).sum(dim=0)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    # 增加细分 loss 的追踪器
    metric_logger.add_meter('loss_full', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_masked', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_causal', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_sparse', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000

    accum_iter = args.accum_iter

    # 获取当前 epoch 的退火 lambda 参数
    total_epochs = getattr(args, 'epochs', 300)
    lambda_c, lambda_s = get_annealed_lambdas(epoch, total_epochs)
    causal_progress = get_causal_progress(epoch)
    causal_active = causal_progress > 0

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            output_full, output_masked, mask_prob, fr_full, fr_masked, firing_rate = model(
                samples, epoch=epoch, enable_causal=causal_active
            )

            # 完整特征分支损失
            output_full_fused = fuse_logits_by_firing(output_full, None, eps=1e-6)
            loss_full = criterion(output_full_fused, targets)

            loss_masked = torch.tensor(0.0, device=device)
            loss_causal = torch.tensor(0.0, device=device)
            loss_sparse = torch.tensor(0.0, device=device)

            if causal_active:
                # 掩码分支损失
                output_masked_fused = fuse_logits_by_firing(output_masked, None, eps=1e-6)
                loss_masked = criterion(output_masked_fused, targets)

                # 因果散度损失 (KL Divergence)
                if lambda_c > 0:
                    T_kd = 3.0
                    log_prob_masked = F.log_softmax(output_masked_fused / T_kd, dim=-1)
                    prob_full = F.softmax(output_full_fused.detach() / T_kd, dim=-1)
                    loss_causal = F.kl_div(log_prob_masked, prob_full, reduction='batchmean') * (T_kd * T_kd)

                # 掩码稀疏度损失
                if lambda_s > 0:
                    loss_sparse = mask_prob.mean()

            # 融合总损失
            loss = loss_full + causal_progress * loss_masked + lambda_c * loss_causal + lambda_s * loss_sparse

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter

        # 梯度累加更新
        update_grad = ((data_iter_step + 1) % accum_iter == 0) or ((data_iter_step + 1) == len(data_loader))

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        functional.reset_net(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_full=loss_full.item())
        metric_logger.update(loss_masked=loss_masked.item())
        metric_logger.update(loss_causal=loss_causal.item())
        metric_logger.update(loss_sparse=loss_sparse.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_full', loss_full.item(), epoch_1000x)
            log_writer.add_scalar('loss_causal', loss_causal.item(), epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None, epoch=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    metric_logger.add_meter('loss_full', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_masked', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_causal', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_sparse', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))

    # switch to evaluation mode
    model.eval()

    total_epochs = getattr(args, 'epochs', 200) if args else 200
    current_epoch = epoch if epoch is not None else total_epochs
    lambda_c, lambda_s = get_annealed_lambdas(current_epoch, total_epochs)
    causal_progress = get_causal_progress(current_epoch)
    causal_active = causal_progress > 0

    for batch in metric_logger.log_every(data_loader, 500, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output_full, output_masked, mask_prob, fr_full, fr_masked, firing_rate = model(
                images, epoch=current_epoch, enable_causal=causal_active
            )

            output_full_fused = fuse_logits_by_firing(output_full, None, eps=1e-6)
            loss_full = criterion(output_full_fused, target)

            loss_masked = torch.tensor(0.0, device=device)
            loss_causal = torch.tensor(0.0, device=device)
            loss_sparse = torch.tensor(0.0, device=device)

            if causal_active:
                output_masked_fused = fuse_logits_by_firing(output_masked, None, eps=1e-6)
                loss_masked = criterion(output_masked_fused, target)

                if lambda_c > 0:
                    T_kd = 3.0
                    log_prob_masked = F.log_softmax(output_masked_fused / T_kd, dim=-1)
                    prob_full = F.softmax(output_full_fused.detach() / T_kd, dim=-1)
                    loss_causal = F.kl_div(log_prob_masked, prob_full, reduction='batchmean') * (T_kd * T_kd)

                if lambda_s > 0:
                    loss_sparse = mask_prob.mean()

            loss_total = loss_full + causal_progress * loss_masked + lambda_c * loss_causal + lambda_s * loss_sparse

        output = output_masked_fused if causal_active else output_full_fused
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        functional.reset_net(model)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss_total.item())
        metric_logger.update(loss_full=loss_full.item())
        metric_logger.update(loss_masked=loss_masked.item())
        metric_logger.update(loss_causal=loss_causal.item())
        metric_logger.update(loss_sparse=loss_sparse.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} L_Tot {losses.global_avg:.4f} L_Caus {l_caus.global_avg:.4f} L_Spar {l_spar.global_avg:.4f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss,
                l_caus=metric_logger.loss_causal, l_spar=metric_logger.loss_sparse))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
