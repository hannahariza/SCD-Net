# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from spikingjelly.clock_driven import functional


def fuse_logits_by_firing(output, firing_num_t=None, eps=1e-6):
    if output.dim() == 2:
        return output
    assert output.dim() == 3, f"Expect output dim=2 or 3, got {output.dim()}"

    if firing_num_t is None or firing_num_t.dim() != 2:
        return output.mean(dim=0)

    denom = firing_num_t.sum(dim=0, keepdim=True) + eps
    weights = firing_num_t / denom
    return (output * weights.unsqueeze(-1)).sum(dim=0)


def causal_intervention_loss(outputs, targets, criterion, args, eval_mode=False):
    if not isinstance(outputs, (tuple, list)):
        loss = criterion(outputs, targets)
        zero = loss.new_zeros(())
        return loss, outputs, {
            'loss_full': loss,
            'loss_intervened': zero,
            'loss_kl': zero,
            'loss_sparse': zero,
        }

    output_full, output_intervened, mask_prob, fr_full, fr_intervened, _ = outputs
    output_full = fuse_logits_by_firing(output_full, fr_full)
    output_intervened = fuse_logits_by_firing(output_intervened, fr_intervened)

    loss_full = criterion(output_full, targets)
    loss_intervened = criterion(output_intervened, targets)

    kd_temp = getattr(args, 'causal_kd_temp', 3.0) if args is not None else 3.0
    log_prob_intervened = F.log_softmax(output_intervened / kd_temp, dim=-1)
    prob_full = F.softmax(output_full.detach() / kd_temp, dim=-1)
    loss_kl = F.kl_div(log_prob_intervened, prob_full, reduction='batchmean') * (kd_temp * kd_temp)
    loss_sparse = mask_prob.mean()

    lambda_intervened = getattr(args, 'lambda_intervened_ce', 1.0) if args is not None else 1.0
    lambda_kl = getattr(args, 'lambda_kl', 1.0) if args is not None else 1.0
    lambda_sparse = getattr(args, 'lambda_sparse', 5e-4) if args is not None else 5e-4

    loss = loss_full + lambda_intervened * loss_intervened + lambda_kl * loss_kl + lambda_sparse * loss_sparse
    logits_for_acc = output_intervened if eval_mode else output_full
    return loss, logits_for_acc, {
        'loss_full': loss_full,
        'loss_intervened': loss_intervened,
        'loss_kl': loss_kl,
        'loss_sparse': loss_sparse,
    }


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, model_ema=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_full', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_intervened', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_kl', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_sparse', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 2000

    accum_iter = args.accum_iter

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
            outputs = model(samples, epoch=epoch, enable_causal=True)
            loss, _, loss_parts = causal_intervention_loss(outputs, targets, criterion, args)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        update_grad = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=update_grad)
        if update_grad:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()
        functional.reset_net(model)
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_full=loss_parts['loss_full'].item())
        metric_logger.update(loss_intervened=loss_parts['loss_intervened'].item())
        metric_logger.update(loss_kl=loss_parts['loss_kl'].item())
        metric_logger.update(loss_sparse=loss_parts['loss_sparse'].item())
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_full', loss_parts['loss_full'].item(), epoch_1000x)
            log_writer.add_scalar('loss_intervened', loss_parts['loss_intervened'].item(), epoch_1000x)
            log_writer.add_scalar('loss_kl', loss_parts['loss_kl'].item(), epoch_1000x)
            log_writer.add_scalar('loss_sparse', loss_parts['loss_sparse'].item(), epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None, epoch=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_full', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_intervened', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_kl', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_sparse', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 500, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(images, epoch=epoch, enable_causal=True)
            loss, output, loss_parts = causal_intervention_loss(outputs, target, criterion, args, eval_mode=True)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        functional.reset_net(model)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_full=loss_parts['loss_full'].item())
        metric_logger.update(loss_intervened=loss_parts['loss_intervened'].item())
        metric_logger.update(loss_kl=loss_parts['loss_kl'].item())
        metric_logger.update(loss_sparse=loss_parts['loss_sparse'].item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} '
          'L_full {loss_full.global_avg:.4f} L_int {loss_intervened.global_avg:.4f} '
          'L_kl {loss_kl.global_avg:.4f} L_sparse {loss_sparse.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss,
                  loss_full=metric_logger.loss_full,
                  loss_intervened=metric_logger.loss_intervened,
                  loss_kl=metric_logger.loss_kl,
                  loss_sparse=metric_logger.loss_sparse))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
