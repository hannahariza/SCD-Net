import datetime
import os
import time
import logging

import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
from torch.cuda import amp
import QK_SCD_Net_model_dvs128, utils
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import dvs128_gesture
from timm.models import create_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import autoaugment
import random
import numpy as np
import torch.nn.functional as F

_seed_ = 2021
random.seed(2021)
root_path = os.path.abspath(__file__)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

# ===========================================================
# 1. 日志与辅助函数
# ===========================================================
_logger = logging.getLogger('train')


def setup_train_logger_with_temp(log_tmp_dir: str):
    os.makedirs(log_tmp_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tmp_path = os.path.join(log_tmp_dir, f"train_{ts}.log")

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger, tmp_path

    fmt = logging.Formatter(
        '%(asctime)s %(levelname)s:%(name)s:%(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(tmp_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger, tmp_path


def move_train_log_to_output_dir(logger: logging.Logger, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    file_handler = None
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            file_handler = h
            break
    if file_handler is None:
        return

    old_path = file_handler.baseFilename
    new_path = os.path.join(output_dir, os.path.basename(old_path))

    if os.path.abspath(old_path) == os.path.abspath(new_path):
        return

    file_handler.close()
    try:
        os.replace(old_path, new_path)
    except FileNotFoundError:
        pass

    file_handler.baseFilename = new_path
    file_handler.stream = open(new_path, file_handler.mode, encoding=file_handler.encoding)
    logger.info(f"Log file moved to: {new_path}")

def get_annealed_lambdas(epoch, max_epochs, warmup_epochs=40, anneal_epochs=45):
    """
    针对 DVS 较短 epoch 调整退火策略
    """
    max_lambda_c = 1.0     #5.0
    max_lambda_s = 0.0        #5e-4

    if epoch is None:
        return max_lambda_c, max_lambda_s

    if epoch < warmup_epochs:
        return 0.0, 0.0
    elif epoch < warmup_epochs + anneal_epochs:
        progress = (epoch - warmup_epochs) / anneal_epochs
        return max_lambda_c * progress, max_lambda_s * progress
    else:
        return max_lambda_c, max_lambda_s


def fuse_logits_by_firing(output, firing_num_t, eps=1e-6):
    """如果模型内部已经mean(0)输出了2维[B, C]，则直接返回"""
    if output.dim() == 2:
        return output
    T, N, C = output.shape
    if firing_num_t is None or firing_num_t.dim() == 1:
        return output.mean(dim=0)
    denom = firing_num_t.sum(dim=0, keepdim=True) + eps
    w = firing_num_t / denom
    return (output * w.unsqueeze(-1)).sum(dim=0)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--model', default='QKFormer', help='model')
    parser.add_argument('--dataset', default='DVS128Gesture', help='dataset')
    parser.add_argument('--num-classes', type=int, default=11, metavar='N',
                        help='number of label classes (default: 11)')
    parser.add_argument('--data-path', default='D:/dataset/data/DVS128Gesture', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=256, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./output/train_causal_dvs128', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")

    # Mixed precision training parameters
    parser.add_argument('--amp', default=True, action='store_true', help='Use AMP training')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', default=True, action='store_true', help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')

    # Optimizer Parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=0.06, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')

    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', default=None, type=int)

    # Learning rate scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N', help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min-lr', type=float, default=2e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (2e-5)')
    parser.add_argument('--epochs', type=int, default=192, metavar='N', help='number of epochs to train (default: 192)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N', help='epoch repeat multiplier')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.5, help='mixup alpha, mixup enabled if > 0. (default: 0.5)')
    parser.add_argument('--cutmix', type=float, default=0., help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio')
    parser.add_argument('--mixup-prob', type=float, default=0.5, help='Probability of performing mixup or cutmix')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Probability of switching to cutmix')
    parser.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params.')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N', help='Turn off mixup after this epoch')
    args = parser.parse_args()
    return args


def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int,
                            random_split: bool = False):
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, T_train=None,
                    aug=None, trival_aug=None, mixup_fn=None, num_epochs=192, start_time_global=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    lambda_c, lambda_s = get_annealed_lambdas(epoch, num_epochs)

    for batch_idx, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        image = image.float()  # [N, T, C, H, W]
        N, T, C, H, W = image.shape

        if aug != None:
            image = torch.stack([(aug(image[i])) for i in range(N)])

        if trival_aug != None:
            image = torch.stack([(trival_aug(image[i])) for i in range(N)])

        #=======================================================
        # 新增：时间维度的泛化增强 (Temporal Jitter & Drop)
         # 仅在训练阶段应用，打破模型对固定时间步序列的死记硬背
        # =======================================================
        # 1. 随机时间平移 (Temporal Roll): 模拟动作提前或延后发生
        if np.random.rand() < 0.5:
            shift = np.random.randint(-2, 3)  # 随机整体平移 -2 到 +2 帧
            image = torch.roll(image, shifts=shift, dims=1)

        # 若平移引入了环形回绕，为防止物理逻辑错乱，将回绕的部分置零
            if shift > 0:
                image[:, :shift, :, :, :] = 0
            elif shift < 0:
                image[:, shift:, :, :, :] = 0

        # 2. 随机时间步丢弃 (Temporal Cutout): 模拟动态视觉传感器的瞬时丢帧现象
        if np.random.rand() < 0.3:  # 30% 概率触发
            drop_t = np.random.randint(0, T)
            image[:, drop_t, :, :, :] = 0
        # =======================================================

        if mixup_fn is not None:
            image, target = mixup_fn(image, target)
            target_for_compu_acc = target.argmax(dim=-1)

        if T_train:
            sec_list = np.random.choice(image.shape[1], T_train, replace=False)
            sec_list.sort()
            image = image[:, sec_list]

        if scaler is not None:
            with amp.autocast():
                # output_full, output_masked, mask_prob, firing_num_t, firing_rate = model(image)
                output_full, output_masked, mask_prob, fr_full, fr_masked, firing_rate = model(image)

                output_full = fuse_logits_by_firing(output_full, None, eps=1e-6)
                output_masked = fuse_logits_by_firing(output_masked, fr_masked, eps=1e-6)

                loss_full = criterion(output_full, target)
                loss_masked = criterion(output_masked, target)

                loss_causal = torch.tensor(0.0, device=image.device)
                loss_sparse = torch.tensor(0.0, device=image.device)

                if lambda_c > 0:
                    # log_prob_masked = F.log_softmax(output_masked, dim=-1)
                    # prob_full = F.softmax(output_full.detach(), dim=-1)
                    # loss_causal = F.kl_div(log_prob_masked, prob_full, reduction='batchmean')
                    T_kd = 3.0
                    log_prob_masked = F.log_softmax(output_masked / T_kd, dim=-1)
                    prob_full = F.softmax(output_full.detach() / T_kd, dim=-1)
                    loss_causal = F.kl_div(log_prob_masked, prob_full, reduction='batchmean') * (T_kd * T_kd)

                if lambda_s > 0:
                    loss_sparse = mask_prob.mean()

                loss = loss_full + loss_masked + lambda_c * loss_causal + lambda_s * loss_sparse
                # output = model(image)
                # loss = criterion(output, target)
        else:
            output_full, output_masked, mask_prob, fr_full, fr_masked, firing_rate = model(image)

            output_full = fuse_logits_by_firing(output_full, None, eps=1e-6)
            output_masked = fuse_logits_by_firing(output_masked, fr_masked, eps=1e-6)

            loss_full = criterion(output_full, target)
            loss_masked = criterion(output_masked, target)
            loss_causal = torch.tensor(0.0, device=image.device)
            loss_sparse = torch.tensor(0.0, device=image.device)

            if lambda_c > 0:
                # log_prob_masked = F.log_softmax(output_masked, dim=-1)
                # prob_full = F.softmax(output_full.detach(), dim=-1)
                # loss_causal = F.kl_div(log_prob_masked, prob_full, reduction='batchmean')
                T_kd = 3.0
                log_prob_masked = F.log_softmax(output_masked / T_kd, dim=-1)
                prob_full = F.softmax(output_full.detach() / T_kd, dim=-1)
                loss_causal = F.kl_div(log_prob_masked, prob_full, reduction='batchmean') * (T_kd * T_kd)
            if lambda_s > 0:
                loss_sparse = mask_prob.mean()
            loss = loss_full + loss_masked + lambda_c * loss_causal + lambda_s * loss_sparse
            # output = model(image)
            # loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        if mixup_fn is not None:
            # acc1, acc5 = utils.accuracy(output, target_for_compu_acc, topk=(1, 5))
            acc1, acc5 = utils.accuracy(output_masked, target_for_compu_acc, topk=(1, 5))
        else:
            # acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            acc1, acc5 = utils.accuracy(output_masked, target, topk=(1, 5))

        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

        # === 自定义终端/文件日志输出 ===
        if (batch_idx + 1) % print_freq == 0 or batch_idx == len(data_loader) - 1:
            et = time.time() - start_time_global
            et_str = str(datetime.timedelta(seconds=et))[:-7]
            fr_val = firing_rate.mean().item() if isinstance(firing_rate, torch.Tensor) else 0.0

            _logger.info(
                f"Train: [{et_str}] [Epoch: {epoch}/{num_epochs}] [Iter: {batch_idx + 1:>4d}/{len(data_loader)}]  "
                f"Loss_Tot: {loss.item():>6.4f}  "
                f"L_Full: {loss_full.item():>6.4f}  "
                f"L_Mask: {loss_masked.item():>6.4f}  "
                f"L_Caus: {loss_causal.item():>6.4f}  "
                f"L_Spar: {loss_sparse.item():>6.4f}  "
                f"FR: {fr_val:.4f}  "
                f"Wgt(C/S): {lambda_c:.2f}/{lambda_s:.2f}  "
                f"LR: {optimizer.param_groups[0]['lr']:.3e}  "
                f"Acc@1: {acc1.item():>7.4f}"
            )

    metric_logger.synchronize_between_processes()

    # 额外返回分量 loss 供 wandb 记录（如果需要可记录更详细信息）
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, \
        loss_full.item(), loss_masked.item(), loss_causal.item(), loss_sparse.item()

def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:', epoch=None, num_epochs=192,
             start_time_global=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    lambda_c, lambda_s = get_annealed_lambdas(epoch if epoch is not None else num_epochs, num_epochs)

    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = image.float()

            # output_full, output_masked, mask_prob, firing_num_t, firing_rate = model(image)
            output_full, output_masked, mask_prob, fr_full, fr_masked, firing_rate = model(image)

            output_full = fuse_logits_by_firing(output_full, None, eps=1e-6)
            output_masked = fuse_logits_by_firing(output_masked, fr_masked, eps=1e-6)

            loss_full = criterion(output_full, target)
            loss_masked = criterion(output_masked, target)
            loss_causal = torch.tensor(0.0, device=image.device)
            loss_sparse = torch.tensor(0.0, device=image.device)

            if lambda_c > 0:
                # log_prob_masked = F.log_softmax(output_masked, dim=-1)
                # prob_full = F.softmax(output_full, dim=-1)
                # loss_causal = F.kl_div(log_prob_masked, prob_full, reduction='batchmean')
                T_kd = 3.0
                log_prob_masked = F.log_softmax(output_masked / T_kd, dim=-1)
                prob_full = F.softmax(output_full.detach() / T_kd, dim=-1)
                loss_causal = F.kl_div(log_prob_masked, prob_full, reduction='batchmean') * (T_kd * T_kd)

            if lambda_s > 0:
                loss_sparse = mask_prob.mean()

            loss = loss_full + loss_masked + lambda_c * loss_causal + lambda_s * loss_sparse

            functional.reset_net(model)

            # 测试使用掩码分支的结果
            acc1, acc5 = utils.accuracy(output_masked, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

            # === 自定义终端/文件日志输出 ===
            if (batch_idx + 1) % print_freq == 0 or batch_idx == len(data_loader) - 1:
                et = time.time() - start_time_global
                et_str = str(datetime.timedelta(seconds=et))[:-7]

                _logger.info(
                    f"Test:  [{et_str}] [Iter: {batch_idx + 1:>4d}/{len(data_loader)}]  "
                    f"Loss_Tot: {loss.item():>6.4f}  "
                    f"L_Caus: {loss_causal.item():>6.4f}  "
                    f"L_Spar: {loss_sparse.item():>6.4f}  "
                    f"Acc@1: {acc1.item():>7.4f}  "
                    f"Acc@5: {acc5.item():>7.4f}"
                )

    metric_logger.synchronize_between_processes()
    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    _logger.info(f' * Acc@1 = {acc1:.4f}, Acc@5 = {acc5:.4f}, loss = {loss:.4f}')
    return loss, acc1, acc5


def load_data(dataset_dir, distributed, T):
    _logger.info("Loading data")
    st = time.time()

    dataset_train = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=True, data_type='frame', frames_number=T,
                                                 split_by='number')
    dataset_test = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=False, data_type='frame', frames_number=T,
                                                split_by='number')

    _logger.info(f"Took {time.time() - st:.2f} seconds")
    _logger.info("Creating data loaders")

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler


def main(args):
    # ==========================
    # 初始化临时 Logger (稍后会移动到专属 logs 文件夹)
    # ==========================
    global _logger
    _logger, tmp_log_path = setup_train_logger_with_temp("./output/tmp_logs")
    _logger.info(f"Logging initialized at temp path: {tmp_log_path}")

    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.

    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    _logger.info(str(args))

    # ==========================
    # 1. 构建主输出文件夹命名 (时间戳 + 参数)
    # ==========================
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_{args.model}_b{args.batch_size}_T{args.T}"

    if args.T_train:
        run_name += f'_Ttrain{args.T_train}'
    if args.weight_decay:
        run_name += f'_wd{args.weight_decay}'
    if args.opt == 'adamw':
        run_name += '_adamw'
    else:
        run_name += '_sgd'
    if args.connect_f:
        run_name += f'_cnf_{args.connect_f}'

    run_name += f'_lr{args.lr}'

    # 设定根目录为指定的 output_dir为 './output/src'
    base_output_dir = "./output/src"
    run_dir = os.path.join(base_output_dir, run_name)

    # ==========================
    # 2. 构建并创建分类子文件夹
    # ==========================
    logs_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "tensorboard")

    if utils.is_main_process():
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)

    # 将日志文件从临时目录移动到正式的 logs_dir 中
    move_train_log_to_output_dir(_logger, logs_dir)
    _logger.info(f"All outputs for this run will be saved to: {run_dir}")

    device = torch.device(args.device)
    data_path = args.data_path

    dataset_train, dataset_test, train_sampler, test_sampler = load_data(data_path, args.distributed, args.T)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True)

    model = create_model(
        args.model,
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    _logger.info("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f"number of params: {n_parameters}")
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion_train = SoftTargetCrossEntropy().cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(args, model)
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    start_epoch = 0
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    start_time_global = time.time()  # 全局绝对时间启动点

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device, header='Test:', epoch=num_epochs,
                 num_epochs=num_epochs, start_time_global=start_time_global)
        return

    # ==========================
    # 3. 初始化 TensorBoard 和保存 args.txt 到指定子文件夹
    # ==========================
    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch

        # TensorBoard 文件存入 tensorboard 子文件夹
        train_tb_writer = SummaryWriter(os.path.join(tb_dir, 'train'), purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(os.path.join(tb_dir, 'te'), purge_step=purge_step_te)

        # 实验参数 txt 存入 logs 子文件夹
        with open(os.path.join(logs_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        _logger.info(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    train_snn_aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
    train_trivalaug = autoaugment.SNNAugmentWide()
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)

    _logger.info("Start training")

    for epoch in range(args.start_epoch, num_epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= 75:
            mixup_fn.mixup_enabled = False

        train_loss, train_acc1, train_acc5, l_full, l_mask, l_caus, l_spar = train_one_epoch(
            model, criterion_train, optimizer, data_loader, device, epoch,
            args.print_freq, scaler, args.T_train,
            train_snn_aug, train_trivalaug, mixup_fn, num_epochs, start_time_global)

        if utils.is_main_process() and train_tb_writer:
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)

        lr_scheduler.step(epoch + 1)

        test_loss, test_acc1, test_acc5 = evaluate(
            model, criterion, data_loader_test, device=device, header='Test:',
            epoch=epoch, num_epochs=num_epochs, start_time_global=start_time_global)

        if te_tb_writer is not None and utils.is_main_process():
            te_tb_writer.add_scalar('test_loss', test_loss, epoch)
            te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
            te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)

        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True

        # ==========================
        # 4. 将检查点保存至 checkpoints 子文件夹
        # ==========================
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
            'max_test_acc1': max_test_acc1,
            'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
        }

        if save_max:
            utils.save_on_master(checkpoint, os.path.join(ckpt_dir, 'checkpoint_max_test_acc1.pth'))

        total_time = time.time() - start_time_global
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        _logger.info(
            f'Training time {total_time_str}, max_test_acc1 {max_test_acc1:.4f}, test_acc5_at_max_test_acc1 {test_acc5_at_max_test_acc1:.4f}')
        _logger.info(f'Current Run Directory: {run_dir}')

    # 保存最后一个 Epoch 的权重
    utils.save_on_master(checkpoint, os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pth'))

    return max_test_acc1


if __name__ == "__main__":
    args = parse_args()
    main(args)
