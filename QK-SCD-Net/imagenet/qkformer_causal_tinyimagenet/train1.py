import argparse
import datetime
import json
import numpy as np
import sys
import time
from pathlib import Path

import yaml
import os
import torch.nn as nn

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
import timm.optim.optim_factory as optim_factory
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import model.qkformer_causal_tinyimagenet.util.lr_decay_hst as lrd
import model.qkformer_causal_tinyimagenet.util.misc as misc
from model.qkformer_causal_tinyimagenet.util.datasets import build_dataset
from model.qkformer_causal_tinyimagenet.util.misc import NativeScalerWithGradNormCount as NativeScaler
from timm.utils import CheckpointSaver

import model.qkformer_causal_tinyimagenet.qkformer as qkformer

from model.qkformer_causal_tinyimagenet.engine_finetune import train_one_epoch, evaluate

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

def enable_console_log(output_dir, filename="console.log"):
    log_path = Path(output_dir) / filename
    log_file = open(log_path, mode="a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    return log_file

def get_args_parser():
    # important params
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--in_channels', default=3, type=int,
                    help='number of input image channels')
    parser.add_argument('--config', default='', type=str, help='path to yaml config file')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--data_path', default='/root/lanyun-tmp/data/tiny_imagenet_qkformer/tiny-imagenet-200', type=str,
                        help='dataset path')

    # Model parameters
    parser.add_argument('--model', default='QKFormer_10_384', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--time_step', default=4, type=int,
                        help='images input size')
    parser.add_argument('--input_size', default=64, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=6e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=1.0,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params

    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters

    parser.add_argument('--nb_classes', default=200, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def load_yaml_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f'YAML config must be a dict, but got {type(cfg)}')
    if 'config' in cfg and isinstance(cfg['config'], dict):
        cfg = cfg['config']
    elif 'train' in cfg and isinstance(cfg['train'], dict):
        cfg = cfg['train']
    return cfg

def parse_args_with_yaml():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', default='', type=str)
    config_args, remaining = config_parser.parse_known_args()

    parser = get_args_parser()
    if config_args.config:
        yaml_cfg = load_yaml_config(config_args.config)
        parser.set_defaults(**yaml_cfg)

    return parser.parse_args(remaining)

def rebuild_classifier_head_if_needed(model, nb_classes: int):
    if not hasattr(model, 'head'):
        return model
    if isinstance(model.head, nn.Linear) and model.head.out_features != nb_classes:
        in_features = model.head.in_features
        out_features = model.head.out_features
        print(f'[Info] Replace classifier head: {out_features} -> {nb_classes}')
        model.head = nn.Linear(in_features, nb_classes)
        trunc_normal_(model.head.weight, std=.02)
        if model.head.bias is not None:
            nn.init.constant_(model.head.bias, 0)
    return model

def build_model_from_args(args):
    if args.model not in qkformer.__dict__:
        candidates = sorted([k for k, v in qkformer.__dict__.items() if callable(v) and k.startswith('QKFormer_')])
        raise ValueError(f'Unknown model: {args.model}. Available models: {candidates}')

    model = qkformer.__dict__[args.model](
        T=args.time_step,
        img_size=args.input_size,
        num_classes=args.nb_classes,
        in_channels=args.in_channels,
        drop_path_rate=args.drop_path,
    )
    model = rebuild_classifier_head_if_needed(model, args.nb_classes)
    return model

def prepare_run_dirs(args):
    if args.eval:
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        if args.log_dir:
            Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        return None

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = f'{args.model}_inp{args.input_size}_T{args.time_step}_{timestamp}'

    base_output_dir = args.output_dir
    base_log_dir = args.log_dir if args.log_dir else args.output_dir

    args.output_dir = os.path.join(base_output_dir, run_name)
    args.log_dir = os.path.join(base_log_dir, run_name)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    run_info = {
        'run_name': run_name,
        'timestamp': timestamp,
        'base_output_dir': base_output_dir,
        'base_log_dir': base_log_dir,
        'json_log_path': os.path.join(args.output_dir, 'log.txt'),
    }

    with open(os.path.join(args.output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    return run_info

def save_checkpoint_fixed(args, model_without_ddp, optimizer, loss_scaler, epoch, filename, max_accuracy=None, current_accuracy=None):
    if not args.output_dir:
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / filename
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
        'args': args,
    }
    if max_accuracy is not None:
        to_save['max_accuracy'] = float(max_accuracy)
    if current_accuracy is not None:
        to_save['current_accuracy'] = float(current_accuracy)
    misc.save_on_master(to_save, checkpoint_path)


def cleanup_old_checkpoints(output_dir):
    if not output_dir or not misc.is_main_process():
        return
    output_dir = Path(output_dir)
    keep_names = {'checkpoint-best.pth', 'checkpoint-last.pth'}
    for ckpt_path in output_dir.glob('checkpoint*.pth'):
        if ckpt_path.name not in keep_names:
            try:
                ckpt_path.unlink()
            except FileNotFoundError:
                pass

    
def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # model = qkformer.__dict__[args.model](T=args.time_step)
    print(f'[Config] data_path={args.data_path}')
    print(f'[Config] model={args.model}')
    print(f'[Config] input_size={args.input_size}')
    print(f'[Config] time_step={args.time_step}')
    print(f'[Config] nb_classes={args.nb_classes}')
    print(f'[Config] batch_size={args.batch_size}')
    print(f'[Config] drop_path={args.drop_path}')
    print(f'[Config] in_channels={args.in_channels}')

    model = build_model_from_args(args)
    
    print(f'[Model] head_out_features={model.head.out_features}')

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        # interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        # no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    saver = None
    if misc.is_main_process() and args.output_dir:
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, amp_scaler=loss_scaler,
            checkpoint_dir=args.output_dir, recovery_dir=args.output_dir,
            decreasing=False, max_history=1)  # decreasing=False 表示 acc1 越大越好

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    max_accuracy = 0.0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ====== Starting Epoch {epoch} ======")

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        
        test_stats = evaluate(data_loader_val, model, device, args=args, epoch=epoch)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        if test_stats["acc1"] > max_accuracy:
            max_accuracy = test_stats["acc1"]
            best_epoch = epoch

            # 按照 Maxformer 逻辑，使用 timm 的 CheckpointSaver 自动管理（每个 epoch 都会执行）
        if saver is not None:
            saver.save_checkpoint(epoch, metric=test_stats["acc1"])

        epoch_duration = time.time() - epoch_start_time
        print(f'Epoch {epoch} finished in {datetime.timedelta(seconds=int(epoch_duration))}')
        print(f'==> Currently Top-1 Acc: {max_accuracy:.2f}% (Achieved at Epoch {best_epoch})')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\nTraining complete! Total time {total_time_str}')
    print(f'Final Best Acc@1: {max_accuracy:.2f}% at epoch {best_epoch}')

if __name__ == '__main__':
    args = parse_args_with_yaml()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        enable_console_log(args.output_dir)
    main(args)

# if __name__ == '__main__':
#     args = get_args_parser()
#     args = args.parse_args()
#     if args.output_dir:
#         Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#         # 初始化控制台日志记录
#         enable_console_log(args.output_dir)
#     main(args)
