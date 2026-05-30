import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import model.qkformer_causal_tinyimagenet.util.lr_decay_hst as lrd
import model.qkformer_causal_tinyimagenet.util.misc as misc
from model.qkformer_causal_tinyimagenet.util.datasets import build_dataset
from model.qkformer_causal_tinyimagenet.util.misc import NativeScalerWithGradNormCount as NativeScaler

try:
    import qkformer_fixed as qkformer
except ImportError:
    import model.qkformer_causal_tinyimagenet.qkformer as qkformer
from model.qkformer_causal_tinyimagenet.engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--config', default='', type=str, help='path to yaml config file')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--finetune', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--data_path', default='/media/data/imagenet2012', type=str)

    parser.add_argument('--model', default='QKFormer_10_384', type=str, metavar='MODEL')
    parser.add_argument('--time_step', default=4, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--nb_classes', default=1000, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')

    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=6e-4, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')

    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT')
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)

    parser.add_argument('--mixup', type=float, default=0)
    parser.add_argument('--cutmix', type=float, default=0)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')

    parser.add_argument('--output_dir', default='./output_dir_qkformer')
    parser.add_argument('--log_dir', default='./output_dir_qkformer')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--find_unused_parameters', action='store_true', default=False)
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
    config_args, _ = config_parser.parse_known_args()

    parser = get_args_parser()
    if config_args.config:
        yaml_cfg = load_yaml_config(config_args.config)
        parser.set_defaults(**yaml_cfg)

    return parser.parse_args()


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
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print('Sampler_train = %s' % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
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
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print('Mixup is activated!')
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = build_model_from_args(args)

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print('Load pre-trained checkpoint from: %s' % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f'Removing key {k} from pretrained checkpoint')
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            trunc_normal_(model.head.weight, std=2e-5)
            if model.head.bias is not None:
                nn.init.constant_(model.head.bias, 0)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model = %s' % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print('base lr: %.2e' % (args.lr * 256 / eff_batch_size))
    print('actual lr: %.2e' % args.lr)
    print('accumulate grad iterations: %d' % args.accum_iter)
    print('effective batch size: %d' % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_parameters,
        )
        model_without_ddp = model.module

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print('criterion = %s' % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    max_accuracy = 0.0
    cleanup_old_checkpoints(args.output_dir)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args,
        )

        test_stats = evaluate(data_loader_val, model, device)
        current_accuracy = float(test_stats['acc1'])
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {current_accuracy:.1f}%")

        save_checkpoint_fixed(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            filename='checkpoint-last.pth',
            max_accuracy=max_accuracy,
            current_accuracy=current_accuracy,
        )

        is_best = current_accuracy > max_accuracy
        if is_best:
            max_accuracy = current_accuracy
            save_checkpoint_fixed(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                filename='checkpoint-best.pth',
                max_accuracy=max_accuracy,
                current_accuracy=current_accuracy,
            )

        cleanup_old_checkpoints(args.output_dir)
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
            log_writer.flush()

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = parse_args_with_yaml()
    run_info = prepare_run_dirs(args)
    if run_info is not None:
        print('[Run] output_dir =', args.output_dir)
        print('[Run] log_dir    =', args.log_dir)
        print('[Run] json log   =', run_info['json_log_path'])
    main(args)
