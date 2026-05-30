import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spikingjelly.clock_driven import functional


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

SCRIPT_DIR = Path(__file__).resolve().parent
CIFAR10_BASELINE_MODEL_DIR = SCRIPT_DIR.parent / 'baseline'

DEFAULT_REFERENCE_CKPT = CIFAR10_BASELINE_MODEL_DIR / 'checkpoint' / 'model_best.pth.tar'

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'output' / 'baseline_firing_rate' 

MODULE_CLEANUP_NAMES = [
    'max_former',
    'mixer_hub',
    'embedding_hub',
    'ms_qkformer',
    'misc',
    'utils',
]

# ------------------------ 加载本地模块 ------------------------ #
def load_local_python_module(module_file: Path, unique_name: str):
    module_dir = str(module_file.parent)
    original_sys_path = list(sys.path)
    original_unique_module = sys.modules.get(unique_name)
    for name in MODULE_CLEANUP_NAMES:
        sys.modules.pop(name, None)

    sys.path.insert(0, module_dir)
    try:
        spec = importlib.util.spec_from_file_location(unique_name, module_file)
        if spec is None or spec.loader is None:
            raise ImportError(f'Unable to load module spec from: {module_file}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if original_unique_module is not None:
            sys.modules[unique_name] = original_unique_module
        else:
            sys.modules.pop(unique_name, None)
        sys.path[:] = original_sys_path

# ------------------------ Hook 与统计 ------------------------ #
class FeatureHook:
    def __init__(self):
        self.outputs: List[torch.Tensor] = []

    def __call__(self, module, inputs, output):
        if torch.is_tensor(output):
            self.outputs.append(output.detach())
        elif isinstance(output, (tuple, list)):
            tensors = [item.detach() for item in output if torch.is_tensor(item)]
            if tensors:
                self.outputs.append(tensors[-1])

    def reset(self) -> None:
        self.outputs.clear()

    def get_output(self, output_index: int = -1) -> Optional[torch.Tensor]:
        if not self.outputs:
            return None
        return self.outputs[output_index]

def is_spike_layer(module: torch.nn.Module) -> bool:
    return module.__class__.__name__ == 'MultiStepLIFNode'

def get_spike_layer_names(model: torch.nn.Module) -> List[str]:
    names: List[str] = []
    for name, module in model.named_modules():
        if name and is_spike_layer(module):
            names.append(name)
    if not names:
        raise RuntimeError('No MultiStepLIFNode layers found in the model.')
    return names

def get_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    for name, module in model.named_modules():
        if name == module_name:
            return module
    available = [name for name, _ in model.named_modules()]
    preview = ', '.join(available[:50])
    raise ValueError(f"Module '{module_name}' not found. First modules: {preview}")

# ------------------------ 数据 ------------------------ #
def build_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def create_cifar10_loader(
    dataset_root: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = datasets.CIFAR10(
        root=dataset_root,
        train=False,
        download=False,
        transform=build_eval_transform(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

def clear_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def find_checkpoint(path_str: str) -> str:
    path = Path(path_str)
    if path.is_file():
        return str(path)
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint path does not exist: {path}')

    candidates = (
        sorted(path.glob('*.pth'))
        + sorted(path.glob('*.pth.tar'))
        + sorted(path.glob('*.pt'))
        + sorted(path.glob('*.bin'))
    )
    if not candidates:
        raise FileNotFoundError(f'No checkpoint file found under directory: {path}')

    score_keywords = ['best', 'model_best', 'checkpoint', 'last', 'final']
    ranked = sorted(
        candidates,
        key=lambda p: (
            min((i for i, kw in enumerate(score_keywords) if kw in p.name.lower()), default=len(score_keywords)),
            p.name.lower(),
        )
    )
    return str(ranked[0])

def extract_state_dict(ckpt_obj: dict) -> dict:
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj
    for key in ['state_dict', 'model', 'model_state_dict']:
        if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]
    return ckpt_obj

def load_checkpoint_flex(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(ckpt)

    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith('module.') else key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f'Missing keys for {checkpoint_path}: {missing}')
    if unexpected:
        print(f'Unexpected keys for {checkpoint_path}: {unexpected}')

# ------------------------ 模型 ------------------------ #
def instantiate_cifar10_baseline_model(
    time_step: int,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    model_file = CIFAR10_BASELINE_MODEL_DIR / 'max_former.py'
    module = load_local_python_module(model_file, 'maxformer_baseline_cifar10_model')
    if not hasattr(module, 'max_former'):
        raise RuntimeError(f"'max_former' factory was not found in {model_file}")
    model = module.max_former(T=time_step, num_classes=num_classes, in_channels=3, embed_dims=384)
    model.to(device)
    model.eval()
    return model

# ------------------------ 发放率统计函数 ------------------------ #
def tensor_binary_counts(x: torch.Tensor) -> Tuple[int, int]:
    x = x.float()
    if not torch.all((x == 0) | (x == 1)):
        x = (x > 0).float()
    return int(x.sum().item()), int(x.numel())

def binary_metrics_from_counts(num_ones: int, num_values: int, eps: float = 1e-12) -> Dict[str, float]:
    p1 = float(num_ones) / float(num_values)
    p0 = 1.0 - p1
    def term(probability: float) -> float:
        if probability < eps:
            return 0.0
        return -probability * math.log2(probability)
    entropy = term(p0) + term(p1)
    return {
        'firing_rate': p1,
        'p0': p0,
        'p1': p1,
        'entropy': entropy,
    }

def merge_batch_shapes(existing_shape: Optional[Tuple[int, ...]], new_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if existing_shape is None:
        return new_shape
    if len(existing_shape) != len(new_shape):
        raise ValueError(f'Inconsistent tensor rank during accumulation: {existing_shape} vs {new_shape}')
    merged = list(existing_shape)
    if len(new_shape) >= 2:
        merged[1] += new_shape[1]
    else:
        merged[0] += new_shape[0]
    return tuple(merged)

@torch.no_grad()
def collect_all_spike_stats(
    model: torch.nn.Module,
    layer_names: List[str],
    data_loader: Iterable,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, object]], int, List[str], float]:
    """
    返回：
    - per_layer_stats: 每层统计信息
    - num_images: 样本数量
    - active_layer_names: 被激活的 spike 层
    - total_firing_rate: 整体发放率
    """
    spike_layer_names = [name for name in layer_names if is_spike_layer(get_module_by_name(model, name))]
    if not spike_layer_names:
        raise RuntimeError('No MultiStepLIFNode layers found for spike stats.')

    hooks: Dict[str, FeatureHook] = {}
    handles = []
    stats: Dict[str, Dict[str, object]] = {
        name: {'num_ones': 0, 'num_values': 0, 'shape': None} for name in spike_layer_names
    }

    total_ones_all_layers = 0
    total_values_all_layers = 0
    num_images = 0

    try:
        for layer_name in spike_layer_names:
            hook = FeatureHook()
            hooks[layer_name] = hook
            handles.append(get_module_by_name(model, layer_name).register_forward_hook(hook))

        for batch_idx, (images, _) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            for hook in hooks.values():
                hook.reset()
            _ = model(images)
            for layer_name, hook in hooks.items():
                feat = hook.get_output(-1)
                if feat is None:
                    continue
                num_ones, num_values = tensor_binary_counts(feat)
                layer_stat = stats[layer_name]
                layer_stat['num_ones'] += num_ones
                layer_stat['num_values'] += num_values
                layer_stat['shape'] = merge_batch_shapes(layer_stat['shape'], tuple(feat.shape))

                # 累加总发放数
                total_ones_all_layers += num_ones
                total_values_all_layers += num_values

            num_images += images.shape[0]
            functional.reset_net(model)
    finally:
        for handle in handles:
            handle.remove()

    active_layer_names = sorted(
        name for name, layer_stat in stats.items() if int(layer_stat['num_values']) > 0
    )

    # 计算总发放率
    total_firing_rate = float(total_ones_all_layers) / float(total_values_all_layers) if total_values_all_layers > 0 else 0.0

    return stats, num_images, active_layer_names, total_firing_rate

# ------------------------ 保存结果 ------------------------ #
def save_result_text(result: Dict[str, object], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as handle:
        def write_block(prefix: str, value):
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    write_block(f'{prefix}.{sub_key}' if prefix else sub_key, sub_value)
            elif isinstance(value, list):
                handle.write(f'{prefix}: {value}\n')
            else:
                handle.write(f'{prefix}: {value}\n')
        write_block('', result)

# ------------------------ 主函数 ------------------------ #
def parse_args():
    parser = argparse.ArgumentParser(description='Compute baseline firing rate for CIFAR-10 MaxFormer.')
    parser.add_argument('--dataset_root', type=str,
                        default=r'C:\Users\86191\Desktop\ZQL\Project\experiment\datasets\cifar10')
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_REFERENCE_CKPT))
    parser.add_argument('--time_step', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_batches', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--tag', type=str, default='baseline')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    max_batches = None if args.max_batches is not None and args.max_batches < 0 else args.max_batches

    checkpoint_path = find_checkpoint(args.checkpoint)
    current_batch_size = args.batch_size

    while True:
        reference_model = None
        data_loader = None
        try:
            clear_cuda_memory()
            print(f'Using batch_size={current_batch_size}')
            data_loader = create_cifar10_loader(
                dataset_root=args.dataset_root,
                batch_size=current_batch_size,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda'),
            )

            reference_model = instantiate_cifar10_baseline_model(args.time_step, args.num_classes, device)
            print(f'Loading checkpoint: {checkpoint_path}')
            load_checkpoint_flex(reference_model, checkpoint_path, device)

            spike_layer_names = get_spike_layer_names(reference_model)
            per_layer_stats, num_images, active_layers, total_firing_rate = collect_all_spike_stats(
                reference_model, spike_layer_names, data_loader, device, max_batches=max_batches
            )

            result_summary = {
                'num_images': num_images,
                'num_spike_layers': len(active_layers),
                'spike_layers': active_layers,
                'per_layer': per_layer_stats,
                'total_firing_rate': total_firing_rate,
                'total_ones_all_layers': total_ones_all_layers,
                'total_values_all_layers': total_values_all_layers
            }

            save_path = Path(args.output_dir) / f'firing_rate_{args.tag}_t{args.time_step}1.txt'
            save_result_text(result_summary, save_path)

            # 计算全网络总脉冲数和总神经元数
            total_ones_all_layers = sum(stat['num_ones'] for stat in per_layer_stats.values())
            total_values_all_layers = sum(stat['num_values'] for stat in per_layer_stats.values())

            print(f'Total spikes (num_ones): {total_ones_all_layers}')
            print(f'Total spike neurons (num_values): {total_values_all_layers}')
            print(f'Total firing rate: {total_firing_rate:.6f}')
            print(f'Saved baseline firing rate result to: {save_path}')
            break

        except torch.cuda.OutOfMemoryError:
            if current_batch_size <= 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            print(f'CUDA OOM. Retry with batch_size={current_batch_size}')
        finally:
            del data_loader
            del reference_model
            if 'per_layer_stats' in locals():
                del per_layer_stats
            clear_cuda_memory()

if __name__ == '__main__':
    main()