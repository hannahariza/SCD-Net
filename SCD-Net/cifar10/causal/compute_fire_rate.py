import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.clock_driven import functional

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

SCRIPT_DIR = Path(__file__).resolve().parent
CAUSAL_MODEL_DIR = SCRIPT_DIR
DEFAULT_CAUSAL_CKPT = CAUSAL_MODEL_DIR / 'checkpoint' / 'causal' / 'model_best.pth.tar'

DEFAULT_OUTPUT_DIR = CAUSAL_MODEL_DIR / 'output' / 'firing_rate'

CAUSAL_MODEL_SPECS = {
    'causal': {
        'module_file': 'max_former_causal.py',
        'checkpoint_subdir': 'causal',
    },
    'after_stage3': {
        'module_file': 'max_former_causal_after_stage3.py',
        'checkpoint_subdir': 'after_stage3',
    },
    'after_patch_embed1': {
        'module_file': 'max_former_causal_stage1.py',
        'checkpoint_subdir': 'after_patch_embed1',
    },
    'without_kl': {
        'module_file': 'max_former_causal.py',
        'checkpoint_subdir': 'without_kl',
    },
    'without_sparse': {
        'module_file': 'max_former_causal_after_stage3.py',
        'checkpoint_subdir': 'without_sparse',
    },
}

MODULE_CLEANUP_NAMES = [
    'max_former',
    'max_former_causal',
    'mixer_hub',
    'embedding_hub',
    'ms_qkformer',
    'misc',
    'utils',
]

# -------------------- 加载本地模块 -------------------- #
def load_local_python_module(module_file: Path, unique_name: str):
    import sys
    import importlib.util
    original_sys_path = list(sys.path)
    original_unique_module = sys.modules.get(unique_name)
    for name in MODULE_CLEANUP_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(module_file.parent))
    try:
        spec = importlib.util.spec_from_file_location(unique_name, module_file)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load module {module_file}')
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

# -------------------- Spike 层判断 -------------------- #
def is_spike_layer(module: torch.nn.Module) -> bool:
    return module.__class__.__name__ == 'MultiStepLIFNode'

def get_spike_layer_names(model: torch.nn.Module) -> List[str]:
    names = [name for name, m in model.named_modules() if is_spike_layer(m)]
    if not names:
        raise RuntimeError('No MultiStepLIFNode layers found')
    return names

def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    for n, m in model.named_modules():
        if n == name:
            return m
    raise ValueError(f"Module {name} not found")

# -------------------- 数据加载 -------------------- #
def build_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def create_cifar10_loader(dataset_root: str, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    dataset = datasets.CIFAR10(root=dataset_root, train=False, download=False, transform=build_eval_transform())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

def clear_cuda_memory():
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

# -------------------- 统计函数 -------------------- #
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

    def reset(self):
        self.outputs.clear()

    def get_output(self, idx=-1):
        if not self.outputs:
            return None
        return self.outputs[idx]

def tensor_binary_counts(x: torch.Tensor) -> Tuple[int, int]:
    x = x.float()
    if not torch.all((x == 0) | (x == 1)):
        x = (x > 0).float()
    return int(x.sum().item()), int(x.numel())

def merge_shapes(existing: Optional[Tuple[int, ...]], new: Tuple[int, ...]) -> Tuple[int, ...]:
    if existing is None:
        return new
    merged = list(existing)
    if len(new) >= 2:
        merged[1] += new[1]
    else:
        merged[0] += new[0]
    return tuple(merged)

@torch.no_grad()
def collect_causal_spike_stats(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> Dict:
    spike_layer_names = get_spike_layer_names(model)
    hooks: Dict[str, FeatureHook] = {}
    handles = []
    stats: Dict[str, Dict] = {name: {'num_ones': 0, 'num_values': 0, 'shape': None} for name in spike_layer_names}

    total_ones_all = 0
    total_values_all = 0
    num_images = 0

    try:
        for name in spike_layer_names:
            hook = FeatureHook()
            hooks[name] = hook
            handles.append(get_module_by_name(model, name).register_forward_hook(hook))

        for batch_idx, (images, _) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            for hook in hooks.values():
                hook.reset()
            _ = model(images)
            for name, hook in hooks.items():
                feat = hook.get_output(-1)
                if feat is None:
                    continue
                num_ones, num_values = tensor_binary_counts(feat)
                layer_stat = stats[name]
                layer_stat['num_ones'] += num_ones
                layer_stat['num_values'] += num_values
                layer_stat['shape'] = merge_shapes(layer_stat['shape'], tuple(feat.shape))

                total_ones_all += num_ones
                total_values_all += num_values
            num_images += images.shape[0]
            functional.reset_net(model)
    finally:
        for h in handles:
            h.remove()

    # 每层发放率
    active_stats: Dict[str, Dict] = {}
    inactive_layers: List[str] = []
    for name, layer_stat in stats.items():
        if layer_stat['num_values'] <= 0:
            layer_stat['firing_rate'] = None
            inactive_layers.append(name)
            continue
        layer_stat['firing_rate'] = layer_stat['num_ones'] / layer_stat['num_values']
        active_stats[name] = layer_stat

    # 全网络总发放率
    total_firing_rate = total_ones_all / total_values_all if total_values_all > 0 else 0.0
    return {
        'num_images': num_images,
        'num_spike_layers': len(spike_layer_names),
        'num_active_spike_layers': len(active_stats),
        'inactive_spike_layers': inactive_layers,
        'per_layer': active_stats,
        'total_firing_rate': total_firing_rate
    }

# -------------------- 主函数 -------------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_CAUSAL_CKPT))
    parser.add_argument('--variant', type=str, default='causal', choices=list(CAUSAL_MODEL_SPECS.keys()))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--time_step', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_batches', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--tag', type=str, default='causal')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    max_batches = None if args.max_batches < 0 else args.max_batches
    checkpoint_path = find_checkpoint(args.checkpoint)
    current_batch_size = args.batch_size

    while True:
        model = None
        data_loader = None
        try:
            clear_cuda_memory()
            data_loader = create_cifar10_loader(args.dataset_root, current_batch_size, args.num_workers, pin_memory=(device.type=='cuda'))

            # 加载模型
            model_file = CAUSAL_MODEL_DIR / CAUSAL_MODEL_SPECS[args.variant]['module_file']
            module = load_local_python_module(model_file, f'maxformer_causal_model_{args.variant}')
            if not hasattr(module, 'max_former'):
                raise RuntimeError("max_former factory not found")
            model = module.max_former(T=args.time_step, num_classes=args.num_classes, in_channels=3, embed_dims=384)
            model.to(device)
            model.eval()

            # 加载 checkpoint
            print(f"Loading checkpoint: {checkpoint_path}")
            load_checkpoint_flex(model, checkpoint_path, device)

            # 统计脉冲发放率
            result = collect_causal_spike_stats(model, data_loader, device, max_batches=max_batches)
            result['checkpoint'] = str(checkpoint_path)
            result['variant'] = args.variant

            # 输出文件
            save_path = Path(args.output_dir) / f'firing_rate_{args.tag}_t{args.time_step}.txt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(f"Total firing rate: {result['total_firing_rate']:.6f}\n")
                f.write(f"Number of spike layers: {result['num_spike_layers']}\n")
                f.write(f"Number of active spike layers: {result['num_active_spike_layers']}\n")
                f.write(f"Inactive spike layers: {result['inactive_spike_layers']}\n")
                f.write("Per layer firing rates:\n")
                for name, stats in result['per_layer'].items():
                    f.write(f"{name}: {stats['firing_rate']:.6f}\n")

            # 控制台打印
            print(f"Total firing rate: {result['total_firing_rate']:.6f}")
            print(f"Number of spike layers: {result['num_spike_layers']}")
            print(f"Number of active spike layers: {result['num_active_spike_layers']}")
            if result['inactive_spike_layers']:
                print(f"Inactive spike layers: {result['inactive_spike_layers']}")
            print("First 5 layer firing rates:")
            for i, (name, stats) in enumerate(result['per_layer'].items()):
                if i >= 5:
                    break
                print(f"{name}: {stats['firing_rate']:.6f}")

            break

        except torch.cuda.OutOfMemoryError:
            if current_batch_size <= 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            print(f"CUDA OOM, retry with batch_size={current_batch_size}")
        finally:
            if model is not None:
                del model
            if data_loader is not None:
                del data_loader
            clear_cuda_memory()

if __name__ == "__main__":
    main()
