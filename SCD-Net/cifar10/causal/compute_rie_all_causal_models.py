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
BASELINE_MODEL_DIR = SCRIPT_DIR.parent / 'baseline'

CAUSAL_MODEL_DIR = SCRIPT_DIR

DEFAULT_BASELINE_CKPT = BASELINE_MODEL_DIR / 'checkpoint' / 'model_best.pth.tar'

DEFAULT_CAUSAL_CKPT_ROOT = CAUSAL_MODEL_DIR / 'checkpoint'

DEFAULT_OUTPUT_DIR = CAUSAL_MODEL_DIR / 'output' / 'rie'

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
    'max_former_causal_after_stage3',
    'max_former_causal_stage1',
    'mixer_hub',
    'embedding_hub',
    'ms_qkformer',
    'misc',
    'utils',
]


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
        raise RuntimeError('No MultiStepLIFNode layers were found in the model.')
    return names


def get_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    for name, module in model.named_modules():
        if name == module_name:
            return module
    available = [name for name, _ in model.named_modules()]
    preview = ', '.join(available[:50])
    raise ValueError(f"Module '{module_name}' not found. First modules: {preview}")


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


def instantiate_baseline_model(
    time_step: int,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    model_file = BASELINE_MODEL_DIR / 'max_former.py'
    module = load_local_python_module(model_file, 'maxformer_baseline_cifar10_model')
    if not hasattr(module, 'max_former'):
        raise RuntimeError(f"'max_former' factory was not found in {model_file}")
    model = module.max_former(T=time_step, num_classes=num_classes, in_channels=3, embed_dims=384)
    model.to(device)
    model.eval()
    return model


def instantiate_causal_model(
    variant_name: str,
    time_step: int,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    if variant_name not in CAUSAL_MODEL_SPECS:
        raise ValueError(f'Unsupported causal variant: {variant_name}')

    model_file = CAUSAL_MODEL_DIR / CAUSAL_MODEL_SPECS[variant_name]['module_file']
    unique_name = f'maxformer_causal_cifar10_{variant_name}'
    module = load_local_python_module(model_file, unique_name)
    if not hasattr(module, 'max_former'):
        raise RuntimeError(f"'max_former' factory was not found in {model_file}")
    model = module.max_former(T=time_step, num_classes=num_classes, in_channels=3, embed_dims=384)
    model.to(device)
    model.eval()
    return model


def binary_metrics_from_counts(num_ones: int, num_values: int, eps: float = 1e-12) -> Dict[str, float]:
    if num_values <= 0:
        raise ValueError('num_values must be positive when computing binary metrics.')

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


@torch.no_grad()
def tensor_binary_counts(x: torch.Tensor) -> Tuple[int, int]:
    x = x.float()
    if not torch.all((x == 0) | (x == 1)):
        x = (x > 0).float()
    return int(x.sum().item()), int(x.numel())


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
) -> Tuple[Dict[str, Dict[str, object]], int, List[str]]:
    hooks: Dict[str, FeatureHook] = {}
    handles = []
    stats: Dict[str, Dict[str, object]] = {
        name: {'num_ones': 0, 'num_values': 0, 'shape': None}
        for name in layer_names
    }
    num_images = 0

    try:
        for layer_name in layer_names:
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

            num_images += images.shape[0]
            functional.reset_net(model)
    finally:
        for handle in handles:
            handle.remove()

    if num_images == 0:
        raise RuntimeError('No features were collected. Check the CIFAR-10 dataset path and loader settings.')

    active_layer_names = sorted(
        name for name, layer_stat in stats.items()
        if int(layer_stat['num_values']) > 0
    )
    active_stats = {name: stats[name] for name in active_layer_names}
    return active_stats, num_images, active_layer_names


@torch.no_grad()
def compute_rie_all_spike_layers(
    baseline_model: torch.nn.Module,
    causal_model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    baseline_layer_names = get_spike_layer_names(baseline_model)
    causal_layer_names = get_spike_layer_names(causal_model)

    common_layer_names = sorted(set(baseline_layer_names) & set(causal_layer_names))
    if not common_layer_names:
        raise RuntimeError('No common MultiStepLIFNode layer names were found between baseline and causal models.')

    print(f'Found {len(common_layer_names)} common spike layers. Collecting baseline features...')
    baseline_stats, num_images, baseline_active_layer_names = collect_all_spike_stats(
        baseline_model,
        common_layer_names,
        data_loader,
        device,
        max_batches=max_batches,
    )
    print(f'Baseline active spike layers: {len(baseline_active_layer_names)}. Collecting causal features...')
    causal_stats, _, causal_active_layer_names = collect_all_spike_stats(
        causal_model,
        common_layer_names,
        data_loader,
        device,
        max_batches=max_batches,
    )

    active_common_layer_names = sorted(set(baseline_active_layer_names) & set(causal_active_layer_names))
    if not active_common_layer_names:
        raise RuntimeError('No common spike layers were activated during forward passes.')

    layer_results: Dict[str, Dict[str, object]] = {}
    baseline_total_ones = 0
    baseline_total_values = 0
    causal_total_ones = 0
    causal_total_values = 0

    for layer_idx, layer_name in enumerate(active_common_layer_names, start=1):
        baseline_layer_stat = baseline_stats[layer_name]
        causal_layer_stat = causal_stats[layer_name]

        baseline_metrics = binary_metrics_from_counts(
            num_ones=int(baseline_layer_stat['num_ones']),
            num_values=int(baseline_layer_stat['num_values']),
        )
        causal_metrics = binary_metrics_from_counts(
            num_ones=int(causal_layer_stat['num_ones']),
            num_values=int(causal_layer_stat['num_values']),
        )
        rie = causal_metrics['entropy'] / (baseline_metrics['entropy'] + 1e-12)

        if layer_idx == 1 or layer_idx % 10 == 0 or layer_idx == len(active_common_layer_names):
            print(f'Processed layer {layer_idx}/{len(active_common_layer_names)}: {layer_name}')

        layer_results[layer_name] = {
            'baseline_firing_rate': baseline_metrics['firing_rate'],
            'baseline_p0': baseline_metrics['p0'],
            'baseline_p1': baseline_metrics['p1'],
            'baseline_entropy': baseline_metrics['entropy'],
            'baseline_shape': baseline_layer_stat['shape'],
            'causal_firing_rate': causal_metrics['firing_rate'],
            'causal_p0': causal_metrics['p0'],
            'causal_p1': causal_metrics['p1'],
            'causal_entropy': causal_metrics['entropy'],
            'causal_shape': causal_layer_stat['shape'],
            'rie_causal_over_baseline': rie,
            'baseline_num_values': int(baseline_layer_stat['num_values']),
            'baseline_num_ones': int(baseline_layer_stat['num_ones']),
            'causal_num_values': int(causal_layer_stat['num_values']),
            'causal_num_ones': int(causal_layer_stat['num_ones']),
        }

        baseline_total_ones += int(baseline_layer_stat['num_ones'])
        baseline_total_values += int(baseline_layer_stat['num_values'])
        causal_total_ones += int(causal_layer_stat['num_ones'])
        causal_total_values += int(causal_layer_stat['num_values'])

    baseline_all_metrics = binary_metrics_from_counts(baseline_total_ones, baseline_total_values)
    causal_all_metrics = binary_metrics_from_counts(causal_total_ones, causal_total_values)
    overall_rie = causal_all_metrics['entropy'] / (baseline_all_metrics['entropy'] + 1e-12)

    return {
        'num_images': num_images,
        'num_spike_layers': len(active_common_layer_names),
        'spike_layers': active_common_layer_names,
        'overall': {
            'baseline_firing_rate': baseline_all_metrics['firing_rate'],
            'baseline_p0': baseline_all_metrics['p0'],
            'baseline_p1': baseline_all_metrics['p1'],
            'baseline_entropy': baseline_all_metrics['entropy'],
            'causal_firing_rate': causal_all_metrics['firing_rate'],
            'causal_p0': causal_all_metrics['p0'],
            'causal_p1': causal_all_metrics['p1'],
            'causal_entropy': causal_all_metrics['entropy'],
            'rie_causal_over_baseline': overall_rie,
            'baseline_num_values': baseline_total_values,
            'causal_num_values': causal_total_values,
            'baseline_num_ones': baseline_total_ones,
            'causal_num_ones': causal_total_ones,
        },
        'per_layer': layer_results,
    }


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute RIE for all CIFAR-10 causal MaxFormer checkpoints against the CIFAR-10 baseline model.'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default=r'C:\Users\86191\Desktop\ZQL\Project\experiment\datasets\cifar10',
        help='Root directory that contains the CIFAR-10 dataset files used by torchvision.datasets.CIFAR10.',
    )
    parser.add_argument(
        '--baseline_ckpt',
        type=str,
        default=str(DEFAULT_BASELINE_CKPT),
        help='Baseline checkpoint file or checkpoint directory.',
    )
    parser.add_argument(
        '--causal_ckpt_root',
        type=str,
        default=str(DEFAULT_CAUSAL_CKPT_ROOT),
        help='Root directory that contains the causal checkpoint subdirectories.',
    )
    parser.add_argument(
        '--causal_variants',
        type=str,
        nargs='+',
        default=['causal'],
        choices=list(CAUSAL_MODEL_SPECS.keys()),
        help='One or more causal checkpoint variants to evaluate.',
    )
    parser.add_argument('--time_step', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--max_batches',
        type=int,
        default=-1,
        help='Set to -1 to run the full CIFAR-10 test set. Otherwise limit the number of batches.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help='Directory used to save one text result per causal variant.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    max_batches = None if args.max_batches is not None and args.max_batches < 0 else args.max_batches

    baseline_ckpt = find_checkpoint(args.baseline_ckpt)
    print(f'Loading baseline checkpoint: {baseline_ckpt}')

    for variant_name in args.causal_variants:
        print(f'\n===== Evaluating causal variant: {variant_name} =====')
        variant_ckpt_dir = Path(args.causal_ckpt_root) / CAUSAL_MODEL_SPECS[variant_name]['checkpoint_subdir']
        causal_ckpt = find_checkpoint(str(variant_ckpt_dir))
        current_batch_size = args.batch_size

        while True:
            baseline_model = None
            causal_model = None
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

                baseline_model = instantiate_baseline_model(args.time_step, args.num_classes, device)
                causal_model = instantiate_causal_model(variant_name, args.time_step, args.num_classes, device)

                load_checkpoint_flex(baseline_model, baseline_ckpt, device)
                print(f'Loading causal checkpoint: {causal_ckpt}')
                load_checkpoint_flex(causal_model, causal_ckpt, device)

                result = compute_rie_all_spike_layers(
                    baseline_model=baseline_model,
                    causal_model=causal_model,
                    data_loader=data_loader,
                    device=device,
                    max_batches=max_batches,
                )
                result['causal_variant'] = variant_name
                result['baseline_checkpoint'] = baseline_ckpt
                result['causal_checkpoint'] = causal_ckpt
                result['batch_size'] = current_batch_size

                print('\n===== Overall =====')
                for key, value in result['overall'].items():
                    print(f'{key}: {value}')

                save_path = Path(args.output_dir) / f'rie_{variant_name}_t{args.time_step}.txt'
                save_result_text(result, save_path)
                print(f'Saved result to: {save_path}')
                break
            except torch.cuda.OutOfMemoryError:
                if current_batch_size <= 1:
                    raise
                next_batch_size = max(1, current_batch_size // 2)
                print(
                    f'CUDA out of memory while evaluating {variant_name} with batch_size={current_batch_size}. '
                    f'Retrying with batch_size={next_batch_size}.'
                )
                current_batch_size = next_batch_size
            finally:
                del data_loader
                del baseline_model
                del causal_model
                if 'result' in locals():
                    del result
                clear_cuda_memory()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'\nExecution failed: {exc}')
        raise
