import argparse
import importlib
import importlib.util
import math
import random
import sys
from collections import OrderedDict
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent
BASELINE_MODEL_DIR = ROOT_DIR / "model" / "maxformer" / "cifar10" / "baseline"
CAUSAL_MODEL_DIR = SCRIPT_DIR
DEFAULT_BASELINE_CKPT = BASELINE_MODEL_DIR / "checkpoint" / "model_best.pth.tar"
DEFAULT_CAUSAL_CKPT_ROOT = CAUSAL_MODEL_DIR / "checkpoint" / "withput_loss_mask"
DEFAULT_OUTPUT_DIR = CAUSAL_MODEL_DIR / "output" / "rie_energy_aligned"

CAUSAL_MODEL_SPECS = {
    "causal": {
        "module_file": "max_former_causal.py",
        "checkpoint_subdir": "causal",
        "variant": "causal",
    },
    "after_stage3": {
        "module_file": "max_former_causal_after_stage3.py",
        "checkpoint_subdir": "after_stage3",
        "variant": "after_stage3",
    },
    "after_patch_embed1": {
        "module_file": "max_former_causal_stage1.py",
        "checkpoint_subdir": "after_patch_embed1",
        "variant": "after_patch_embed1",
    },
    "without_kl": {
        "module_file": "max_former_causal.py",
        "checkpoint_subdir": "without_kl",
        "variant": "causal",
    },
    "without_sparse": {
        "module_file": "max_former_causal.py",
        "checkpoint_subdir": "without_sparse",
        "variant": "after_stage3",
    },
    "without_loss_mask": {
        "module_file": "max_former_causal.py",
        "checkpoint_subdir": "without_loss_mask",
        "variant": "causal",
    },
    "without_loss_full": {
        "module_file": "max_former_causal.py",
        "checkpoint_subdir": "without_loss_full",
        "variant": "causal",
    },
}

MODULE_CLEANUP_NAMES = [
    "max_former",
    "max_former_causal",
    "max_former_causal_after_stage3",
    "max_former_causal_stage1",
    "mixer_hub",
    "embedding_hub",
    "ms_qkformer",
    "misc",
    "utils",
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
            raise ImportError(f"Unable to load module spec from: {module_file}")
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


def load_variant_module_for_attention(variant: str):
    module_name_map = {
        "causal": "max_former_causal",
        "after_patch_embed1": "max_former_causal_stage1",
        "after_stage3": "max_former_causal_after_stage3",
    }
    if variant not in module_name_map:
        raise ValueError(f"Unsupported variant: {variant}")
    if str(CAUSAL_MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(CAUSAL_MODEL_DIR))
    return importlib.import_module(module_name_map[variant])


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_eval_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def create_cifar10_loader(
    dataset_root: str,
    input_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    max_samples: int,
) -> DataLoader:
    dataset = datasets.CIFAR10(
        root=dataset_root,
        train=False,
        download=False,
        transform=build_eval_transform(input_size),
    )
    if max_samples > 0:
        dataset.data = dataset.data[:max_samples]
        dataset.targets = dataset.targets[:max_samples]
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
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidates = (
        sorted(path.glob("*.pth"))
        + sorted(path.glob("*.pth.tar"))
        + sorted(path.glob("*.pt"))
        + sorted(path.glob("*.bin"))
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under directory: {path}")

    score_keywords = ["best", "model_best", "checkpoint", "last", "final"]
    ranked = sorted(
        candidates,
        key=lambda p: (
            min((i for i, kw in enumerate(score_keywords) if kw in p.name.lower()), default=len(score_keywords)),
            p.name.lower(),
        ),
    )
    return str(ranked[0])


def extract_state_dict(ckpt_obj: dict) -> dict:
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj
    for key in ["state_dict", "model", "model_state_dict"]:
        if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]
    return ckpt_obj


def load_checkpoint_flex(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(ckpt)

    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"Missing keys for {checkpoint_path}: {missing}")
    if unexpected:
        print(f"Unexpected keys for {checkpoint_path}: {unexpected}")


def configure_spiking_backend(model: nn.Module, backend: str = "torch"):
    for module in model.modules():
        if isinstance(module, MultiStepLIFNode):
            module.backend = backend


def instantiate_baseline_model(time_step: int, num_classes: int, device: torch.device) -> torch.nn.Module:
    model_file = BASELINE_MODEL_DIR / "max_former.py"
    module = load_local_python_module(model_file, "energy_aligned_baseline_model")
    model = module.max_former(T=time_step, num_classes=num_classes, in_channels=3, embed_dims=384)
    configure_spiking_backend(model, backend="torch")
    model.to(device)
    model.eval()
    return model


def instantiate_causal_model(variant_name: str, time_step: int, num_classes: int, device: torch.device) -> torch.nn.Module:
    spec = CAUSAL_MODEL_SPECS[variant_name]
    model_file = CAUSAL_MODEL_DIR / spec["module_file"]
    module = load_local_python_module(model_file, f"energy_aligned_causal_model_{variant_name}")
    model = module.max_former(T=time_step, num_classes=num_classes, in_channels=3, embed_dims=384)
    configure_spiking_backend(model, backend="torch")
    model.to(device)
    model.eval()
    return model


def tensor_binary_counts(x: torch.Tensor) -> Tuple[int, int]:
    x = x.detach().float()
    if not torch.all((x == 0) | (x == 1)):
        x = (x > 0).float()
    return int(x.sum().item()), int(x.numel())


def binary_metrics_from_counts(num_ones: int, num_values: int, eps: float = 1e-12) -> Dict[str, float]:
    if num_values <= 0:
        raise ValueError("num_values must be positive when computing binary metrics.")

    p1 = float(num_ones) / float(num_values)
    p0 = 1.0 - p1

    def term(probability: float) -> float:
        if probability < eps:
            return 0.0
        return -probability * math.log2(probability)

    entropy = term(p0) + term(p1)
    return {
        "firing_rate": p1,
        "p0": p0,
        "p1": p1,
        "entropy": entropy,
    }


class InputRateAnalyzer:
    def __init__(self, model: nn.Module, first_layer_name: str, attention_module_types: Tuple[type, ...]):
        self.model = model
        self.first_layer_name = first_layer_name
        self.attention_module_types = attention_module_types
        self.hooks = []
        self.layer_stats = OrderedDict()
        self._patch_original_forwards = {}

        self._register_standard_hooks()
        self._patch_attention_modules()

    def _ensure_layer(self, name: str, layer_type: str):
        if name not in self.layer_stats:
            self.layer_stats[name] = {
                "type": layer_type,
                "num_ones": 0,
                "num_values": 0,
            }

    def _record_input_counts(self, name: str, layer_type: str, x: torch.Tensor, is_first_layer: bool):
        self._ensure_layer(name, layer_type)
        if is_first_layer:
            return
        num_ones, num_values = tensor_binary_counts(x)
        self.layer_stats[name]["num_ones"] += num_ones
        self.layer_stats[name]["num_values"] += num_values

    def _register_standard_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                self.hooks.append(module.register_forward_hook(self._make_op_hook(name)))

    def _make_op_hook(self, name: str):
        def hook(module, inputs, output):
            x = inputs[0]
            if isinstance(module, nn.Conv2d):
                layer_type = "Conv2d"
            elif isinstance(module, nn.Conv1d):
                layer_type = "Conv1d"
            elif isinstance(module, nn.Linear):
                layer_type = "Linear"
            else:
                return
            self._record_input_counts(name, layer_type, x, is_first_layer=(name == self.first_layer_name))

        return hook

    def _patch_attention_modules(self):
        for name, module in self.model.named_modules():
            if isinstance(module, self.attention_module_types):
                self._patch_single_attention_module(name, module)

    def _patch_single_attention_module(self, module_name: str, module: nn.Module):
        original_forward = module.forward
        self._patch_original_forwards[module_name] = original_forward
        analyzer = self

        def patched_forward(this, x):
            t, b, c, h, w = x.shape
            identity = x
            x = this.x_lif(x)
            x = x.flatten(3).contiguous()

            t, b, c, n = x.shape
            x_for_qkv = x.flatten(0, 1).contiguous()

            q_conv_out = this.q_conv(x_for_qkv)
            q_conv_out = this.q_bn(q_conv_out).reshape(t, b, c, n).contiguous()
            q_conv_out = this.q_lif(q_conv_out)
            q = q_conv_out.transpose(-1, -2).reshape(
                t, b, n, this.num_heads, c // this.num_heads
            ).permute(0, 1, 3, 2, 4).contiguous()

            k_conv_out = this.k_conv(x_for_qkv)
            k_conv_out = this.k_bn(k_conv_out).reshape(t, b, c, n).contiguous()
            k_conv_out = this.k_lif(k_conv_out)
            k = k_conv_out.transpose(-1, -2).reshape(
                t, b, n, this.num_heads, c // this.num_heads
            ).permute(0, 1, 3, 2, 4).contiguous()

            v_conv_out = this.v_conv(x_for_qkv)
            v_conv_out = this.v_bn(v_conv_out).reshape(t, b, c, n).contiguous()
            v_conv_out = this.v_lif(v_conv_out)
            v = v_conv_out.transpose(-1, -2).reshape(
                t, b, n, this.num_heads, c // this.num_heads
            ).permute(0, 1, 3, 2, 4).contiguous()

            analyzer._record_input_counts(f"{module_name}.matmul_kv", "AttentionMatMul", k, is_first_layer=False)
            analyzer._record_input_counts(f"{module_name}.matmul_qkv", "AttentionMatMul", q, is_first_layer=False)

            x = k.transpose(-2, -1) @ v
            x = (q @ x) * this.scale
            x = x.transpose(3, 4).reshape(t, b, c, n).contiguous()
            x = this.attn_lif(x)
            x = x.flatten(0, 1)
            x = this.proj_bn(this.proj_conv(x)).reshape(t, b, c, h, w)

            if hasattr(this, "dwc_neuron") and hasattr(this, "dwc") and hasattr(this, "dwc_bn"):
                x = this.dwc_neuron(x).flatten(0, 1).contiguous()
                x = this.dwc(x)
                x = this.dwc_bn(x).reshape(t, b, c, h, w)

            x = x + identity
            return x

        module.forward = MethodType(patched_forward, module)

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        modules = dict(self.model.named_modules())
        for module_name, original_forward in self._patch_original_forwards.items():
            modules[module_name].forward = original_forward
        self._patch_original_forwards.clear()


def forward_causal_masked_stage2(model: nn.Module, images: torch.Tensor):
    x = images.unsqueeze(0).repeat(model.T, 1, 1, 1, 1)

    x = model.patch_embed1(x)
    for blk in model.stage1:
        x = blk(x)

    x = model.patch_embed2(x)
    for blk in model.stage2:
        x = blk(x)

    x_masked, _mask_prob = model.causal_mask(x)

    functional.reset_net(model.patch_embed3)
    functional.reset_net(model.stage3)
    functional.reset_net(model.head_lif)

    x = model.patch_embed3(x_masked)
    for blk in model.stage3:
        x = blk(x)

    feat_masked = x.flatten(3).mean(3)
    feat_masked_lif = model.head_lif(feat_masked)
    out_masked = model.head(feat_masked_lif)
    return out_masked.mean(0)


def forward_causal_masked_stage1(model: nn.Module, images: torch.Tensor):
    x = images.unsqueeze(0).repeat(model.T, 1, 1, 1, 1)

    x_full = model.patch_embed1(x)
    x_masked, _mask_prob = model.causal_mask(x_full)

    functional.reset_net(model.stage1)
    functional.reset_net(model.patch_embed2)
    functional.reset_net(model.stage2)
    functional.reset_net(model.patch_embed3)
    functional.reset_net(model.stage3)
    functional.reset_net(model.head_lif)

    x = x_masked
    for blk in model.stage1:
        x = blk(x)
    x = model.patch_embed2(x)
    for blk in model.stage2:
        x = blk(x)
    x = model.patch_embed3(x)
    for blk in model.stage3:
        x = blk(x)

    feat_masked = x.flatten(3).mean(3)
    feat_masked_lif = model.head_lif(feat_masked)
    out_masked = model.head(feat_masked_lif)
    return out_masked.mean(0)


def forward_causal_masked_after_stage3(model: nn.Module, images: torch.Tensor):
    x = images.unsqueeze(0).repeat(model.T, 1, 1, 1, 1)

    x = model.patch_embed1(x)
    for blk in model.stage1:
        x = blk(x)
    x = model.patch_embed2(x)
    for blk in model.stage2:
        x = blk(x)
    x = model.patch_embed3(x)
    for blk in model.stage3:
        x = blk(x)

    x_masked, _mask_prob = model.causal_mask(x)
    feat_masked = x_masked.flatten(3).mean(3)
    feat_masked_lif = model.head_lif(feat_masked)
    out_masked = model.head(feat_masked_lif)
    return out_masked.mean(0)


def forward_causal_masked_only(model: nn.Module, images: torch.Tensor, variant: str):
    if variant == "after_patch_embed1":
        return forward_causal_masked_stage1(model, images)
    if variant == "after_stage3":
        return forward_causal_masked_after_stage3(model, images)
    return forward_causal_masked_stage2(model, images)


def forward_baseline_time_expanded(model: nn.Module, images: torch.Tensor):
    x = images.unsqueeze(0).repeat(model.T, 1, 1, 1, 1)
    x = model.forward_features(x)
    x = model.head_lif(x)
    x = model.head(x)
    return x.mean(0)


def collect_energy_aligned_stats(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
    run_fn,
    attention_module_types: Tuple[type, ...],
) -> Tuple[Dict[str, Dict[str, object]], int]:
    first_layer_name = next(
        name for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear))
    )
    analyzer = InputRateAnalyzer(model, first_layer_name, attention_module_types)
    num_images = 0

    try:
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(data_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                images = images.to(device, non_blocking=True)
                run_fn(model, images)
                num_images += images.shape[0]
                functional.reset_net(model)
    finally:
        analyzer.remove()

    if num_images == 0:
        raise RuntimeError("No features were collected. Check dataset path and loader settings.")

    active_stats = OrderedDict(
        (name, stat) for name, stat in analyzer.layer_stats.items() if int(stat["num_values"]) > 0
    )
    return active_stats, num_images


def compute_rie_energy_aligned(
    baseline_stats: Dict[str, Dict[str, object]],
    causal_stats: Dict[str, Dict[str, object]],
    num_images: int,
) -> Dict[str, object]:
    common_layer_names = sorted(set(baseline_stats.keys()) & set(causal_stats.keys()))
    if not common_layer_names:
        raise RuntimeError("No common operator input stats were found between baseline and causal models.")

    per_layer = OrderedDict()
    baseline_total_ones = 0
    baseline_total_values = 0
    causal_total_ones = 0
    causal_total_values = 0

    for layer_name in common_layer_names:
        baseline_layer = baseline_stats[layer_name]
        causal_layer = causal_stats[layer_name]

        baseline_metrics = binary_metrics_from_counts(
            int(baseline_layer["num_ones"]),
            int(baseline_layer["num_values"]),
        )
        causal_metrics = binary_metrics_from_counts(
            int(causal_layer["num_ones"]),
            int(causal_layer["num_values"]),
        )

        per_layer[layer_name] = {
            "type": baseline_layer["type"],
            "baseline_firing_rate": baseline_metrics["firing_rate"],
            "baseline_p0": baseline_metrics["p0"],
            "baseline_p1": baseline_metrics["p1"],
            "baseline_entropy": baseline_metrics["entropy"],
            "baseline_num_ones": int(baseline_layer["num_ones"]),
            "baseline_num_values": int(baseline_layer["num_values"]),
            "causal_firing_rate": causal_metrics["firing_rate"],
            "causal_p0": causal_metrics["p0"],
            "causal_p1": causal_metrics["p1"],
            "causal_entropy": causal_metrics["entropy"],
            "causal_num_ones": int(causal_layer["num_ones"]),
            "causal_num_values": int(causal_layer["num_values"]),
            "rie_causal_over_baseline": causal_metrics["entropy"] / (baseline_metrics["entropy"] + 1e-12),
        }

        baseline_total_ones += int(baseline_layer["num_ones"])
        baseline_total_values += int(baseline_layer["num_values"])
        causal_total_ones += int(causal_layer["num_ones"])
        causal_total_values += int(causal_layer["num_values"])

    baseline_overall = binary_metrics_from_counts(baseline_total_ones, baseline_total_values)
    causal_overall = binary_metrics_from_counts(causal_total_ones, causal_total_values)

    return {
        "num_images": num_images,
        "num_common_layers": len(common_layer_names),
        "common_layers": common_layer_names,
        "overall": {
            "baseline_firing_rate": baseline_overall["firing_rate"],
            "baseline_p0": baseline_overall["p0"],
            "baseline_p1": baseline_overall["p1"],
            "baseline_entropy": baseline_overall["entropy"],
            "causal_firing_rate": causal_overall["firing_rate"],
            "causal_p0": causal_overall["p0"],
            "causal_p1": causal_overall["p1"],
            "causal_entropy": causal_overall["entropy"],
            "rie_causal_over_baseline": causal_overall["entropy"] / (baseline_overall["entropy"] + 1e-12),
            "baseline_num_ones": baseline_total_ones,
            "baseline_num_values": baseline_total_values,
            "causal_num_ones": causal_total_ones,
            "causal_num_values": causal_total_values,
        },
        "per_layer": per_layer,
    }


def save_result_text(result: Dict[str, object], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        def write_block(prefix: str, value):
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    write_block(f"{prefix}.{sub_key}" if prefix else sub_key, sub_value)
            elif isinstance(value, list):
                handle.write(f"{prefix}: {value}\n")
            else:
                handle.write(f"{prefix}: {value}\n")

        write_block("", result)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute energy-aligned RIE for CIFAR-10 MaxFormer causal checkpoints."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=r"C:\Users\86191\Desktop\ZQL\Project\experiment\datasets\cifar10",
    )
    parser.add_argument("--baseline_ckpt", type=str, default=str(DEFAULT_BASELINE_CKPT))
    parser.add_argument("--causal_ckpt_root", type=str, default=str(DEFAULT_CAUSAL_CKPT_ROOT))
    parser.add_argument(
        "--causal_variants",
        type=str,
        nargs="+",
        default=["causal"],
        choices=list(CAUSAL_MODEL_SPECS.keys()),
    )
    parser.add_argument("--time_step", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    max_batches = None if args.max_batches is not None and args.max_batches < 0 else args.max_batches

    baseline_ckpt = find_checkpoint(args.baseline_ckpt)
    print(f"Loading baseline checkpoint: {baseline_ckpt}")

    attention_module = load_variant_module_for_attention("causal")
    attention_module_types = tuple(
        t for t in (
            getattr(attention_module, "SSA", None),
            getattr(attention_module, "SSA_dwc", None),
        ) if t is not None
    )

    for variant_name in args.causal_variants:
        print(f"\n===== Evaluating energy-aligned RIE for causal variant: {variant_name} =====")
        causal_ckpt_root_path = Path(args.causal_ckpt_root)
        if causal_ckpt_root_path.is_file():
            causal_ckpt = str(causal_ckpt_root_path)
        else:
            subdir_path = causal_ckpt_root_path / CAUSAL_MODEL_SPECS[variant_name]["checkpoint_subdir"]
            if subdir_path.is_dir():
                causal_ckpt = find_checkpoint(str(subdir_path))
            else:
                causal_ckpt = find_checkpoint(str(causal_ckpt_root_path))
        current_batch_size = args.batch_size

        while True:
            baseline_model = None
            causal_model = None
            data_loader = None
            try:
                clear_cuda_memory()
                print(f"Using batch_size={current_batch_size}")
                data_loader = create_cifar10_loader(
                    dataset_root=args.dataset_root,
                    input_size=args.input_size,
                    batch_size=current_batch_size,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"),
                    max_samples=args.max_samples,
                )

                baseline_model = instantiate_baseline_model(args.time_step, args.num_classes, device)
                causal_model = instantiate_causal_model(variant_name, args.time_step, args.num_classes, device)

                load_checkpoint_flex(baseline_model, baseline_ckpt, device)
                print(f"Loading causal checkpoint: {causal_ckpt}")
                load_checkpoint_flex(causal_model, causal_ckpt, device)

                baseline_stats, num_images = collect_energy_aligned_stats(
                    baseline_model,
                    data_loader,
                    device,
                    max_batches=max_batches,
                    run_fn=lambda model, images: forward_baseline_time_expanded(model, images),
                    attention_module_types=attention_module_types,
                )
                causal_stats, _ = collect_energy_aligned_stats(
                    causal_model,
                    data_loader,
                    device,
                    max_batches=max_batches,
                    run_fn=lambda model, images: forward_causal_masked_only(
                        model, images, CAUSAL_MODEL_SPECS[variant_name]["variant"]
                    ),
                    attention_module_types=attention_module_types,
                )

                result = compute_rie_energy_aligned(baseline_stats, causal_stats, num_images)
                result["causal_variant"] = variant_name
                result["baseline_checkpoint"] = baseline_ckpt
                result["causal_checkpoint"] = causal_ckpt
                result["batch_size"] = current_batch_size
                result["input_size"] = args.input_size
                result["max_samples"] = args.max_samples
                result["max_batches"] = args.max_batches

                print("\n===== Overall =====")
                for key, value in result["overall"].items():
                    print(f"{key}: {value}")

                save_path = Path(args.output_dir) / f"rie_energy_aligned_{variant_name}_t{args.time_step}.txt"
                save_result_text(result, save_path)
                print(f"Saved result to: {save_path}")
                break
            except torch.cuda.OutOfMemoryError:
                if current_batch_size <= 1:
                    raise
                next_batch_size = max(1, current_batch_size // 2)
                print(
                    f"CUDA out of memory while evaluating {variant_name} with batch_size={current_batch_size}. "
                    f"Retrying with batch_size={next_batch_size}."
                )
                current_batch_size = next_batch_size
            finally:
                del data_loader
                del baseline_model
                del causal_model
                if "result" in locals():
                    del result
                clear_cuda_memory()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nExecution failed: {exc}")
        raise
