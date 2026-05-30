import argparse
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from spikingjelly.clock_driven import functional
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
DEFAULT_CKPT = THIS_DIR / "checkpoint" / "10-512-t4" / "model_best.pth.tar"
DEFAULT_IMAGE = PROJECT_ROOT / "datasets" / "imagenet1k" / "1.png"
DEFAULT_OUTPUT = THIS_DIR / "output" / "causal_heatmaps_10_512_t4"


def register_legacy_package_alias():
    """Make local nested sources importable as model.maxformer_causal_imagenet100."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    package_name = "model.maxformer_causal_imagenet100"
    package = types.ModuleType(package_name)
    package.__path__ = [str(THIS_DIR)]
    sys.modules[package_name] = package


def load_state_dict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return missing, unexpected


def set_lif_backend(model, backend):
    changed = 0
    for module in model.modules():
        if hasattr(module, "backend"):
            module.backend = backend
            changed += 1
    return changed


def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert("RGB")
    crop_pct = 224 / 256 if input_size <= 224 else 1.0
    resize_size = int(input_size / crop_pct)

    resample = Image.Resampling.BICUBIC
    width, height = image.size
    if width < height:
        new_width = resize_size
        new_height = int(height * resize_size / width)
    else:
        new_height = resize_size
        new_width = int(width * resize_size / height)
    resized = image.resize((new_width, new_height), resample)

    left = (new_width - input_size) // 2
    top = (new_height - input_size) // 2
    cropped = resized.crop((left, top, left + input_size, top + input_size))

    array = np.asarray(cropped).astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_DEFAULT_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_DEFAULT_STD, dtype=np.float32)
    array = (array - mean) / std
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, cropped


@torch.no_grad()
def forward_pre_head_maps(model, image_tensor):
    x = image_tensor.unsqueeze(0).repeat(model.T, 1, 1, 1, 1)

    x = model.patch_embed1(x)
    for block in model.stage1:
        x = block(x)

    x = model.patch_embed2(x)
    for block in model.stage2:
        x = block(x)

    x_full = model.patch_embed3(x)
    for block in model.stage3:
        x_full = block(x_full)

    x_masked, mask_prob = model.causal_mask(x)
    functional.reset_net(model.patch_embed3)
    functional.reset_net(model.stage3)

    x_causal = model.patch_embed3(x_masked)
    for block in model.stage3:
        x_causal = block(x_causal)

    feat_full = x_full.flatten(3).mean(3)
    feat_causal = x_causal.flatten(3).mean(3)
    out_full = model.head(model.head_lif(feat_full))
    functional.reset_net(model.head_lif)
    out_causal = model.head(model.head_lif(feat_causal))
    functional.reset_net(model.head_lif)

    return x_full, x_causal, mask_prob, out_full.mean(0), out_causal.mean(0)


def aggregate_channels(feature_map, mode):
    if mode == "mean":
        heat = feature_map.mean(dim=2)
    elif mode == "sum":
        heat = feature_map.sum(dim=2)
    elif mode == "max":
        heat = feature_map.max(dim=2).values
    elif mode == "mean_abs":
        heat = feature_map.abs().mean(dim=2)
    else:
        raise ValueError(f"Unsupported aggregation mode: {mode}")
    return heat[:, 0].detach().float().cpu().numpy()


def normalize_heatmap(heatmap):
    heatmap = heatmap.astype(np.float32)
    heatmap = heatmap - float(heatmap.min())
    denom = float(heatmap.max())
    if denom > 1e-12:
        heatmap = heatmap / denom
    return heatmap


def colorize_heatmap(heatmap):
    h = normalize_heatmap(heatmap)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def draw_feature_grid(image, grid_size, color=(255, 255, 255), width=1):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    for idx in range(1, grid_size):
        x = round(idx * image_width / grid_size)
        y = round(idx * image_height / grid_size)
        draw.line((x, 0, x, image_height), fill=color, width=width)
        draw.line((0, y, image_width, y), fill=color, width=width)

    draw.rectangle((0, 0, image_width - 1, image_height - 1), outline=color, width=width)
    return image


def save_heatmaps(branch_name, heatmaps, base_image, output_dir, overlay_alpha, grid_size):
    branch_dir = output_dir / branch_name
    branch_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    grid_panels = []
    for time_idx, heatmap in enumerate(heatmaps):
        colored = colorize_heatmap(heatmap).resize(base_image.size, Image.Resampling.BICUBIC)
        overlay = Image.blend(base_image, colored, overlay_alpha)

        heat_path = branch_dir / f"{branch_name}_t{time_idx}.png"
        overlay_path = branch_dir / f"{branch_name}_overlay_t{time_idx}.png"
        colored.save(heat_path)
        overlay.save(overlay_path)
        draw_feature_grid(colored, grid_size).save(branch_dir / f"{branch_name}_t{time_idx}_grid.png")
        draw_feature_grid(overlay, grid_size).save(branch_dir / f"{branch_name}_overlay_t{time_idx}_grid.png")

        panel = overlay.copy()
        grid_panel = draw_feature_grid(overlay, grid_size)
        for item in (panel, grid_panel):
            draw = ImageDraw.Draw(item)
            draw.rectangle((0, 0, 58, 24), fill=(0, 0, 0))
            draw.text((8, 5), f"t={time_idx}", fill=(255, 255, 255))
        panels.append(panel)
        grid_panels.append(grid_panel)

    sheet = Image.new("RGB", (base_image.width * len(panels), base_image.height))
    for idx, panel in enumerate(panels):
        sheet.paste(panel, (idx * base_image.width, 0))
    sheet.save(output_dir / f"{branch_name}_timesteps.png")

    grid_sheet = Image.new("RGB", (base_image.width * len(grid_panels), base_image.height))
    for idx, panel in enumerate(grid_panels):
        grid_sheet.paste(panel, (idx * base_image.width, 0))
    grid_sheet.save(output_dir / f"{branch_name}_timesteps_grid.png")


def save_comparison(full_heatmaps, causal_heatmaps, base_image, output_dir, overlay_alpha, grid_size):
    rows = []
    grid_rows = []
    for branch_heatmaps, label in [(full_heatmaps, "full"), (causal_heatmaps, "causal")]:
        panels = []
        grid_panels = []
        for time_idx, heatmap in enumerate(branch_heatmaps):
            colored = colorize_heatmap(heatmap).resize(base_image.size, Image.Resampling.BICUBIC)
            panel = Image.blend(base_image, colored, overlay_alpha)
            grid_panel = draw_feature_grid(panel, grid_size)
            for item in (panel, grid_panel):
                draw = ImageDraw.Draw(item)
                draw.rectangle((0, 0, 116, 24), fill=(0, 0, 0))
                draw.text((8, 5), f"{label} t={time_idx}", fill=(255, 255, 255))
            panels.append(panel)
            grid_panels.append(grid_panel)

        row = Image.new("RGB", (base_image.width * len(panels), base_image.height))
        for idx, panel in enumerate(panels):
            row.paste(panel, (idx * base_image.width, 0))
        rows.append(row)

        grid_row = Image.new("RGB", (base_image.width * len(grid_panels), base_image.height))
        for idx, panel in enumerate(grid_panels):
            grid_row.paste(panel, (idx * base_image.width, 0))
        grid_rows.append(grid_row)

    comparison = Image.new("RGB", (rows[0].width, rows[0].height * 2))
    comparison.paste(rows[0], (0, 0))
    comparison.paste(rows[1], (0, rows[0].height))
    comparison.save(output_dir / "full_vs_causal_timesteps.png")

    grid_comparison = Image.new("RGB", (grid_rows[0].width, grid_rows[0].height * 2))
    grid_comparison.paste(grid_rows[0], (0, 0))
    grid_comparison.paste(grid_rows[1], (0, grid_rows[0].height))
    grid_comparison.save(output_dir / "full_vs_causal_timesteps_grid.png")


def main():
    parser = argparse.ArgumentParser(description="Export pre-head causal/non-causal MaxFormer heatmaps.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="maxformer_10_512")
    parser.add_argument("--time-step", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--aggregation", choices=["mean_abs", "mean", "sum", "max"], default="mean_abs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lif-backend", choices=["torch", "cupy"], default="torch")
    parser.add_argument("--grid-size", type=int, default=14)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    args = parser.parse_args()

    register_legacy_package_alias()
    importlib.import_module("model.maxformer_causal_imagenet100.max_former")
    from timm.models import create_model

    device = torch.device(args.device)
    model = create_model(args.model, T=args.time_step, num_classes=args.num_classes)
    missing, unexpected = load_state_dict(model, args.checkpoint)
    changed_backend = set_lif_backend(model, args.lif_backend)
    model.to(device).eval()

    image_tensor, base_image = preprocess_image(args.image, args.input_size)
    image_tensor = image_tensor.to(device)

    x_full, x_causal, mask_prob, logits_full, logits_causal = forward_pre_head_maps(model, image_tensor)
    full_heatmaps = aggregate_channels(x_full, args.aggregation)
    causal_heatmaps = aggregate_channels(x_causal, args.aggregation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_heatmaps("full", full_heatmaps, base_image, args.output_dir, args.overlay_alpha, args.grid_size)
    save_heatmaps("causal", causal_heatmaps, base_image, args.output_dir, args.overlay_alpha, args.grid_size)
    save_comparison(full_heatmaps, causal_heatmaps, base_image, args.output_dir, args.overlay_alpha, args.grid_size)
    np.savez_compressed(
        args.output_dir / "pre_head_channel_heatmaps.npz",
        full=full_heatmaps,
        causal=causal_heatmaps,
        mask_prob=mask_prob.detach().float().cpu().numpy(),
        logits_full=logits_full.detach().float().cpu().numpy(),
        logits_causal=logits_causal.detach().float().cpu().numpy(),
    )

    top_full = int(logits_full.argmax(dim=1).item())
    top_causal = int(logits_causal.argmax(dim=1).item())
    print(f"checkpoint: {args.checkpoint}")
    print(f"image: {args.image}")
    print(f"feature shape full: {tuple(x_full.shape)}")
    print(f"feature shape causal: {tuple(x_causal.shape)}")
    print(f"aggregation: channel {args.aggregation}, timesteps={args.time_step}")
    print(f"lif backend: {args.lif_backend} ({changed_backend} modules)")
    print(f"grid size: {args.grid_size}x{args.grid_size}")
    print(f"top1 full={top_full}, causal={top_causal}")
    print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    print(f"saved to: {args.output_dir}")

    functional.reset_net(model)


if __name__ == "__main__":
    main()
