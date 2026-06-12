"""
Build the model from a config file and print parameter counts / layer summary.
No training, dataloaders, or experiment directories.

Usage:
    python count_params.py -c config/npy_1000linear_-6_-2_b4_nc64.json
    python count_params.py -c config/npy_1000linear_-6_-2_b4_nc64.json --detailed

Install torchinfo for the detailed layer-by-layer table (optional):
    pip install torchinfo
"""
import argparse
import json
from collections import OrderedDict

import torch

from models import define_network


class _StdoutLogger:
    def info(self, msg, *args, **kwargs):
        print(msg)

    def warning(self, msg, *args, **kwargs):
        print(f"WARNING: {msg}")


def parse_config(config_path):
    """Load JSON config (supports // comments), without creating experiment dirs."""
    json_str = ""
    with open(config_path, "r") as f:
        for line in f:
            json_str += line.split("//")[0] + "\n"
    return json.loads(json_str, object_pairs_hook=OrderedDict)


def param_stats(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_param_stats(module, name):
    total, trainable = param_stats(module)
    print(f"\n{'=' * 60}")
    print(name)
    print(f"  Total params:      {total:>15,}")
    print(f"  Trainable params:  {trainable:>15,}")
    print(f"  Non-trainable:     {total - trainable:>15,}")
    print(f"  Size (float32):    {total * 4 / 1024 ** 2:>12.2f} MB")
    return total


def print_torchinfo_summary(module, name, input_data=None, **summary_kwargs):
    try:
        from torchinfo import summary
    except ImportError:
        print(
            "\nSkipping detailed summary (torchinfo not installed). "
            "Run: pip install torchinfo"
        )
        return

    print(f"\n{'=' * 60}")
    print(f"torchinfo summary: {name}")
    print("=" * 60)
    summary(module, input_data=input_data, **summary_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Count model parameters from a config file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/npy_1000linear_-6_-2_b4_nc64.json",
        help="JSON config (same as run.py)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print layer-by-layer torchinfo summary for each sub-network",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for dummy inputs in detailed summary",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override image size (default: read from config unet.image_size)",
    )
    args = parser.parse_args()

    opt = parse_config(args.config)
    opt["phase"] = "test"
    opt["distributed"] = False
    opt["global_rank"] = 0
    opt["local_rank"] = 0

    logger = _StdoutLogger()
    print(f"Config: {args.config}")
    print(f"Model:  {opt['name']}")

    networks = [
        define_network(logger, opt, item_opt)
        for item_opt in opt["model"]["which_networks"]
    ]
    net = networks[0]

    unet_cfg = opt["model"]["which_networks"][0]["args"]["unet"]
    image_size = args.image_size or unet_cfg["image_size"]
    b = args.batch_size

    # Sub-networks (same structure as models/network.py)
    print_param_stats(net, f"Full Network ({net.__class__.__name__})")
    print_param_stats(net.denoise_fn, "  └─ denoise_fn (diffusion UNet)")
    print_param_stats(net.initial_net, "  └─ initial_net (Pix2Pix-style UNet)")

    if not args.detailed:
        print(
            "\nTip: pass --detailed for a per-layer breakdown (requires torchinfo)."
        )
        return

    device = torch.device("cpu")
    net = net.to(device)
    net.set_new_noise_schedule(device=device, phase="test")

    # denoise_fn(x, gammas, cond) — x is cat([y_cond, y_noisy], dim=1) => 5 channels
    dummy_x = torch.randn(b, unet_cfg["in_channel"], image_size, image_size, device=device)
    dummy_gammas = torch.rand(b, device=device)
    dummy_init_cond = net.initial_net(torch.randn(b, 4, image_size, image_size, device=device))[1]

    print_torchinfo_summary(
        net.denoise_fn,
        "denoise_fn",
        input_data=(dummy_x, dummy_gammas, dummy_init_cond),
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        depth=4,
        verbose=1,
    )

    print_torchinfo_summary(
        net.initial_net,
        "initial_net",
        input_size=(b, 4, image_size, image_size),
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        depth=4,
        verbose=1,
    )


if __name__ == "__main__":
    main()
