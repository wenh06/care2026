"""Transfer nnUNet Dataset293 encoder weights to VNet encoder.

nnUNet ResEnc U-Net and VNet share similar 3D CNN encoder structures
(Conv + InstanceNorm/BatchNorm at each stage).  This script matches
parameters by their tensor shape — no hardcoded layer names.

Usage:
    python scripts/transfer_nnunet_weights.py \\
        --nnunet-ckpt checkpoints/nnunet/results/Dataset293_.../fold_0/checkpoint_final.pth \\
        --backbone vnet_ct \\
        --output checkpoints/vnet_ct_nnunet_pretrained.safetensors
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfg import ModelCfg  # noqa: E402
from models.vnet import VNet  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Transfer nnUNet weights to VNet encoder")
    parser.add_argument("--nnunet-ckpt", required=True, help="Path to nnUNet checkpoint_final.pth")
    parser.add_argument("--backbone", default="vnet_ct", choices=["vnet_ct", "nested_vnet_ct"])
    parser.add_argument("--output", required=True, help="Output .safetensors path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Load nnUNet weights ---
    print(f"Loading nnUNet checkpoint: {args.nnunet_ckpt}")
    nnunet_ckpt = torch.load(args.nnunet_ckpt, map_location=device, weights_only=False)
    nnunet_sd = nnunet_ckpt.get("network_weights", nnunet_ckpt.get("state_dict", {}))
    if not nnunet_sd:
        raise ValueError("Could not find network_weights or state_dict in nnUNet checkpoint")

    # --- Build VNet ---
    print(f"Building VNet backbone: {args.backbone}")
    vnet_cfg = ModelCfg[args.backbone]
    vnet = VNet(vnet_cfg)
    vnet_sd = vnet.state_dict()

    # --- Match by shape ---
    # Build a shape → list of nnUNet keys map (excluding decoder keys)
    nnunet_by_shape = {}
    for k, v in nnunet_sd.items():
        # Only encoder
        if not k.startswith("encoder"):
            continue
        shape = tuple(v.shape)
        nnunet_by_shape.setdefault(shape, []).append(k)

    matched = {}
    skipped = []
    unmatched_vnet = []

    for vnet_key, vnet_param in vnet_sd.items():
        if not vnet_key.startswith("encoder"):
            continue
        shape = tuple(vnet_param.shape)
        candidates = nnunet_by_shape.get(shape, [])
        if candidates:
            # Pick the first unused candidate
            picked = candidates.pop(0)
            if not candidates:  # remove empty list
                del nnunet_by_shape[shape]
            matched[vnet_key] = nnunet_sd[picked]
        else:
            unmatched_vnet.append((vnet_key, shape))

    # Report
    n_vnet_enc = sum(1 for k in vnet_sd if k.startswith("encoder"))
    n_matched = len(matched)
    print(f"\nVNet encoder params: {n_vnet_enc}")
    print(f"Matched: {n_matched} ({100*n_matched/n_vnet_enc:.0f}%)")
    print(f"Unmatched (train from scratch): {len(unmatched_vnet)}")
    if unmatched_vnet:
        print("  Examples:")
        for k, s in unmatched_vnet[:5]:
            print(f"    {k}: {s}")

    # --- Load matched weights into VNet ---
    # First, normalize BatchNorm weights: nnUNet uses InstanceNorm but our VNet uses BatchNorm.
    # The norm weight/bias shapes are compatible (both are per-channel), so they transfer fine.
    vnet_sd.update(matched)
    vnet.load_state_dict(vnet_sd, strict=False)

    # --- Save ---
    # Save just the weight tensors (not the full checkpoint, for size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove running_mean/var from save (they're from scratch init)
    save_sd = {
        k: v
        for k, v in vnet.state_dict().items()
        if not any(x in k for x in ["running_mean", "running_var", "num_batches_tracked"])
    }
    save_file(save_sd, output_path)
    print(f"\nSaved {len(save_sd)} tensors to {output_path}")


if __name__ == "__main__":
    main()
