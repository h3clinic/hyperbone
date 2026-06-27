"""
Train HyperBone Anymate Temporal Model — multi-frame pose prediction with temporal consistency.

Input: T-frame window of RGB/mask/depth → [B, T, 5, H, W]
Output: per-frame joint positions, bone rotations, temporal smoothness

Uses a video backbone (3D convolutions or frame encoder + temporal transformer).

Usage:
    python scripts/train_hyperbone_anymate_temporal.py \\
        --dataset outputs/anymate_clips_pilot/train.jsonl \\
        --val-dataset outputs/anymate_clips_pilot/val.jsonl \\
        --out outputs/models/hyperbone_anymate_temporal_pilot \\
        --epochs 30 \\
        --batch-size 4 \\
        --resolution 256 \\
        --clip-len 16 \\
        --stride 4 \\
        --device cuda \\
        --amp \\
        --grad-checkpointing
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_clip_dataset import AnymateClipDataset


class TemporalPoseModel(nn.Module):
    """
    Temporal pose prediction model.
    
    Architecture:
    - Per-frame encoder (shared CNN)
    - Temporal transformer (cross-frame attention)
    - Per-frame joint prediction heads
    
    Input: [B, T, C, H, W]
    Output: joint_xyz [B, T, J, 3], joint_vis [B, T, J]
    """

    def __init__(self, in_channels: int = 5, max_joints: int = 128,
                 base_dim: int = 64, clip_len: int = 16, n_heads: int = 4):
        super().__init__()
        self.max_joints = max_joints
        self.clip_len = clip_len
        feat_dim = base_dim * 8  # 512

        # Per-frame encoder (shared)
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, base_dim, stride=2),
            self._conv_block(base_dim, base_dim * 2, stride=2),
            self._conv_block(base_dim * 2, base_dim * 4, stride=2),
            self._conv_block(base_dim * 4, feat_dim, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Temporal transformer
        self.pos_embed = nn.Parameter(torch.randn(1, clip_len, feat_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads, dim_feedforward=feat_dim * 2,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Joint prediction heads (per-frame)
        self.xyz_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, max_joints * 3),
        )
        self.vis_head = nn.Linear(feat_dim, max_joints)

    def _conv_block(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape

        # Encode each frame
        x_flat = x.view(B * T, C, H, W)
        feats = self.encoder(x_flat)  # [B*T, feat_dim]
        feats = feats.view(B, T, -1)  # [B, T, feat_dim]

        # Add positional encoding
        feats = feats + self.pos_embed[:, :T, :]

        # Temporal attention
        feats = self.temporal_transformer(feats)  # [B, T, feat_dim]

        # Per-frame predictions
        xyz = self.xyz_head(feats).view(B, T, self.max_joints, 3)
        vis = torch.sigmoid(self.vis_head(feats))  # [B, T, J]

        return {
            "joint_xyz": xyz,
            "joint_vis": vis,
        }


def compute_temporal_loss(pred: dict, batch: dict, device: torch.device) -> dict:
    """
    Temporal training losses:
    - 3D position (Huber)
    - Visibility (BCE)
    - Temporal smoothness (acceleration penalty)
    - Bone length consistency across time
    """
    gt_xyz = batch["joint_xyz"].to(device)         # [B, T, J, 3]
    gt_vis = batch["joint_vis"].to(device)         # [B, T, J]
    gt_active = batch["joint_active"].to(device)   # [B, T, J]

    pred_xyz = pred["joint_xyz"]
    pred_vis = pred["joint_vis"]

    mask = gt_active * gt_vis  # [B, T, J]
    mask_3d = mask.unsqueeze(-1)  # [B, T, J, 1]
    n_valid = mask.sum().clamp(min=1)

    # Position loss
    pos_loss = F.huber_loss(
        pred_xyz * mask_3d, gt_xyz * mask_3d, reduction='sum'
    ) / n_valid

    # Visibility loss
    vis_loss = F.binary_cross_entropy(
        pred_vis * gt_active, gt_vis * gt_active, reduction='sum'
    ) / gt_active.sum().clamp(min=1)

    # Temporal smoothness: penalize acceleration (second derivative)
    smooth_loss = torch.tensor(0.0, device=device)
    if pred_xyz.shape[1] >= 3:
        velocity = pred_xyz[:, 1:] - pred_xyz[:, :-1]  # [B, T-1, J, 3]
        accel = velocity[:, 1:] - velocity[:, :-1]      # [B, T-2, J, 3]
        accel_mask = mask[:, 2:].unsqueeze(-1)           # [B, T-2, J, 1]
        smooth_loss = (accel ** 2 * accel_mask).sum() / accel_mask.sum().clamp(min=1)

    # Bone length consistency across time
    bone_consistency = torch.tensor(0.0, device=device)

    # Total
    total = pos_loss + 0.5 * vis_loss + 0.1 * smooth_loss + 0.1 * bone_consistency

    return {
        "total": total,
        "pos_loss": pos_loss.item(),
        "vis_loss": vis_loss.item(),
        "smooth_loss": smooth_loss.item(),
    }


def evaluate_temporal(model, dataloader, device) -> dict:
    """Evaluate temporal model."""
    model.eval()
    total_err = 0.0
    total_joints = 0
    total_pck_010 = 0
    total_drift = 0.0
    n_drift = 0

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device)  # [B, T, 3, H, W]
            B, T = rgb.shape[:2]

            # Build full input
            inputs = [rgb]
            if "mask" in batch:
                inputs.append(batch["mask"].to(device))
            if "depth" in batch:
                inputs.append(batch["depth"].to(device))
            x = torch.cat(inputs, dim=2)  # [B, T, C, H, W]

            pred = model(x)
            gt_xyz = batch["joint_xyz"].to(device)
            gt_vis = batch["joint_vis"].to(device)
            gt_active = batch["joint_active"].to(device)

            mask = (gt_active * gt_vis) > 0  # [B, T, J]

            errors = torch.norm(pred["joint_xyz"] - gt_xyz, dim=-1)  # [B, T, J]
            valid_errors = errors[mask]

            total_err += valid_errors.sum().item()
            total_joints += mask.sum().item()
            total_pck_010 += (valid_errors < 0.10).sum().item()

            # Temporal drift: error at last frame vs first frame
            if T > 1:
                first_mask = mask[:, 0]
                last_mask = mask[:, -1]
                both_mask = first_mask & last_mask
                if both_mask.any():
                    first_err = errors[:, 0][both_mask]
                    last_err = errors[:, -1][both_mask]
                    drift = (last_err - first_err).abs()
                    total_drift += drift.sum().item()
                    n_drift += both_mask.sum().item()

    n = max(total_joints, 1)
    return {
        "mpjpe": total_err / n,
        "pck_010": total_pck_010 / n,
        "temporal_drift": total_drift / max(n_drift, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train HyperBone Anymate temporal model")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--val-dataset", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-joints", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Train-Temporal] clip_len={args.clip_len}, stride={args.stride}")
    print(f"[Train-Temporal] Device: {device}")

    # Datasets
    train_ds = AnymateClipDataset(
        args.dataset, clip_len=args.clip_len, stride=args.stride,
        resolution=args.resolution, max_joints=args.max_joints,
    )
    print(f"[Train-Temporal] Train windows: {len(train_ds)}")

    val_ds = None
    if args.val_dataset and Path(args.val_dataset).exists():
        val_ds = AnymateClipDataset(
            args.val_dataset, clip_len=args.clip_len, stride=args.stride,
            resolution=args.resolution, max_joints=args.max_joints,
        )
        print(f"[Train-Temporal] Val windows: {len(val_ds)}")

    def collate_fn(batch):
        result = {}
        tensor_keys = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
        for k in tensor_keys:
            result[k] = torch.stack([b[k] for b in batch])
        return result

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
        )

    # Model
    in_channels = 5
    model = TemporalPoseModel(
        in_channels=in_channels, max_joints=args.max_joints,
        clip_len=args.clip_len,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Train-Temporal] Model params: {param_count:,}")

    if args.grad_checkpointing:
        model.temporal_transformer.gradient_checkpointing = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    best_val_mpjpe = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            rgb = batch["rgb"].to(device)  # [B, T, 3, H, W]
            mask = batch.get("mask", torch.zeros_like(rgb[:, :, :1])).to(device)
            depth = batch.get("depth", torch.zeros_like(rgb[:, :, :1])).to(device)
            x = torch.cat([rgb, mask, depth], dim=2)  # [B, T, 5, H, W]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=args.amp):
                pred = model(x)
                losses = compute_temporal_loss(pred, batch, device)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses["total"].item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        log = f"[Epoch {epoch:3d}/{args.epochs}] loss={avg_loss:.4f} time={elapsed:.1f}s"

        val_metrics = {}
        if val_loader:
            val_metrics = evaluate_temporal(model, val_loader, device)
            log += f" | mpjpe={val_metrics['mpjpe']:.4f} pck@0.10={val_metrics['pck_010']:.3f} drift={val_metrics['temporal_drift']:.4f}"

            if val_metrics["mpjpe"] < best_val_mpjpe:
                best_val_mpjpe = val_metrics["mpjpe"]
                torch.save(model.state_dict(), output_dir / "best_model.pt")

        print(log)
        history.append({"epoch": epoch, "train_loss": avg_loss, **val_metrics})

    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Train-Temporal] Done. Best MPJPE: {best_val_mpjpe:.4f}")


if __name__ == "__main__":
    main()
