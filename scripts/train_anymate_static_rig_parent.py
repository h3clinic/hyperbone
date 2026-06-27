from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.losses.rig_graph_losses_v2 import RigGraphLossV2, hungarian_match_batch
from hyperbone.models.hyperbone_static_parent import HyperBoneStaticParentModel
from hyperbone.rigs.parent_candidates import build_parent_candidates, gather_nonroot_candidate_logits, gather_parent_candidate_logits, mask_parent_logits_with_candidates


def build_direct_parent_supervision(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    candidate_parent_k: int | None = None,
    force_gt_parent_candidate: bool = False,
    two_stage_root_parent: bool = False,
) -> Dict[str, torch.Tensor]:
    """Use GT joint slots directly for parent supervision."""
    joint_active = batch["joint_active"].float().to(device)
    parent_index = batch["parent_index"].long().to(device)
    root_mask = batch["root_mask"].float().to(device)
    valid_parent_mask = batch["valid_parent_mask"].float().to(device)
    bone_vector_to_parent = batch["bone_vector_to_parent"].float().to(device)

    B, J = parent_index.shape
    root_class = J

    parent_targets = parent_index.clone()
    parent_targets[parent_targets < 0] = root_class
    parent_targets[joint_active < 0.5] = root_class

    # Self-parent is forbidden; force such targets to ROOT.
    slot_ids = torch.arange(J, device=device).unsqueeze(0).expand(B, J)
    self_parent = (parent_targets == slot_ids) & (joint_active > 0.5)
    parent_targets[self_parent] = root_class

    root_targets = root_mask.clone()
    root_targets[joint_active < 0.5] = 0.0

    nonroot_mask = (valid_parent_mask > 0.5) & (joint_active > 0.5) & (~self_parent)
    valid_mask = joint_active > 0.5

    # Convention audit: class space must be [0..J], where J is ROOT class.
    if valid_mask.any():
        valid_targets = parent_targets[valid_mask]
        if int(valid_targets.min().item()) < 0 or int(valid_targets.max().item()) > root_class:
            raise ValueError("Invalid parent target class detected outside [0..ROOT].")

    supervision = {
        "parent_targets": parent_targets,
        "root_targets": root_targets,
        "nonroot_mask": nonroot_mask.float(),
        "valid_mask": valid_mask.float(),
        "offset_targets": bone_vector_to_parent,
    }

    if candidate_parent_k is not None:
        candidate_info = build_parent_candidates(
            batch["joint_pos"],
            joint_active > 0.5,
            parent_index,
            k=int(candidate_parent_k),
            include_root=not two_stage_root_parent,
            force_gt_parent=force_gt_parent_candidate,
        )
        supervision["candidate_indices"] = candidate_info["candidate_indices"]
        supervision["candidate_mask"] = candidate_info["candidate_mask"]
        supervision["candidate_targets"] = candidate_info["target_candidate_class"]
        supervision["candidate_valid_mask"] = candidate_info["gt_parent_in_candidates"] | (~nonroot_mask.bool())
        supervision["forced_gt_parent"] = candidate_info["forced_gt_parent"]
        supervision["two_stage_root_parent"] = two_stage_root_parent

    return supervision


def build_parent_supervision(batch: Dict[str, torch.Tensor], pred_matched: torch.Tensor, gt_matched: torch.Tensor, match_valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build parent-pointer training targets in prediction slot space."""
    device = pred_matched.device
    B, J = batch["joint_pos"].shape[:2]
    root_class = J

    parent_targets = torch.full((B, J), root_class, dtype=torch.long, device=device)
    root_targets = torch.zeros((B, J), dtype=torch.float32, device=device)
    offset_targets = torch.zeros((B, J, 3), dtype=torch.float32, device=device)
    offset_valid = torch.zeros((B, J), dtype=torch.float32, device=device)

    gt_parent_index = batch["parent_index"].long().to(device)
    gt_root_mask = batch["root_mask"].float().to(device)
    gt_offset = batch["bone_vector_to_parent"].float().to(device)

    for b in range(B):
        valid = match_valid[b]
        if not valid.any():
            continue
        p_idx = pred_matched[b, valid]
        g_idx = gt_matched[b, valid]
        gt_to_pred = {int(g.item()): int(p.item()) for p, g in zip(p_idx, g_idx)}

        for p_slot, g_slot in zip(p_idx.tolist(), g_idx.tolist()):
            parent_targets[b, p_slot] = root_class
            root_targets[b, p_slot] = gt_root_mask[b, g_slot]
            offset_targets[b, p_slot] = gt_offset[b, g_slot]
            offset_valid[b, p_slot] = 1.0

            parent_gt = int(gt_parent_index[b, g_slot].item())
            if parent_gt >= 0 and parent_gt in gt_to_pred:
                parent_targets[b, p_slot] = gt_to_pred[parent_gt]
            else:
                parent_targets[b, p_slot] = root_class

    return parent_targets, root_targets, offset_targets, offset_valid


@torch.no_grad()
def decode_cycle_rate(parent_logits: torch.Tensor, active_prob: torch.Tensor, active_threshold: float) -> float:
    """Non-differentiable cycle-rate diagnostic from argmax parent pointers."""
    B, J, K = parent_logits.shape
    root_class = K - 1
    rates = []
    for b in range(B):
        active = active_prob[b] > active_threshold
        if active.sum() == 0:
            continue
        parent_choice = parent_logits[b].argmax(dim=-1)
        parent_ptr = torch.full((J,), -1, dtype=torch.long, device=parent_logits.device)
        for j in torch.where(active)[0].tolist():
            choice = int(parent_choice[j].item())
            if choice != root_class and choice != j:
                parent_ptr[j] = choice

        cycle_nodes = 0
        for node in torch.where(active)[0].tolist():
            seen = set()
            cur = node
            while cur >= 0 and cur not in seen:
                seen.add(cur)
                cur = int(parent_ptr[cur].item())
            if cur in seen and cur >= 0:
                cycle_nodes += 1
        rates.append(cycle_nodes / max(int(active.sum().item()), 1))
    return float(sum(rates) / max(len(rates), 1)) if rates else 0.0


def compute_parent_losses(
    pred: Dict[str, torch.Tensor],
    supervision: Dict[str, torch.Tensor],
    epoch: int,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the parent-pointer objective with explicit non-root pressure."""
    parent_logits = pred["parent_logits"]
    parent_offset = pred["parent_offset"]
    edge_conf_logits = pred["edge_confidence_logits"]
    device = parent_logits.device
    root_class = parent_logits.shape[-1] - 1

    def _compute_root_rank_loss() -> torch.Tensor:
        if getattr(args, "root_rank_loss_weight", 0.0) <= 0.0:
            return torch.tensor(0.0, device=device)
        if "root_targets" not in supervision or "valid_mask" not in supervision:
            return torch.tensor(0.0, device=device)

        valid = supervision["valid_mask"] > 0.5
        root_targets_local = supervision["root_targets"] > 0.5
        losses = []

        for batch_index in range(parent_logits.shape[0]):
            pos_mask = valid[batch_index] & root_targets_local[batch_index]
            neg_mask = valid[batch_index] & (~root_targets_local[batch_index])
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            neg_logits = parent_logits[batch_index, neg_mask, root_class]
            hard_k = min(int(getattr(args, "root_hard_neg_k", 8)), int(neg_logits.numel()))
            if hard_k <= 0:
                continue

            hard_neg_logits = torch.topk(neg_logits, k=hard_k, largest=True).values
            pos_logits = parent_logits[batch_index, pos_mask, root_class]
            violation = getattr(args, "root_rank_margin", 1.0) - pos_logits.unsqueeze(1) + hard_neg_logits.unsqueeze(0)
            losses.append(F.relu(violation).pow(2).mean())

        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    valid_mask = supervision["valid_mask"] > 0.5
    nonroot_mask = supervision["nonroot_mask"] > 0.5
    parent_targets = supervision["parent_targets"]
    root_targets = supervision["root_targets"]
    offset_targets = supervision["offset_targets"]
    candidate_logits = supervision.get("candidate_logits")
    candidate_targets = supervision.get("candidate_targets")
    candidate_valid_mask = supervision.get("candidate_valid_mask")

    if args.root_only_training:
        root_logits = pred.get("root_logits")
        if root_logits is None:
            root_logits = parent_logits[:, :, root_class]

        pos_count = (root_targets * valid_mask.float()).sum().clamp(min=1.0)
        neg_count = ((1.0 - root_targets) * valid_mask.float()).sum().clamp(min=1.0)
        pos_weight = (neg_count / pos_count).detach()
        if args.root_pos_weight_cap > 0.0:
            pos_weight = torch.clamp(pos_weight, max=args.root_pos_weight_cap)

        bce_raw = F.binary_cross_entropy_with_logits(
            root_logits,
            root_targets,
            reduction="none",
            pos_weight=pos_weight,
        )
        if args.root_focal_gamma > 0.0:
            p = torch.sigmoid(root_logits)
            p_t = p * root_targets + (1.0 - p) * (1.0 - root_targets)
            bce_raw = bce_raw * (1.0 - p_t).pow(args.root_focal_gamma)
        root_loss = (bce_raw * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)

        gt_root_ratio_val = float(root_targets[valid_mask].mean().item()) if valid_mask.any() else 0.03
        pred_root_prob = torch.sigmoid(root_logits)
        pred_root_ratio_val = (pred_root_prob * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
        root_ratio_loss = F.smooth_l1_loss(
            pred_root_ratio_val,
            torch.tensor(gt_root_ratio_val, device=device, dtype=pred_root_ratio_val.dtype),
            reduction="sum",
        )
        root_overpredict_loss = F.relu(
            pred_root_ratio_val - torch.tensor(gt_root_ratio_val, device=device, dtype=pred_root_ratio_val.dtype)
        ).pow(2)

        # FP margin loss: penalize root logit on GT non-root nodes if > margin.
        fp_margin_loss = torch.tensor(0.0, device=device)
        if getattr(args, "root_fp_margin_weight", 0.0) > 0.0:
            nonroot_mask_f = ((1.0 - root_targets) * valid_mask.float())
            if nonroot_mask_f.sum() > 0.0:
                margin_violation = F.relu(root_logits - args.root_fp_margin)
                fp_margin_loss = (margin_violation.pow(2) * nonroot_mask_f).sum() / nonroot_mask_f.sum()

        root_rank_loss = _compute_root_rank_loss()

        ratio_w = args.root_ratio_weight if args.root_ratio_weight > 0.0 else 1.0
        total = (
            root_loss
            + ratio_w * root_ratio_loss
            + args.root_overpredict_weight * root_overpredict_loss
            + getattr(args, "root_fp_margin_weight", 0.0) * fp_margin_loss
            + getattr(args, "root_rank_loss_weight", 0.0) * root_rank_loss
        )
        metrics = {
            "parent_ce": 0.0,
            "parent_nonroot": 0.0,
            "parent_offset": 0.0,
            "parent_root": float(root_loss.detach().item()),
            "parent_edge_conf": 0.0,
            "parent_ce_scale": 1.0,
            "root_bce": float(root_loss.detach().item()),
            "focal_weighted_loss": float(root_loss.detach().item()),
            "root_rank_loss": float(root_rank_loss.detach().item()),
            "root_ratio_loss": float(root_ratio_loss.detach().item()),
            "root_overpredict_loss": float(root_overpredict_loss.detach().item()),
            "root_fp_margin_loss": float(fp_margin_loss.detach().item()),
            "pred_root_ratio": float(pred_root_ratio_val.detach().item()),
            "gt_root_ratio": float(gt_root_ratio_val),
            "effective_pos_weight": float(pos_weight.item()) if isinstance(pos_weight, torch.Tensor) else float(pos_weight),
        }
        return total, metrics

    if args.two_stage_root_parent:
        root_logits = pred.get("root_logits")
        if root_logits is None:
            root_logits = parent_logits[:, :, root_class]
        candidate_pair_logits = supervision.get("candidate_pair_logits")
        if candidate_pair_logits is None and pred.get("parent_pair_logits") is not None and supervision.get("candidate_indices") is not None:
            candidate_pair_logits = gather_nonroot_candidate_logits(
                pred["parent_pair_logits"],
                supervision["candidate_indices"],
                supervision["candidate_mask"],
            )

        ce_loss = torch.tensor(0.0, device=device)
        if candidate_pair_logits is not None and candidate_targets is not None:
            ce_targets = candidate_targets.clone().masked_fill(candidate_targets < 0, 0)
            ce_raw = F.cross_entropy(
                candidate_pair_logits.reshape(-1, candidate_pair_logits.shape[-1]),
                ce_targets.reshape(-1),
                reduction="none",
            )
            ce_raw = ce_raw.view_as(candidate_targets)
            ce_mask = (valid_mask & nonroot_mask).float()
            if candidate_valid_mask is not None:
                ce_mask = ce_mask * candidate_valid_mask.float()
            ce_loss = (ce_raw * ce_mask).sum() / ce_mask.sum().clamp(min=1.0)

        root_bce_raw = F.binary_cross_entropy_with_logits(
            root_logits, root_targets, reduction="none",
            pos_weight=(
                torch.clamp(
                    ((1.0 - root_targets) * valid_mask.float()).sum().clamp(min=1.0)
                    / (root_targets * valid_mask.float()).sum().clamp(min=1.0),
                    max=args.root_pos_weight_cap,
                ).detach()
                if args.root_pos_weight_cap > 0.0
                else None
            ),
        )
        if args.root_focal_gamma > 0.0:
            p = torch.sigmoid(root_logits)
            p_t = p * root_targets + (1.0 - p) * (1.0 - root_targets)
            root_bce_raw = root_bce_raw * (1.0 - p_t).pow(args.root_focal_gamma)
        root_bce_loss = (root_bce_raw * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)

        gt_root_ratio = float(root_targets[valid_mask].mean().item()) if valid_mask.any() else 0.03
        pred_root_prob = torch.sigmoid(root_logits)
        pred_root_ratio = (pred_root_prob * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
        gt_root_ratio_t = torch.tensor(gt_root_ratio, device=device, dtype=pred_root_ratio.dtype)
        root_ratio_loss = F.smooth_l1_loss(
            pred_root_ratio,
            gt_root_ratio_t,
            reduction="sum",
        )
        root_overpredict_loss = F.relu(pred_root_ratio - gt_root_ratio_t).pow(2)

        # FP margin loss: penalize root logit on GT non-root nodes if > margin.
        fp_margin_loss = torch.tensor(0.0, device=device)
        if getattr(args, 'root_fp_margin_weight', 0.0) > 0.0:
            nonroot_mask_f = ((1.0 - root_targets) * valid_mask.float())
            if nonroot_mask_f.sum() > 0.0:
                margin_violation = F.relu(root_logits - args.root_fp_margin)
                fp_margin_loss = (margin_violation.pow(2) * nonroot_mask_f).sum() / nonroot_mask_f.sum()

        root_rank_loss = _compute_root_rank_loss()

        offset_loss = F.smooth_l1_loss(
            parent_offset * nonroot_mask.unsqueeze(-1).float(),
            offset_targets * nonroot_mask.unsqueeze(-1).float(),
            reduction="sum",
        )
        offset_loss = offset_loss / nonroot_mask.float().sum().clamp(min=1.0) / 3.0

        total = (
            ce_loss
            + args.parent_root_weight * root_bce_loss
            + args.root_ratio_weight * root_ratio_loss
            + args.root_overpredict_weight * root_overpredict_loss
            + getattr(args, 'root_fp_margin_weight', 0.0) * fp_margin_loss
            + getattr(args, 'root_rank_loss_weight', 0.0) * root_rank_loss
            + args.parent_offset_weight * offset_loss
        )
        metrics = {
            "parent_ce": float(ce_loss.detach().item()),
            "parent_nonroot": 0.0,
            "parent_offset": float(offset_loss.detach().item()),
            "parent_root": float(root_bce_loss.detach().item()),
            "parent_edge_conf": float(0.0),
            "parent_ce_scale": float(1.0),
            "root_bce": float(root_bce_loss.detach().item()),
            "focal_weighted_loss": float(root_bce_loss.detach().item()),
            "root_rank_loss": float(root_rank_loss.detach().item()),
            "root_ratio_loss": float(root_ratio_loss.detach().item()),
            "root_overpredict_loss": float(root_overpredict_loss.detach().item()),
            "root_fp_margin_loss": float(fp_margin_loss.detach().item()),
            "pred_root_ratio": float(pred_root_ratio.detach().item()),
            "gt_root_ratio": float(gt_root_ratio),
        }
        return total, metrics

    if candidate_logits is not None and candidate_targets is not None:
        candidate_root_class = candidate_logits.shape[-1] - 1
        class_weights = torch.ones(candidate_root_class + 1, device=device)
        class_weights[candidate_root_class] = args.root_class_weight
        ce_targets = candidate_targets.clone().masked_fill(candidate_targets < 0, candidate_root_class)
        ce_raw = F.cross_entropy(
            candidate_logits.reshape(-1, candidate_root_class + 1),
            ce_targets.reshape(-1),
            weight=class_weights,
            reduction="none",
        )
        ce_raw = ce_raw.view_as(candidate_targets)
        ce_mask = valid_mask.float()
        if candidate_valid_mask is not None:
            ce_mask = ce_mask * candidate_valid_mask.float()
        ce_loss = (ce_raw * ce_mask).sum() / ce_mask.sum().clamp(min=1.0)
    else:
        class_weights = torch.ones(root_class + 1, device=device)
        class_weights[root_class] = args.root_class_weight

        ce_raw = F.cross_entropy(
            parent_logits.reshape(-1, root_class + 1),
            parent_targets.reshape(-1),
            weight=class_weights,
            reduction="none",
        )
        ce_raw = ce_raw.view_as(parent_targets)
        ce_loss = (ce_raw * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)

    nonroot_logit = torch.logsumexp(parent_logits[:, :, :root_class], dim=-1) - parent_logits[:, :, root_class]
    nonroot_loss = F.binary_cross_entropy_with_logits(
        nonroot_logit, nonroot_mask.float(), reduction="none"
    )
    nonroot_loss = (nonroot_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)

    offset_loss = F.smooth_l1_loss(
        parent_offset * nonroot_mask.unsqueeze(-1).float(),
        offset_targets * nonroot_mask.unsqueeze(-1).float(),
        reduction="sum",
    )
    offset_loss = offset_loss / nonroot_mask.float().sum().clamp(min=1.0) / 3.0

    root_loss = F.binary_cross_entropy_with_logits(
        parent_logits[:, :, root_class],
        root_targets,
        reduction="none",
    )
    root_loss = (root_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)

    root_rank_loss = _compute_root_rank_loss()

    edge_conf_target = nonroot_mask.float()
    edge_conf_loss = F.binary_cross_entropy_with_logits(edge_conf_logits, edge_conf_target, reduction="none")
    edge_conf_loss = (edge_conf_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)

    # Root ratio calibration loss (for pairwise head).
    root_logit_col = parent_logits[:, :, root_class]
    gt_root_ratio_val = float(root_targets[valid_mask].mean().item()) if valid_mask.any() else 0.03
    pred_root_prob = torch.sigmoid(root_logit_col)
    pred_root_ratio_val = (pred_root_prob * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
    root_ratio_loss = F.smooth_l1_loss(
        pred_root_ratio_val,
        torch.tensor(gt_root_ratio_val, device=device, dtype=pred_root_ratio_val.dtype),
        reduction="sum",
    )

    ce_warmup = min(1.0, float(epoch + 1) / max(args.parent_ce_warmup_epochs, 1))
    ce_scale = args.parent_ce_start_weight + (1.0 - args.parent_ce_start_weight) * ce_warmup
    offset_scale = args.parent_offset_weight

    total = (
        ce_scale * args.parent_ce_weight * ce_loss
        + args.parent_nonroot_weight * nonroot_loss
        + offset_scale * offset_loss
        + args.parent_root_weight * root_loss
        + args.parent_edge_conf_weight * edge_conf_loss
        + getattr(args, "root_loss_weight", 0.0) * root_loss
        + getattr(args, "root_ratio_weight", 0.0) * root_ratio_loss
        + getattr(args, "root_rank_loss_weight", 0.0) * root_rank_loss
    )

    metrics = {
        "parent_ce": float(ce_loss.detach().item()),
        "parent_nonroot": float(nonroot_loss.detach().item()),
        "parent_offset": float(offset_loss.detach().item()),
        "parent_root": float(root_loss.detach().item()),
        "parent_edge_conf": float(edge_conf_loss.detach().item()),
        "parent_ce_scale": float(ce_scale),
        "root_bce": float(root_loss.detach().item()),
        "focal_weighted_loss": float(root_loss.detach().item()),
        "root_rank_loss": float(root_rank_loss.detach().item()),
        "root_ratio_loss": float(root_ratio_loss.detach().item()),
        "root_overpredict_loss": 0.0,
        "root_fp_margin_loss": 0.0,
        "pred_root_ratio": float(pred_root_ratio_val.detach().item()),
        "gt_root_ratio": float(gt_root_ratio_val),
    }
    return total, metrics


def train_epoch(model, loader, criterion, optimizer, device, amp_scaler, active_threshold: float, epoch: int, args):
    model.train()
    total_loss = 0.0
    total_metrics = {"chamfer": 0.0, "spread_score": 0.0, "count_ratio": 0.0, "cycle_rate": 0.0, "parent_ce": 0.0, "parent_nonroot": 0.0, "parent_offset": 0.0, "parent_root": 0.0, "root_bce": 0.0, "focal_weighted_loss": 0.0, "gt_root_ratio": 0.0, "root_ratio_loss": 0.0, "root_overpredict_loss": 0.0, "root_fp_margin_loss": 0.0, "root_rank_loss": 0.0, "pred_root_ratio": 0.0, "effective_pos_weight": 0.0}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()

        use_gt_nodes = args.teacher_force_gt_nodes or args.parent_supervision_mode == "gt_index"
        if args.parent_supervision_mode == "hybrid":
            use_gt_nodes = epoch < args.hybrid_switch_epoch

        with torch.amp.autocast("cuda", enabled=amp_scaler is not None):
            if use_gt_nodes:
                pred = model.forward_parent_from_joints(
                    batch["joint_pos"],
                    active_mask=batch["joint_active"] > 0.5,
                    no_backbone_for_gt_nodes=args.no_backbone_for_gt_nodes,
                )
                supervision = build_direct_parent_supervision(
                    batch,
                    device,
                    candidate_parent_k=args.candidate_parent_k,
                    force_gt_parent_candidate=args.force_gt_parent_candidate,
                    two_stage_root_parent=args.two_stage_root_parent,
                )
                if args.two_stage_root_parent and "candidate_indices" in supervision:
                    supervision["candidate_pair_logits"] = gather_nonroot_candidate_logits(
                        pred["parent_pair_logits"],
                        supervision["candidate_indices"],
                        supervision["candidate_mask"],
                    )
                elif "candidate_indices" in supervision:
                    supervision["candidate_logits"] = gather_parent_candidate_logits(
                        pred["parent_logits"],
                        supervision["candidate_indices"],
                        supervision["candidate_mask"],
                    )
                loss, loss_metrics = compute_parent_losses(pred, supervision, epoch, args)
                base_losses = {"total": torch.tensor(0.0, device=device), "chamfer": torch.tensor(0.0, device=device)}
                active_prob = batch["joint_active"]
            else:
                pred = model(batch)
                base_losses = criterion(pred, batch)
                pred_matched, gt_matched, match_valid = hungarian_match_batch(
                    pred["joint_pos"], batch["joint_pos"], torch.sigmoid(pred["active_logits"]), batch["joint_active"]
                )
                parent_targets, root_targets, offset_targets, offset_valid = build_parent_supervision(
                    batch, pred_matched, gt_matched, match_valid
                )
                supervision = {
                    "parent_targets": parent_targets,
                    "root_targets": root_targets,
                    "nonroot_mask": offset_valid,
                    "valid_mask": offset_valid,
                    "offset_targets": offset_targets,
                }
                loss, loss_metrics = compute_parent_losses(pred, supervision, epoch, args)
                loss = base_losses["total"] + loss
                active_prob = torch.sigmoid(pred["active_logits"])

            if use_gt_nodes:
                cycle_penalty = torch.tensor(0.0, device=device)
            else:
                cycle_penalty = torch.tensor(decode_cycle_rate(pred["parent_logits"].detach(), active_prob, active_threshold), device=device)
                loss = loss + args.parent_cycle_weight * cycle_penalty

        if amp_scaler:
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_metrics["chamfer"] += base_losses["chamfer"].item()
        total_metrics["spread_score"] += float((active_prob > active_threshold).float().sum().item() > 0)
        total_metrics["count_ratio"] += float((active_prob > active_threshold).float().sum().item() / max(batch["joint_active"].sum().item(), 1.0))
        total_metrics["cycle_rate"] += float(cycle_penalty.item())
        total_metrics["parent_ce"] += loss_metrics["parent_ce"]
        total_metrics["parent_nonroot"] += loss_metrics["parent_nonroot"]
        total_metrics["parent_offset"] += loss_metrics["parent_offset"]
        total_metrics["parent_root"] += loss_metrics["parent_root"]
        total_metrics["root_bce"] += loss_metrics.get("root_bce", loss_metrics["parent_root"])
        total_metrics["focal_weighted_loss"] += loss_metrics.get("focal_weighted_loss", loss_metrics["parent_root"])
        total_metrics["root_ratio_loss"] += loss_metrics.get("root_ratio_loss", 0.0)
        total_metrics["root_overpredict_loss"] += loss_metrics.get("root_overpredict_loss", 0.0)
        total_metrics["root_fp_margin_loss"] += loss_metrics.get("root_fp_margin_loss", 0.0)
        total_metrics["root_rank_loss"] += loss_metrics.get("root_rank_loss", 0.0)
        total_metrics["pred_root_ratio"] += loss_metrics.get("pred_root_ratio", 0.0)
        total_metrics["effective_pos_weight"] += loss_metrics.get("effective_pos_weight", 0.0)
        if use_gt_nodes:
            total_metrics["gt_root_ratio"] += float(supervision["root_targets"][supervision["valid_mask"] > 0.5].mean().item()) if (supervision["valid_mask"] > 0.5).any() else 0.0
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, active_threshold: float, epoch: int, args):
    model.eval()
    total_loss = 0.0
    total_metrics = {"chamfer": 0.0, "spread_score": 0.0, "count_ratio": 0.0, "cycle_rate": 0.0, "parent_ce": 0.0, "parent_nonroot": 0.0, "parent_offset": 0.0, "parent_root": 0.0, "root_bce": 0.0, "focal_weighted_loss": 0.0, "gt_root_ratio": 0.0, "root_ratio_loss": 0.0, "root_overpredict_loss": 0.0, "root_fp_margin_loss": 0.0, "root_rank_loss": 0.0, "pred_root_ratio": 0.0}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        use_gt_nodes = args.teacher_force_gt_nodes or args.parent_supervision_mode == "gt_index"
        if args.parent_supervision_mode == "hybrid":
            use_gt_nodes = epoch < args.hybrid_switch_epoch

        if use_gt_nodes:
            pred = model.forward_parent_from_joints(
                batch["joint_pos"],
                active_mask=batch["joint_active"] > 0.5,
                no_backbone_for_gt_nodes=args.no_backbone_for_gt_nodes,
            )
            supervision = build_direct_parent_supervision(
                batch,
                device,
                candidate_parent_k=args.candidate_parent_k,
                force_gt_parent_candidate=args.force_gt_parent_candidate,
                two_stage_root_parent=args.two_stage_root_parent,
            )
            if args.two_stage_root_parent and "candidate_indices" in supervision:
                supervision["candidate_pair_logits"] = gather_nonroot_candidate_logits(
                    pred["parent_pair_logits"],
                    supervision["candidate_indices"],
                    supervision["candidate_mask"],
                )
            elif "candidate_indices" in supervision:
                supervision["candidate_logits"] = gather_parent_candidate_logits(
                    pred["parent_logits"],
                    supervision["candidate_indices"],
                    supervision["candidate_mask"],
                )
            loss, loss_metrics = compute_parent_losses(pred, supervision, epoch, args)
            base_losses = {"total": torch.tensor(0.0, device=device), "chamfer": torch.tensor(0.0, device=device)}
            active_prob = batch["joint_active"]
        else:
            pred = model(batch)
            base_losses = criterion(pred, batch)
            pred_matched, gt_matched, match_valid = hungarian_match_batch(
                pred["joint_pos"], batch["joint_pos"], torch.sigmoid(pred["active_logits"]), batch["joint_active"]
            )
            parent_targets, root_targets, offset_targets, offset_valid = build_parent_supervision(
                batch, pred_matched, gt_matched, match_valid
            )
            supervision = {
                "parent_targets": parent_targets,
                "root_targets": root_targets,
                "nonroot_mask": offset_valid,
                "valid_mask": offset_valid,
                "offset_targets": offset_targets,
            }
            loss, loss_metrics = compute_parent_losses(pred, supervision, epoch, args)
            loss = base_losses["total"] + loss
            active_prob = torch.sigmoid(pred["active_logits"])

        if use_gt_nodes:
            cycle_penalty = torch.tensor(0.0, device=device)
        else:
            cycle_penalty = torch.tensor(decode_cycle_rate(pred["parent_logits"].detach(), active_prob, active_threshold), device=device)
            loss = loss + args.parent_cycle_weight * cycle_penalty

        total_loss += loss.item()
        total_metrics["chamfer"] += base_losses["chamfer"].item()
        total_metrics["spread_score"] += float((active_prob > active_threshold).float().sum().item() > 0)
        total_metrics["count_ratio"] += float((active_prob > active_threshold).float().sum().item() / max(batch["joint_active"].sum().item(), 1.0))
        total_metrics["cycle_rate"] += float(cycle_penalty)
        total_metrics["parent_ce"] += loss_metrics["parent_ce"]
        total_metrics["parent_nonroot"] += loss_metrics["parent_nonroot"]
        total_metrics["parent_offset"] += loss_metrics["parent_offset"]
        total_metrics["parent_root"] += loss_metrics["parent_root"]
        total_metrics["root_bce"] += loss_metrics.get("root_bce", loss_metrics["parent_root"])
        total_metrics["focal_weighted_loss"] += loss_metrics.get("focal_weighted_loss", loss_metrics["parent_root"])
        total_metrics["root_ratio_loss"] += loss_metrics.get("root_ratio_loss", 0.0)
        total_metrics["root_overpredict_loss"] += loss_metrics.get("root_overpredict_loss", 0.0)
        total_metrics["root_fp_margin_loss"] += loss_metrics.get("root_fp_margin_loss", 0.0)
        total_metrics["root_rank_loss"] += loss_metrics.get("root_rank_loss", 0.0)
        total_metrics["pred_root_ratio"] += loss_metrics.get("pred_root_ratio", 0.0)
        if use_gt_nodes:
            total_metrics["gt_root_ratio"] += float(supervision["root_targets"][supervision["valid_mask"] > 0.5].mean().item()) if (supervision["valid_mask"] > 0.5).any() else 0.0
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train HyperBone static rig parent-pointer model")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--out", default="outputs/models/hyperbone_anymate_static_v2.8_parent_probe")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--pc-points", type=int, default=1024)
    parser.add_argument("--points-per-sample", type=int, default=None)
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--backbone", choices=["pointnet", "dgcnn"], default="dgcnn")
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--active-threshold", type=float, default=0.70)
    parser.add_argument("--parent-supervision-mode", choices=["gt_index", "hungarian", "hybrid"], default="gt_index")
    parser.add_argument("--teacher-force-gt-nodes", action="store_true")
    parser.add_argument("--hybrid-switch-epoch", type=int, default=3)
    parser.add_argument("--parent-ce-weight", type=float, default=1.0)
    parser.add_argument("--parent-ce-start-weight", type=float, default=0.25)
    parser.add_argument("--parent-ce-warmup-epochs", type=int, default=2)
    parser.add_argument("--parent-nonroot-weight", type=float, default=2.0)
    parser.add_argument("--parent-offset-weight", type=float, default=0.5)
    parser.add_argument("--parent-root-weight", type=float, default=0.2)
    parser.add_argument("--parent-edge-conf-weight", type=float, default=0.0)
    parser.add_argument("--parent-cycle-weight", type=float, default=0.05)
    parser.add_argument("--root-class-weight", type=float, default=0.25)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--w-bone", type=float, default=0.8)
    parser.add_argument("--w-unmatched", type=float, default=0.3)
    parser.add_argument("--w-count", type=float, default=0.05)
    parser.add_argument("--w-parent-ce", type=float, default=1.0)
    parser.add_argument("--w-parent-offset", type=float, default=0.5)
    parser.add_argument("--w-root", type=float, default=0.2)
    parser.add_argument("--w-cycle", type=float, default=0.05)
    parser.add_argument("--parent-head", choices=["slot", "pairwise"], default="slot")
    parser.add_argument("--no-backbone-for-gt-nodes", action="store_true")
    parser.add_argument("--root-feature-mode", choices=["none", "structural"], default="none")
    parser.add_argument("--root-loss-weight", type=float, default=0.0)
    parser.add_argument("--root-ratio-weight", type=float, default=0.0)
    parser.add_argument("--root-bias-init", type=float, default=None)
    parser.add_argument("--candidate-parent-k", type=int, default=None)
    parser.add_argument("--force-gt-parent-candidate", action="store_true")
    parser.add_argument("--two-stage-root-parent", action="store_true")
    parser.add_argument("--root-only-training", action="store_true")
    parser.add_argument("--root-focal-gamma", type=float, default=0.0)
    parser.add_argument("--root-pos-weight-cap", type=float, default=8.0)
    parser.add_argument("--root-overpredict-weight", type=float, default=0.0)
    parser.add_argument("--root-fp-margin-weight", type=float, default=0.0,
                        help="Weight for false-positive root margin loss on GT non-root nodes.")
    parser.add_argument("--root-fp-margin", type=float, default=-2.0,
                        help="Margin threshold for non-root root logits (penalize if root_logit > margin).")
    parser.add_argument("--root-rank-loss-weight", type=float, default=0.0,
                        help="Weight for per-sample hard-negative root ranking loss.")
    parser.add_argument("--root-rank-margin", type=float, default=1.0,
                        help="Margin for the hard-negative root ranking loss.")
    parser.add_argument("--root-hard-neg-k", type=int, default=8,
                        help="Number of hardest negative roots per sample to use for ranking loss.")
    parser.add_argument("--root-threshold", type=float, default=None,
                        help="Calibrated root decision threshold for decode/eval. Defaults to active_threshold.")
    parser.add_argument("--root-ambiguity-eps", type=float, default=0.0,
                        help="Exclude conflicting coincident root/non-root nodes within this distance from root metrics.")
    args = parser.parse_args()
    if args.points_per_sample is not None:
        args.pc_points = args.points_per_sample

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train-parent] Device: {device}")
    print(f"[Train-parent] Output: {out_dir}")

    print("[Train-parent] Loading datasets...")
    train_ds = AnymateStaticRigDataset(args.pt, f"{args.splits_dir}/train.jsonl", max_joints=args.max_nodes, pc_points=args.pc_points)
    val_ds = AnymateStaticRigDataset(args.pt, f"{args.splits_dir}/val.jsonl", max_joints=args.max_nodes, pc_points=args.pc_points)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"[Train-parent] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"[Train-parent] Batches/epoch: {len(train_loader)} train, {len(val_loader)} val")

    model = HyperBoneStaticParentModel(
        in_channels=3,
        feat_dim=args.feat_dim,
        max_joints=args.max_nodes,
        predict_skinning=False,
        backbone=args.backbone,
        knn_k=args.knn_k,
        parent_head=getattr(args, "parent_head", "slot"),
        root_bias_init=getattr(args, "root_bias_init", None),
        root_feature_mode=getattr(args, "root_feature_mode", "none"),
    ).to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Train-parent] Resumed from {args.resume}")
        print(f"[Train-parent] Missing keys: {len(missing)} Unexpected keys: {len(unexpected)}")

    criterion = RigGraphLossV2(
        w_hungarian_pos=2.0,
        w_chamfer=1.0,
        w_repulsion=0.5,
        w_edge=0.0,
        w_bone_length=args.w_bone,
        w_active=0.3,
        w_unmatched=args.w_unmatched,
        w_count=args.w_count,
        w_skinning=0.0,
        edge_pos_weight=3.0,
        edge_fp_weight=1.0,
        count_overpredict_scale=0.25,
        count_underpredict_scale=1.0,
        active_pos_weight=2.0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    use_amp = args.amp and not args.no_amp
    amp_scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and use_amp) else None

    best_val_loss = float("inf")
    training_log = []

    print(f"\n[Train-parent] Starting: {args.epochs} epochs, lr={args.lr}, bs={args.batch_size}")
    print(f"[Train-parent] Backbone={args.backbone} knn_k={args.knn_k} active_threshold={args.active_threshold} amp={use_amp}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, amp_scaler, args.active_threshold, epoch, args)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device, args.active_threshold, epoch, args)
        scheduler.step()
        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_chamfer": round(train_metrics["chamfer"], 6),
            "val_chamfer": round(val_metrics["chamfer"], 6),
            "train_count_ratio": round(train_metrics["count_ratio"], 4),
            "val_count_ratio": round(val_metrics["count_ratio"], 4),
            "train_cycle_rate": round(train_metrics["cycle_rate"], 4),
            "val_cycle_rate": round(val_metrics["cycle_rate"], 4),
            "train_root_bce": round(train_metrics.get("root_bce", 0.0), 6),
            "val_root_bce": round(val_metrics.get("root_bce", 0.0), 6),
            "train_focal_weighted_loss": round(train_metrics.get("focal_weighted_loss", 0.0), 6),
            "val_focal_weighted_loss": round(val_metrics.get("focal_weighted_loss", 0.0), 6),
            "train_root_ratio_loss": round(train_metrics.get("root_ratio_loss", 0.0), 6),
            "val_root_ratio_loss": round(val_metrics.get("root_ratio_loss", 0.0), 6),
            "train_root_overpredict_loss": round(train_metrics.get("root_overpredict_loss", 0.0), 6),
            "val_root_overpredict_loss": round(val_metrics.get("root_overpredict_loss", 0.0), 6),
            "train_root_fp_margin_loss": round(train_metrics.get("root_fp_margin_loss", 0.0), 6),
            "val_root_fp_margin_loss": round(val_metrics.get("root_fp_margin_loss", 0.0), 6),
            "train_root_rank_loss": round(train_metrics.get("root_rank_loss", 0.0), 6),
            "val_root_rank_loss": round(val_metrics.get("root_rank_loss", 0.0), 6),
            "train_pred_root_ratio": round(train_metrics.get("pred_root_ratio", 0.0), 6),
            "val_pred_root_ratio": round(val_metrics.get("pred_root_ratio", 0.0), 6),
            "train_gt_root_ratio": round(train_metrics.get("gt_root_ratio", 0.0), 6),
            "val_gt_root_ratio": round(val_metrics.get("gt_root_ratio", 0.0), 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "time_s": round(elapsed, 1),
        }
        training_log.append(log_entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), out_dir / f"checkpoint_e{epoch:03d}.pt")

        print(
            f"E{epoch:03d} | loss={train_loss:.4f}/{val_loss:.4f} | "
            f"chamfer={train_metrics['chamfer']:.4f}/{val_metrics['chamfer']:.4f} | "
            f"cnt={val_metrics['count_ratio']:.3f} cyc={val_metrics['cycle_rate']:.3f} | "
            f"parent_ce={val_metrics.get('parent_ce', 0.0):.3f} parent_nr={val_metrics.get('parent_nonroot', 0.0):.3f} | "
            f"{elapsed:.1f}s"
        )

    torch.save(model.state_dict(), out_dir / "model_final.pt")
    with open(out_dir / "training_log.jsonl", "w") as f:
        for row in training_log:
            f.write(json.dumps(row) + "\n")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "model": "HyperBoneStaticParentModel",
            "backbone": args.backbone,
            "knn_k": args.knn_k,
            "max_nodes": args.max_nodes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "best_val_loss": round(best_val_loss, 6),
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("[Train-parent] DONE")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Saved: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
