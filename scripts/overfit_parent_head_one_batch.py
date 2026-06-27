from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.models.hyperbone_static_parent import HyperBoneStaticParentModel
from hyperbone.rigs.parent_candidates import (
    build_parent_candidates,
    gather_nonroot_candidate_logits,
    gather_parent_candidate_logits,
    mask_parent_logits_with_candidates,
)
from hyperbone.rigs.parent_decoder import ParentDecodeConfig, decode_parent_graph, decode_two_stage_root_parent_graph
from scripts.train_anymate_static_rig_parent import build_direct_parent_supervision


def edge_metrics(pred_edges: set[tuple[int, int]], gt_edges: set[tuple[int, int]]) -> dict:
    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


def parent_edge_sets(parent_ptr: torch.Tensor, active_mask: torch.Tensor) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    n_nodes = int(parent_ptr.shape[0])
    for child in range(n_nodes):
        parent = int(parent_ptr[child].item())
        if parent >= 0 and bool(active_mask[child].item()) and bool(active_mask[parent].item()):
            a, b = sorted((parent, child))
            edges.add((a, b))
    return edges


@torch.no_grad()
def compute_ambiguous_root_mask(batch: dict, eps: float = 1e-4) -> torch.Tensor:
    """Mark active nodes that are geometrically unresolvable for root prediction.

    A node is ambiguous if it shares (within ``eps``) its position with another
    active node of the opposite root status. When GT joints bypass the backbone,
    every feature is position-derived, so two coincident joints produce identical
    logits everywhere and no deterministic model can separate them. Such nodes are
    excluded from the root sanity metric so the gate measures learnable signal.

    Returns a boolean mask [B, J] that is True for ambiguous nodes.
    """
    joint_pos = batch["joint_pos"]
    active = batch["joint_active"] > 0.5
    gt_root = batch["root_mask"] > 0.5
    B, J, _ = joint_pos.shape
    ambiguous = torch.zeros(B, J, dtype=torch.bool, device=joint_pos.device)
    for b in range(B):
        valid = torch.where(active[b])[0]
        if valid.numel() < 2:
            continue
        pos = joint_pos[b, valid]
        dmat = torch.cdist(pos, pos)
        root_v = gt_root[b, valid]
        conflict = root_v.unsqueeze(0) != root_v.unsqueeze(1)
        coincident = (dmat < eps) & conflict
        coincident.fill_diagonal_(False)
        ambiguous_local = coincident.any(dim=1)
        ambiguous[b, valid[ambiguous_local]] = True
    return ambiguous



def compute_parent_losses(
    pred: dict,
    supervision: dict,
    root_class_weight: float,
    parent_root_weight: float,
    root_loss_weight: float = 0.0,
    root_ratio_weight: float = 0.0,
    two_stage_parent_margin_weight: float = 0.0,
    two_stage_parent_margin_delta: float = 0.5,
    root_only_overfit: bool = False,
    root_focal_gamma: float = 0.0,
    root_pos_weight_cap: float = 8.0,
    root_overpredict_weight: float = 0.0,
    root_fp_margin_weight: float = 0.0,
    root_fp_margin: float = -2.0,
) -> tuple[torch.Tensor, dict]:
    parent_logits = pred["parent_logits"]
    parent_offset = pred["parent_offset"]
    root_class = parent_logits.shape[-1] - 1

    valid_mask = supervision["valid_mask"] > 0.5
    nonroot_mask = supervision["nonroot_mask"] > 0.5
    parent_targets = supervision["parent_targets"]
    root_targets = supervision["root_targets"]
    offset_targets = supervision["offset_targets"]
    candidate_logits = supervision.get("candidate_logits")
    candidate_targets = supervision.get("candidate_targets")
    candidate_valid_mask = supervision.get("candidate_valid_mask")

    if root_only_overfit:
        root_logits = pred.get("root_logits")
        if root_logits is None:
            root_logits = parent_logits[:, :, root_class]

        pos_count = (root_targets * valid_mask.float()).sum().clamp(min=1.0)
        neg_count = ((1.0 - root_targets) * valid_mask.float()).sum().clamp(min=1.0)
        pos_weight = (neg_count / pos_count).detach()
        if root_pos_weight_cap > 0.0:
            pos_weight = torch.clamp(pos_weight, max=root_pos_weight_cap)

        bce_raw = F.binary_cross_entropy_with_logits(
            root_logits,
            root_targets,
            reduction="none",
            pos_weight=pos_weight,
        )
        if root_focal_gamma > 0.0:
            p = torch.sigmoid(root_logits)
            p_t = p * root_targets + (1.0 - p) * (1.0 - root_targets)
            focal = (1.0 - p_t).pow(root_focal_gamma)
            bce_raw = bce_raw * focal
        root_bce_loss = (bce_raw * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)

        gt_root_ratio = float(root_targets[valid_mask].mean().item()) if valid_mask.any() else 0.03
        pred_root_prob = torch.sigmoid(root_logits)
        pred_root_ratio = (pred_root_prob * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
        root_ratio_loss = F.smooth_l1_loss(
            pred_root_ratio,
            torch.tensor(gt_root_ratio, device=parent_logits.device, dtype=pred_root_ratio.dtype),
            reduction="sum",
        )
        root_overpredict_loss = F.relu(
            pred_root_ratio - torch.tensor(gt_root_ratio, device=parent_logits.device, dtype=pred_root_ratio.dtype)
        ).pow(2)

        # FP margin loss: penalize root logit on GT non-root nodes if > margin.
        fp_margin_loss = torch.tensor(0.0, device=parent_logits.device)
        if root_fp_margin_weight > 0.0:
            nonroot_mask_f = ((1.0 - root_targets) * valid_mask.float())
            if nonroot_mask_f.sum() > 0.0:
                margin_violation = F.relu(root_logits - root_fp_margin)
                fp_margin_loss = (margin_violation.pow(2) * nonroot_mask_f).sum() / nonroot_mask_f.sum()

        ratio_w = root_ratio_weight if root_ratio_weight > 0.0 else 1.0
        total = root_bce_loss + ratio_w * root_ratio_loss + root_overpredict_weight * root_overpredict_loss + root_fp_margin_weight * fp_margin_loss
        stats = {
            "parent_ce": 0.0,
            "parent_margin": 0.0,
            "parent_nonroot": 0.0,
            "parent_offset": 0.0,
            "root_bce": float(root_bce_loss.detach().item()),
            "root_ratio_loss": float(root_ratio_loss.detach().item()),
            "root_overpredict_loss": float(root_overpredict_loss.detach().item()),
            "root_fp_margin_loss": float(fp_margin_loss.detach().item()),
        }
        return total, stats

    if supervision.get("two_stage_root_parent", False):
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

        candidate_valid_mask = supervision.get("candidate_valid_mask")
        if candidate_pair_logits is not None and candidate_targets is not None:
            ce_targets = candidate_targets.clone()
            ce_targets = ce_targets.masked_fill(ce_targets < 0, 0)
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

            margin_loss = torch.tensor(0.0, device=parent_logits.device)
            if two_stage_parent_margin_weight > 0.0:
                gt_logits = candidate_pair_logits.gather(-1, ce_targets.unsqueeze(-1)).squeeze(-1)
                neg_logits = candidate_pair_logits.clone()
                neg_logits.scatter_(-1, ce_targets.unsqueeze(-1), -1e4)
                hardest_neg = neg_logits.max(dim=-1).values
                hinge = F.relu(two_stage_parent_margin_delta - (gt_logits - hardest_neg))
                margin_loss = (hinge * ce_mask).sum() / ce_mask.sum().clamp(min=1.0)
        else:
            ce_loss = torch.tensor(0.0, device=parent_logits.device)
            margin_loss = torch.tensor(0.0, device=parent_logits.device)

        root_bce_raw = F.binary_cross_entropy_with_logits(
            root_logits,
            root_targets,
            reduction="none",
            pos_weight=(
                torch.clamp(
                    ((1.0 - root_targets) * valid_mask.float()).sum().clamp(min=1.0)
                    / (root_targets * valid_mask.float()).sum().clamp(min=1.0),
                    max=root_pos_weight_cap,
                ).detach()
                if root_pos_weight_cap > 0.0
                else None
            ),
        )
        if root_focal_gamma > 0.0:
            p = torch.sigmoid(root_logits)
            p_t = p * root_targets + (1.0 - p) * (1.0 - root_targets)
            root_bce_raw = root_bce_raw * (1.0 - p_t).pow(root_focal_gamma)
        root_bce_loss = (root_bce_raw * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)

        gt_root_ratio = float(root_targets[valid_mask].mean().item()) if valid_mask.any() else 0.03
        pred_root_prob = torch.sigmoid(root_logits)
        pred_root_ratio = (pred_root_prob * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
        gt_root_ratio_t = torch.tensor(gt_root_ratio, device=parent_logits.device, dtype=pred_root_ratio.dtype)
        root_ratio_loss = F.smooth_l1_loss(
            pred_root_ratio,
            gt_root_ratio_t,
            reduction="sum",
        )
        root_overpredict_loss = F.relu(pred_root_ratio - gt_root_ratio_t).pow(2)

        # FP margin loss: penalize root logit on GT non-root nodes if > margin.
        fp_margin_loss = torch.tensor(0.0, device=parent_logits.device)
        if root_fp_margin_weight > 0.0:
            nonroot_mask_f = ((1.0 - root_targets) * valid_mask.float())
            if nonroot_mask_f.sum() > 0.0:
                margin_violation = F.relu(root_logits - root_fp_margin)
                fp_margin_loss = (margin_violation.pow(2) * nonroot_mask_f).sum() / nonroot_mask_f.sum()

        offset_loss = F.smooth_l1_loss(
            parent_offset * nonroot_mask.unsqueeze(-1).float(),
            offset_targets * nonroot_mask.unsqueeze(-1).float(),
            reduction="sum",
        )
        offset_loss = offset_loss / nonroot_mask.float().sum().clamp(min=1.0) / 3.0

        total = (
            ce_loss
            + parent_root_weight * root_bce_loss
            + root_ratio_weight * root_ratio_loss
            + root_overpredict_weight * root_overpredict_loss
            + root_fp_margin_weight * fp_margin_loss
            + 0.5 * offset_loss
            + two_stage_parent_margin_weight * margin_loss
        )
        stats = {
            "parent_ce": float(ce_loss.detach().item()),
            "parent_margin": float(margin_loss.detach().item()),
            "parent_nonroot": float(0.0),
            "parent_offset": float(offset_loss.detach().item()),
            "root_bce": float(root_bce_loss.detach().item()),
            "root_ratio_loss": float(root_ratio_loss.detach().item()),
            "root_overpredict_loss": float(root_overpredict_loss.detach().item()),
            "root_fp_margin_loss": float(fp_margin_loss.detach().item()),
        }
        return total, stats

    if candidate_logits is not None and candidate_targets is not None:
        candidate_root_class = candidate_logits.shape[-1] - 1
        class_weights = torch.ones(candidate_root_class + 1, device=parent_logits.device)
        class_weights[candidate_root_class] = root_class_weight
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
        class_weights = torch.ones(root_class + 1, device=parent_logits.device)
        class_weights[root_class] = root_class_weight

        ce_raw = F.cross_entropy(
            parent_logits.reshape(-1, root_class + 1),
            parent_targets.reshape(-1),
            weight=class_weights,
            reduction="none",
        )
        ce_raw = ce_raw.view_as(parent_targets)
        ce_loss = (ce_raw * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)

    # non-root anti-root loss: pushes non-root nodes away from predicting ROOT.
    nonroot_logit = torch.logsumexp(parent_logits[:, :, :root_class], dim=-1) - parent_logits[:, :, root_class]
    nonroot_loss = F.binary_cross_entropy_with_logits(nonroot_logit, nonroot_mask.float(), reduction="none")
    nonroot_loss = (nonroot_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)

    offset_loss = F.smooth_l1_loss(
        parent_offset * nonroot_mask.unsqueeze(-1).float(),
        offset_targets * nonroot_mask.unsqueeze(-1).float(),
        reduction="sum",
    )
    offset_loss = offset_loss / nonroot_mask.float().sum().clamp(min=1.0) / 3.0

    # Root BCE: separate binary loss on root logit column.
    root_logit_col = parent_logits[:, :, root_class]
    root_bce = F.binary_cross_entropy_with_logits(
        root_logit_col,
        root_targets,
        reduction="none",
    )
    root_bce_loss = (root_bce * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)

    # Root ratio calibration: predicted root fraction should match GT root fraction.
    gt_root_ratio = float(root_targets[valid_mask].mean().item()) if valid_mask.any() else 0.03
    pred_root_prob = torch.sigmoid(root_logit_col)
    pred_root_ratio = (pred_root_prob * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
    root_ratio_loss = F.smooth_l1_loss(
        pred_root_ratio,
        torch.tensor(gt_root_ratio, device=parent_logits.device, dtype=pred_root_ratio.dtype),
        reduction="sum",
    )

    total = 1.0 * ce_loss

    if root_loss_weight > 0.0:
        total = total + root_loss_weight * root_bce_loss
    if root_ratio_weight > 0.0:
        total = total + root_ratio_weight * root_ratio_loss
    if bool(supervision.get("use_nonroot_loss", False)):
        total = total + 0.5 * nonroot_loss
    if bool(supervision.get("use_offset_loss", False)):
        total = total + 0.5 * offset_loss
    if bool(supervision.get("use_root_loss", False)):
        total = total + parent_root_weight * root_bce_loss

    stats = {
        "parent_ce": float(ce_loss.detach().item()),
        "parent_nonroot": float(nonroot_loss.detach().item()),
        "parent_offset": float(offset_loss.detach().item()),
        "root_bce": float(root_bce_loss.detach().item()),
        "root_ratio_loss": float(root_ratio_loss.detach().item()),
    }
    return total, stats


@torch.no_grad()
def evaluate_batch(
    pred: dict,
    batch: dict,
    decode_cfg: ParentDecodeConfig,
    mask_root_for_gt_nonroot: bool = False,
    root_ambiguity_eps: float = 0.0,
) -> dict:
    parent_logits = pred["parent_logits"]
    root_class = parent_logits.shape[-1] - 1
    B, J, _ = parent_logits.shape

    valid_mask = batch["joint_active"] > 0.5
    gt_parent = batch["parent_index"].long()
    gt_root = batch["root_mask"] > 0.5

    if decode_cfg.two_stage_root_parent and pred.get("root_logits") is not None and pred.get("candidate_pair_logits") is not None:
        root_prob = torch.sigmoid(pred["root_logits"])
        root_decode_threshold = decode_cfg.root_threshold if decode_cfg.root_threshold is not None else decode_cfg.active_threshold
        pred_root = root_prob > root_decode_threshold
        pred_parent = pred["candidate_pair_logits"].argmax(dim=-1)
        pred_parent_idx = pred.get("candidate_indices")
    else:
        pred_class = parent_logits.argmax(dim=-1)
        pred_root = pred_class == root_class
        pred_parent = pred_class
        pred_parent_idx = None

    parent_correct = 0
    parent_total = 0
    nonroot_correct = 0
    nonroot_total = 0
    root_correct = 0
    root_total = 0

    edge_f1_values = []
    nonroot_edge_f1_values = []
    component_counts = []
    nonroot_ranks = []
    failed_nonroot_count = 0

    for b in range(B):
        valid_b = valid_mask[b]
        for j in torch.where(valid_b)[0]:
            parent_total += 1
            if gt_root[b, j]:
                root_total += 1
                if pred_root[b, j]:
                    root_correct += 1
                continue
            nonroot_total += 1
            if pred_parent_idx is not None:
                cand = pred_parent_idx[b, j, pred_parent[b, j]]
                if int(cand.item()) == int(gt_parent[b, j].item()):
                    parent_correct += 1
                    nonroot_correct += 1
            elif int(pred_parent[b, j].item()) == int(gt_parent[b, j].item()):
                parent_correct += 1
                nonroot_correct += 1

        if decode_cfg.two_stage_root_parent and pred.get("root_logits") is not None and pred.get("candidate_pair_logits") is not None:
            decoded = decode_two_stage_root_parent_graph(
                positions=batch["joint_pos"][b].detach().cpu().numpy(),
                active_prob=batch["joint_active"][b].detach().cpu().numpy(),
                root_logits=pred["root_logits"][b].detach().cpu().numpy(),
                parent_candidate_logits=pred["candidate_pair_logits"][b].detach().cpu().numpy(),
                candidate_indices=pred["candidate_indices"][b].detach().cpu().numpy(),
                candidate_mask=pred["candidate_mask"][b].detach().cpu().numpy(),
                config=decode_cfg,
                root_threshold=decode_cfg.root_threshold,
            )
            if mask_root_for_gt_nonroot:
                gt_root_b = gt_root[b]
                parent_ptr = decoded["parent_ptr"].copy()
                cand_idx_b = pred["candidate_indices"][b]
                cand_mask_b = pred["candidate_mask"][b]
                cand_logits_b = pred["candidate_pair_logits"][b]
                for child in torch.where(valid_b)[0].tolist():
                    if bool(gt_root_b[child].item()):
                        continue
                    if int(parent_ptr[child]) >= 0:
                        continue
                    valid_slots = torch.where(cand_mask_b[child])[0]
                    if valid_slots.numel() == 0:
                        continue
                    slot_scores = cand_logits_b[child, valid_slots]
                    best_local = int(valid_slots[int(torch.argmax(slot_scores).item())].item())
                    parent_ptr[child] = int(cand_idx_b[child, best_local].item())
                decoded["parent_ptr"] = parent_ptr
        else:
            decoded = decode_parent_graph(
                positions=batch["joint_pos"][b].detach().cpu().numpy(),
                active_prob=batch["joint_active"][b].detach().cpu().numpy(),
                parent_logits=parent_logits[b].detach().cpu().numpy(),
                parent_offset=pred["parent_offset"][b].detach().cpu().numpy(),
                edge_confidence=pred["edge_confidence"][b].detach().cpu().numpy(),
                config=decode_cfg,
            )
        component_counts.append(float(decoded["metadata"]["component_count"]))

        pred_edges = parent_edge_sets(torch.from_numpy(decoded["parent_ptr"]), torch.from_numpy(decoded["active_mask"]))
        gt_edges = set()
        gt_nonroot_edges = set()
        valid_ids = torch.where(valid_b)[0]
        for child in valid_ids:
            child_i = int(child.item())
            if gt_root[b, child_i]:
                continue
            parent_i = int(gt_parent[b, child_i].item())
            if parent_i >= 0 and parent_i != child_i and bool(valid_b[parent_i].item()):
                a, c = sorted((parent_i, child_i))
                gt_edges.add((a, c))
                gt_nonroot_edges.add((a, c))
        edge_f1_values.append(edge_metrics(pred_edges, gt_edges)["f1"])
        nonroot_edge_f1_values.append(edge_metrics(pred_edges, gt_nonroot_edges)["f1"])

        if decode_cfg.two_stage_root_parent and pred.get("candidate_pair_logits") is not None and pred.get("candidate_indices") is not None:
            cand_logits_b = pred["candidate_pair_logits"][b]
            cand_idx_b = pred["candidate_indices"][b]
            for child in valid_ids.tolist():
                if bool(gt_root[b, child].item()):
                    continue
                gt_parent_i = int(gt_parent[b, child].item())
                if gt_parent_i < 0:
                    continue
                present = (cand_idx_b[child] == gt_parent_i).nonzero(as_tuple=True)
                if present[0].numel() == 0:
                    failed_nonroot_count += 1
                    continue
                scores = cand_logits_b[child]
                order = torch.argsort(scores, descending=True)
                gt_slot = int(present[0][0].item())
                rank = int((order == gt_slot).nonzero(as_tuple=True)[0].item()) + 1
                nonroot_ranks.append(rank)
                if rank > 1:
                    failed_nonroot_count += 1

    pred_root_ratio = float((pred_root & valid_mask).float().sum().item() / valid_mask.float().sum().clamp(min=1.0).item())
    gt_root_ratio = float((gt_root & valid_mask).float().sum().item() / valid_mask.float().sum().clamp(min=1.0).item())

    # Exclude geometrically-unresolvable coincident nodes from root metrics.
    root_eval_mask = valid_mask.clone()
    ambiguous_root_nodes = 0
    if root_ambiguity_eps > 0.0:
        ambiguous = compute_ambiguous_root_mask(batch, eps=root_ambiguity_eps)
        ambiguous_root_nodes = int((ambiguous & valid_mask).float().sum().item())
        root_eval_mask = valid_mask & ~ambiguous

    # Root precision/recall/F1 and logit margin diagnostics.
    root_tp = int((pred_root & gt_root & root_eval_mask).float().sum().item())
    root_fp = int((pred_root & ~gt_root & root_eval_mask).float().sum().item())
    root_fn = int((~pred_root & gt_root & root_eval_mask).float().sum().item())
    root_precision = root_tp / max(root_tp + root_fp, 1)
    root_recall = root_tp / max(root_tp + root_fn, 1)
    root_f1 = 2.0 * root_precision * root_recall / max(root_precision + root_recall, 1e-8)

    # Root logit margin: mean(root_logit | GT root) - mean(root_logit | GT non-root).
    root_logit_margin = 0.0
    root_logit_mean_gt = 0.0
    root_logit_mean_nongt = 0.0
    if "root_logits" in pred:
        rl = pred["root_logits"]  # [B, J]
        gt_root_mask_f = (gt_root & valid_mask).float()
        nonroot_mask_f = (~gt_root & valid_mask).float()
        if gt_root_mask_f.sum() > 0:
            root_logit_mean_gt = float((rl * gt_root_mask_f).sum() / gt_root_mask_f.sum())
        if nonroot_mask_f.sum() > 0:
            root_logit_mean_nongt = float((rl * nonroot_mask_f).sum() / nonroot_mask_f.sum())
        root_logit_margin = root_logit_mean_gt - root_logit_mean_nongt
    elif "parent_logits" in pred:
        root_class = pred["parent_logits"].shape[-1] - 1
        rl = pred["parent_logits"][:, :, root_class]
        gt_root_mask_f = (gt_root & valid_mask).float()
        nonroot_mask_f = (~gt_root & valid_mask).float()
        if gt_root_mask_f.sum() > 0:
            root_logit_mean_gt = float((rl * gt_root_mask_f).sum() / gt_root_mask_f.sum())
        if nonroot_mask_f.sum() > 0:
            root_logit_mean_nongt = float((rl * nonroot_mask_f).sum() / nonroot_mask_f.sum())
        root_logit_margin = root_logit_mean_gt - root_logit_mean_nongt

    return {
        "parent_acc": float(parent_correct / max(parent_total, 1)),
        "nonroot_parent_acc": float(nonroot_correct / max(nonroot_total, 1)),
        "root_acc": float(root_correct / max(root_total, 1)),
        "root_precision": float(root_precision),
        "root_recall": float(root_recall),
        "root_f1": float(root_f1),
        "root_false_positives": int(root_fp),
        "root_false_negatives": int(root_fn),
        "ambiguous_root_nodes": int(ambiguous_root_nodes),
        "root_logit_mean_gt": float(root_logit_mean_gt),
        "root_logit_mean_nongt": float(root_logit_mean_nongt),
        "root_logit_margin": float(root_logit_margin),
        "pred_root_ratio": pred_root_ratio,
        "gt_root_ratio": gt_root_ratio,
        "nonroot_edge_f1": float(sum(nonroot_edge_f1_values) / max(len(nonroot_edge_f1_values), 1)),
        "gt_parent_rank_mean": float(sum(nonroot_ranks) / max(len(nonroot_ranks), 1)) if nonroot_ranks else 0.0,
        "gt_parent_rank_p90": float(torch.tensor(nonroot_ranks, dtype=torch.float32).quantile(0.9).item()) if nonroot_ranks else 0.0,
        "failed_nonroot_count": int(failed_nonroot_count),
        "edge_f1": float(sum(edge_f1_values) / max(len(edge_f1_values), 1)),
        "component_count": float(sum(component_counts) / max(len(component_counts), 1)),
    }


@torch.no_grad()
def sweep_root_thresholds(
    pred: dict,
    batch: dict,
    decode_cfg: ParentDecodeConfig,
    thresholds: list[float] | None = None,
    mask_root_for_gt_nonroot: bool = False,
    root_ambiguity_eps: float = 0.0,
) -> dict:
    """Sweep root decision thresholds and report metrics for each."""
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    thresholds = sorted({round(float(t), 6) for t in thresholds if 0.0 < float(t) < 1.0})
    
    parent_logits = pred["parent_logits"]
    root_class = parent_logits.shape[-1] - 1
    B, J, _ = parent_logits.shape
    
    valid_mask = batch["joint_active"] > 0.5
    gt_parent = batch["parent_index"].long()
    gt_root = batch["root_mask"] > 0.5

    # Exclude geometrically-unresolvable coincident nodes from root metrics.
    root_eval_mask = valid_mask.clone()
    ambiguous_root_nodes = 0
    if root_ambiguity_eps > 0.0:
        ambiguous = compute_ambiguous_root_mask(batch, eps=root_ambiguity_eps)
        ambiguous_root_nodes = int((ambiguous & valid_mask).float().sum().item())
        root_eval_mask = valid_mask & ~ambiguous
    
    # Extract root logits or probabilities
    if "root_logits" in pred:
        root_logits = pred["root_logits"]
    else:
        root_logits = parent_logits[:, :, root_class]
    
    root_prob = torch.sigmoid(root_logits)
    gt_root_ratio = float((gt_root & root_eval_mask).float().sum().item() / root_eval_mask.float().sum().clamp(min=1.0).item())
    
    results = {}
    best_threshold = 0.5
    best_score = -1.0
    
    for threshold in thresholds:
        pred_root = root_prob > threshold
        
        root_tp = int((pred_root & gt_root & root_eval_mask).float().sum().item())
        root_fp = int((pred_root & ~gt_root & root_eval_mask).float().sum().item())
        root_fn = int((~pred_root & gt_root & root_eval_mask).float().sum().item())
        root_precision = root_tp / max(root_tp + root_fp, 1)
        root_recall = root_tp / max(root_tp + root_fn, 1)
        root_f1 = 2.0 * root_precision * root_recall / max(root_precision + root_recall, 1e-8)
        
        pred_root_ratio = float((pred_root & root_eval_mask).float().sum().item() / root_eval_mask.float().sum().clamp(min=1.0).item())
        ratio_diff = abs(pred_root_ratio - gt_root_ratio)
        
        # Score: F1 minus ratio difference
        score = root_f1 - ratio_diff
        if score > best_score:
            best_score = score
            best_threshold = threshold
        
        results[threshold] = {
            "root_precision": float(root_precision),
            "root_recall": float(root_recall),
            "root_f1": float(root_f1),
            "root_fp": int(root_fp),
            "root_fn": int(root_fn),
            "pred_root_ratio": pred_root_ratio,
            "gt_root_ratio": gt_root_ratio,
            "ratio_diff": float(ratio_diff),
            "score": float(score),
        }
    
    return {
        "all_results": results,
        "best_threshold": float(best_threshold),
        "best_score": float(best_score),
        "best_metrics": results[best_threshold],
        "ambiguous_root_nodes": int(ambiguous_root_nodes),
    }


@torch.no_grad()
def failed_child_diagnostics(pred: dict, batch: dict, topk: int = 5) -> dict:
    parent_logits = pred["parent_logits"]
    B, J, K = parent_logits.shape
    root_class = K - 1
    gt_parent = batch["parent_index"].long()
    gt_root = batch["root_mask"] > 0.5
    active = batch["joint_active"] > 0.5
    joint_pos = batch["joint_pos"]

    failures = []
    gt_ranks = []
    dist_ranks = []

    for b in range(B):
        logits_b = parent_logits[b]
        for child in torch.where(active[b])[0]:
            child_i = int(child.item())
            if gt_root[b, child_i]:
                gt_cls = root_class
            else:
                gt_cls = int(gt_parent[b, child_i].item())
            if gt_cls < 0 or gt_cls >= K:
                gt_cls = root_class

            scores = logits_b[child_i]
            pred_cls = int(torch.argmax(scores).item())
            if pred_cls == gt_cls:
                continue

            order = torch.argsort(scores, descending=True)
            gt_rank = int((order == gt_cls).nonzero(as_tuple=True)[0].item()) + 1
            gt_ranks.append(gt_rank)

            if gt_cls != root_class and gt_cls < J:
                d = torch.norm(joint_pos[b] - joint_pos[b, child_i].unsqueeze(0), dim=-1)
                d_masked = d.clone()
                d_masked[~active[b]] = 1e9
                d_masked[child_i] = 1e9
                dist_order = torch.argsort(d_masked, descending=False)
                dist_rank = int((dist_order == gt_cls).nonzero(as_tuple=True)[0].item()) + 1
            else:
                dist_rank = -1
            dist_ranks.append(dist_rank)

            top = order[:topk].tolist()
            failures.append(
                {
                    "batch": b,
                    "child": child_i,
                    "predicted_topk": top,
                    "predicted_topk_scores": [float(scores[t].item()) for t in top],
                    "gt_parent": int(gt_cls),
                    "gt_parent_rank": int(gt_rank),
                    "parent_distance_rank": int(dist_rank),
                    "gt_parent_active": bool(gt_cls < J and active[b, gt_cls].item()) if gt_cls != root_class else True,
                    "gt_parent_masked_out": bool(gt_cls < J and (not active[b, gt_cls].item())) if gt_cls != root_class else False,
                }
            )

    return {
        "avg_gt_parent_rank": float(sum(gt_ranks) / max(len(gt_ranks), 1)) if gt_ranks else 0.0,
        "avg_parent_distance_rank": float(sum([r for r in dist_ranks if r > 0]) / max(len([r for r in dist_ranks if r > 0]), 1)) if dist_ranks else 0.0,
        "failed_child_count": len(failures),
        "failed_children": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit parent head on one fixed batch with GT nodes")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--out", default="outputs/models/hyperbone_anymate_static_v2.8c_parent_overfit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--backbone", choices=["pointnet", "dgcnn"], default="dgcnn")
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher-force-gt-nodes", action="store_true")
    parser.add_argument("--parent-head", choices=["slot", "pairwise"], default="slot")
    parser.add_argument("--root-feature-mode", choices=["none", "structural"], default="none")
    parser.add_argument("--no-backbone-for-gt-nodes", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-nonroot-loss", action="store_true")
    parser.add_argument("--use-offset-loss", action="store_true")
    parser.add_argument("--use-root-loss", action="store_true")
    parser.add_argument("--root-class-weight", type=float, default=0.25)
    parser.add_argument("--parent-root-weight", type=float, default=0.2)
    parser.add_argument("--root-loss-weight", type=float, default=0.0,
                        help="Weight for dedicated root BCE loss term.")
    parser.add_argument("--root-ratio-weight", type=float, default=0.0,
                        help="Weight for root-ratio calibration loss term.")
    parser.add_argument("--root-bias-init", type=float, default=-3.47,
                        help="Bias initialization for root logit final layer.")
    parser.add_argument("--candidate-parent-k", type=int, default=None)
    parser.add_argument("--force-gt-parent-candidate", action="store_true")
    parser.add_argument("--enforce-root-per-component", action="store_true")
    parser.add_argument("--two-stage-root-parent", action="store_true")
    parser.add_argument("--two-stage-parent-margin-weight", type=float, default=0.0)
    parser.add_argument("--two-stage-parent-margin-delta", type=float, default=0.5)
    parser.add_argument("--root-threshold", type=float, default=None,
                        help="Calibrated root decision threshold used at decode/eval time "
                             "(two-stage). Defaults to active_threshold (0.5) when unset.")
    parser.add_argument("--mask-root-for-gt-nonroot", action="store_true")
    parser.add_argument("--root-only-overfit", action="store_true")
    parser.add_argument("--root-focal-gamma", type=float, default=0.0)
    parser.add_argument("--root-pos-weight-cap", type=float, default=8.0)
    parser.add_argument("--root-overpredict-weight", type=float, default=0.0)
    parser.add_argument("--sweep-root-threshold", action="store_true",
                        help="Sweep root probability thresholds and report metrics for each.")
    parser.add_argument("--sweep-root-threshold-min", type=float, default=0.05,
                        help="Minimum threshold for root sweep.")
    parser.add_argument("--sweep-root-threshold-max", type=float, default=0.95,
                        help="Maximum threshold for root sweep.")
    parser.add_argument("--sweep-root-threshold-step", type=float, default=0.05,
                        help="Step size for root threshold sweep.")
    parser.add_argument("--root-fp-margin-weight", type=float, default=0.0,
                        help="Weight for false-positive root margin loss on GT non-root nodes.")
    parser.add_argument("--root-fp-margin", type=float, default=-2.0,
                        help="Margin threshold for non-root root logits (penalize if root_logit > margin).")
    parser.add_argument("--root-ambiguity-eps", type=float, default=0.0,
                        help="Exclude conflicting root/non-root nodes that share a position within this "
                             "distance from root metrics (they are geometrically unresolvable). 0 disables.")
    args = parser.parse_args()

    if args.mask_root_for_gt_nonroot and (not args.teacher_force_gt_nodes or not args.two_stage_root_parent):
        raise ValueError("--mask-root-for-gt-nonroot is only valid with --teacher-force-gt-nodes and --two-stage-root-parent")
    if args.root_only_overfit and not args.teacher_force_gt_nodes:
        raise ValueError("--root-only-overfit requires --teacher-force-gt-nodes")
    if args.sweep_root_threshold_step <= 0.0:
        raise ValueError("--sweep-root-threshold-step must be > 0")
    if args.sweep_root_threshold_min <= 0.0 or args.sweep_root_threshold_max >= 1.0 or args.sweep_root_threshold_min >= args.sweep_root_threshold_max:
        raise ValueError("sweep threshold bounds must satisfy 0 < min < max < 1")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = AnymateStaticRigDataset(
        args.pt,
        f"{args.splits_dir}/train.jsonl",
        max_joints=args.max_nodes,
        pc_points=args.points_per_sample,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # For pairwise head, override root bias init from CLI.
    if args.parent_head == "pairwise":
        from hyperbone.models.hyperbone_static_parent import PairwiseParentHead
    model = HyperBoneStaticParentModel(
        in_channels=3,
        feat_dim=512,
        max_joints=args.max_nodes,
        predict_skinning=False,
        backbone=args.backbone,
        knn_k=args.knn_k,
        parent_head=args.parent_head,
        root_bias_init=args.root_bias_init if args.parent_head == "pairwise" else None,
        root_feature_mode=args.root_feature_mode,
    ).to(device)

    for p in model.parameters():
        p.requires_grad = False
    if args.root_only_overfit and args.parent_head == "pairwise":
        for name, p in model.parent_head.named_parameters():
            if name.startswith("root_mlp") or name.startswith("root_logit"):
                p.requires_grad = True
        # The root head consumes node features from gt_pos_encoder when GT nodes
        # bypass the backbone. If that encoder stays frozen, the root classifier
        # only sees a fixed positional embedding and cannot separate roots from
        # non-roots that collide near the decision boundary. Train it too so the
        # one-batch sanity overfit can actually memorize the root labels.
        if args.no_backbone_for_gt_nodes:
            for p in model.gt_pos_encoder.parameters():
                p.requires_grad = True
    else:
        for p in model.parent_head.parameters():
            p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and args.amp) else None

    decode_cfg = ParentDecodeConfig(
        decode_mode="parent_argmax_acyclic",
        active_threshold=0.5,
        max_degree=8,
        enforce_root_per_component=args.enforce_root_per_component,
        two_stage_root_parent=args.two_stage_root_parent,
        root_threshold=args.root_threshold,
    )

    history = []
    print(f"[Overfit-parent] Device: {device}")
    print(f"[Overfit-parent] Batch size: {args.batch_size} | Steps: {args.steps}")
    if args.mask_root_for_gt_nonroot:
        print("[Overfit-parent] DIAGNOSTIC ONLY: uses GT root/non-root labels at decode time.")
    if args.root_only_overfit:
        print("[Overfit-parent] Root-only mode: training only root classifier losses.")

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
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
            )
            if args.two_stage_root_parent:
                candidate_info = build_parent_candidates(
                    batch["joint_pos"],
                    batch["joint_active"] > 0.5,
                    batch["parent_index"].long(),
                    k=args.candidate_parent_k or 8,
                    include_root=False,
                    force_gt_parent=args.force_gt_parent_candidate,
                )
                supervision["two_stage_root_parent"] = True
                supervision["candidate_indices"] = candidate_info["candidate_indices"]
                supervision["candidate_mask"] = candidate_info["candidate_mask"]
                supervision["candidate_targets"] = candidate_info["target_candidate_class"]
                supervision["candidate_valid_mask"] = candidate_info["gt_parent_in_candidates"]
                supervision["candidate_pair_logits"] = gather_nonroot_candidate_logits(
                    pred["parent_pair_logits"],
                    supervision["candidate_indices"],
                    supervision["candidate_mask"],
                )
                pred_eval = dict(pred)
                pred_eval["candidate_indices"] = supervision["candidate_indices"]
                pred_eval["candidate_mask"] = supervision["candidate_mask"]
                pred_eval["candidate_pair_logits"] = supervision["candidate_pair_logits"]
            elif "candidate_indices" in supervision:
                supervision["candidate_logits"] = gather_parent_candidate_logits(
                    pred["parent_logits"],
                    supervision["candidate_indices"],
                    supervision["candidate_mask"],
                )
                pred_eval = dict(pred)
                pred_eval["parent_logits"] = mask_parent_logits_with_candidates(
                    pred["parent_logits"],
                    supervision["candidate_indices"],
                    supervision["candidate_mask"],
                )
            else:
                pred_eval = pred
            supervision["use_nonroot_loss"] = args.use_nonroot_loss
            supervision["use_offset_loss"] = args.use_offset_loss
            supervision["use_root_loss"] = args.use_root_loss
            loss, loss_stats = compute_parent_losses(
                pred,
                supervision,
                root_class_weight=args.root_class_weight,
                parent_root_weight=args.parent_root_weight,
                root_loss_weight=args.root_loss_weight,
                root_ratio_weight=args.root_ratio_weight,
                two_stage_parent_margin_weight=args.two_stage_parent_margin_weight,
                two_stage_parent_margin_delta=args.two_stage_parent_margin_delta,
                root_only_overfit=args.root_only_overfit,
                root_focal_gamma=args.root_focal_gamma,
                root_pos_weight_cap=args.root_pos_weight_cap,
                root_overpredict_weight=args.root_overpredict_weight,
                root_fp_margin_weight=args.root_fp_margin_weight,
                root_fp_margin=args.root_fp_margin,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

        if step % 50 == 0 or step == 1 or step == args.steps:
            metrics = evaluate_batch(
                pred_eval,
                batch,
                decode_cfg,
                mask_root_for_gt_nonroot=args.mask_root_for_gt_nonroot,
                root_ambiguity_eps=args.root_ambiguity_eps,
            )
            row = {
                "step": step,
                "loss": float(loss.detach().item()),
                **loss_stats,
                **metrics,
            }
            history.append(row)
            print(
                f"S{step:04d} | loss={row['loss']:.4f} ce={row['parent_ce']:.4f} "
                f"acc={row['parent_acc']:.3f} nr_acc={row['nonroot_parent_acc']:.3f} "
                f"root_acc={row['root_acc']:.3f} root_f1={row.get('root_f1', 0.0):.3f} "
                f"root_margin={row.get('root_logit_margin', 0.0):.2f} "
                f"edge_f1={row['edge_f1']:.3f} root_pred={row['pred_root_ratio']:.3f}/{row['gt_root_ratio']:.3f} "
                f"comp={row['component_count']:.2f}"
            )

    (out_dir / "overfit_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    final = history[-1] if history else {}
    
    # Threshold sweep if requested
    threshold_sweep_results = None
    if args.sweep_root_threshold and (args.root_only_overfit or args.two_stage_root_parent):
        print("[Overfit-parent] Running root threshold sweep...")
        sweep_thresholds = []
        current = args.sweep_root_threshold_min
        while current <= args.sweep_root_threshold_max + 1e-9:
            sweep_thresholds.append(round(current, 6))
            current += args.sweep_root_threshold_step
        threshold_sweep_results = sweep_root_thresholds(
            pred_eval,
            batch,
            decode_cfg,
            thresholds=sweep_thresholds,
            mask_root_for_gt_nonroot=args.mask_root_for_gt_nonroot,
            root_ambiguity_eps=args.root_ambiguity_eps,
        )
        print("[Overfit-parent] Threshold sweep results:")
        sweep_table = []
        for threshold in sorted(threshold_sweep_results["all_results"].keys()):
            metrics = threshold_sweep_results["all_results"][threshold]
            sweep_table.append({
                "threshold": threshold,
                "f1": metrics["root_f1"],
                "precision": metrics["root_precision"],
                "recall": metrics["root_recall"],
                "pred_ratio": metrics["pred_root_ratio"],
                "ratio_diff": metrics["ratio_diff"],
                "fp": metrics["root_fp"],
                "fn": metrics["root_fn"],
                "score": metrics["score"],
            })
        print(json.dumps(sweep_table, indent=2))
        print(f"[Overfit-parent] Best threshold: {threshold_sweep_results['best_threshold']}")
        print(f"[Overfit-parent] Best metrics: {json.dumps(threshold_sweep_results['best_metrics'], indent=2)}")
    
    nonroot_gate = 0.95 if args.mask_root_for_gt_nonroot else 0.90
    edge_gate = 0.95 if args.mask_root_for_gt_nonroot else 0.90
    if args.root_only_overfit:
        ratio_close = abs(final.get("pred_root_ratio", 0.0) - final.get("gt_root_ratio", 0.0)) <= max(0.005, 0.25 * final.get("gt_root_ratio", 0.0))
        passed = final.get("root_f1", 0.0) > 0.95 and ratio_close
        if threshold_sweep_results and threshold_sweep_results["best_metrics"]["root_f1"] > final.get("root_f1", 0.0):
            print(f"[Overfit-parent] Threshold sweep improved root_f1 to {threshold_sweep_results['best_metrics']['root_f1']:.4f}")
            best_threshold_metrics = threshold_sweep_results["best_metrics"]
            ratio_close_threshold = abs(best_threshold_metrics["pred_root_ratio"] - best_threshold_metrics["gt_root_ratio"]) <= max(0.005, 0.25 * best_threshold_metrics["gt_root_ratio"])
            passed = best_threshold_metrics["root_f1"] > 0.95 and ratio_close_threshold
    elif args.two_stage_root_parent:
        # v2.13 calibrated two-stage gate: strict 0.95 on parent metrics plus
        # ambiguity-excluded root_f1 at the calibrated root threshold.
        nonroot_gate = 0.95
        edge_gate = 0.95
        passed = (
            final.get("parent_acc", 0.0) > 0.95
            and final.get("nonroot_parent_acc", 0.0) > 0.95
            and final.get("edge_f1", 0.0) > 0.95
            and final.get("root_f1", 0.0) > 0.95
        )
    else:
        passed = (
            final.get("parent_acc", 0.0) > 0.95
            and final.get("nonroot_parent_acc", 0.0) > nonroot_gate
            and final.get("edge_f1", 0.0) > edge_gate
        )
    summary = {
        "pass": bool(passed),
        "criteria": {
            "parent_acc_gt": 0.95,
            "nonroot_parent_acc_gt": nonroot_gate,
            "edge_f1_gt": edge_gate,
            "root_f1_gt": 0.95 if (args.root_only_overfit or args.two_stage_root_parent) else None,
            "root_ratio_close": True if args.root_only_overfit else None,
        },
        "final": final,
        "threshold_sweep": threshold_sweep_results,
    }
    if not passed:
        diag = failed_child_diagnostics(pred, batch, topk=5)
        summary["failure_diagnostics"] = diag
        print("[Overfit-parent] failure diagnostics")
        print(json.dumps(diag, indent=2))
    (out_dir / "overfit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[Overfit-parent] DONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
