from __future__ import annotations

from typing import Dict

import torch


ROOT_PARENT_INDEX = -1


def build_parent_candidates(
    joint_pos: torch.Tensor,
    active_mask: torch.Tensor,
    parent_index: torch.Tensor,
    k: int = 8,
    include_root: bool = True,
    force_gt_parent: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build k-nearest parent candidates plus optional ROOT for each child slot."""
    if joint_pos.ndim != 3:
        raise ValueError(f"joint_pos must be [B,N,3], got {tuple(joint_pos.shape)}")

    device = joint_pos.device
    dtype = joint_pos.dtype
    active_mask = active_mask.bool()
    parent_index = parent_index.long()

    batch_size, max_nodes, _ = joint_pos.shape
    nonroot_slots = max(int(k), 0)
    total_slots = nonroot_slots + (1 if include_root else 0)

    candidate_indices = torch.full((batch_size, max_nodes, total_slots), ROOT_PARENT_INDEX, dtype=torch.long, device=device)
    candidate_mask = torch.zeros((batch_size, max_nodes, total_slots), dtype=torch.bool, device=device)
    target_candidate_class = torch.full((batch_size, max_nodes), max(total_slots - 1, 0), dtype=torch.long, device=device)
    gt_parent_in_candidates = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
    forced_gt_parent = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)

    for batch_idx in range(batch_size):
        active_nodes = torch.where(active_mask[batch_idx])[0]
        for child_idx in active_nodes.tolist():
            gt_parent = int(parent_index[batch_idx, child_idx].item())
            root_slot = total_slots - 1 if include_root else -1

            if nonroot_slots > 0:
                candidate_pool = active_nodes[active_nodes != child_idx]
                if candidate_pool.numel() > 0:
                    distances = torch.norm(
                        joint_pos[batch_idx, candidate_pool] - joint_pos[batch_idx, child_idx].unsqueeze(0),
                        dim=-1,
                    )
                    order = torch.argsort(distances, descending=False)
                    selected = candidate_pool[order][:nonroot_slots].tolist()
                else:
                    selected = []
            else:
                selected = []

            gt_is_valid = (
                gt_parent >= 0
                and gt_parent < max_nodes
                and gt_parent != child_idx
                and bool(active_mask[batch_idx, gt_parent].item())
            )
            gt_present_raw = gt_is_valid and gt_parent in selected
            gt_parent_in_candidates[batch_idx, child_idx] = gt_present_raw

            if force_gt_parent and gt_is_valid and not gt_present_raw and nonroot_slots > 0:
                forced_gt_parent[batch_idx, child_idx] = True
                if len(selected) < nonroot_slots:
                    selected.append(gt_parent)
                else:
                    selected[-1] = gt_parent
                gt_present_raw = True

            deduped: list[int] = []
            seen = set()
            for candidate in selected:
                if candidate == child_idx:
                    continue
                if candidate < 0 or candidate >= max_nodes:
                    continue
                if not bool(active_mask[batch_idx, candidate].item()):
                    continue
                if candidate in seen:
                    continue
                seen.add(candidate)
                deduped.append(int(candidate))

            if deduped:
                candidate_indices[batch_idx, child_idx, : len(deduped)] = torch.tensor(deduped, dtype=torch.long, device=device)
                candidate_mask[batch_idx, child_idx, : len(deduped)] = True

            if include_root:
                candidate_mask[batch_idx, child_idx, root_slot] = True
                candidate_indices[batch_idx, child_idx, root_slot] = ROOT_PARENT_INDEX

            if gt_is_valid:
                try:
                    target_candidate_class[batch_idx, child_idx] = deduped.index(gt_parent)
                except ValueError:
                    target_candidate_class[batch_idx, child_idx] = root_slot if include_root else 0
            else:
                target_candidate_class[batch_idx, child_idx] = root_slot if include_root else 0

            if gt_present_raw:
                gt_parent_in_candidates[batch_idx, child_idx] = True

    return {
        "candidate_indices": candidate_indices,
        "candidate_mask": candidate_mask,
        "target_candidate_class": target_candidate_class,
        "gt_parent_in_candidates": gt_parent_in_candidates,
        "forced_gt_parent": forced_gt_parent,
    }


def gather_parent_candidate_logits(
    parent_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather logits for the restricted candidate set, preserving ROOT as the final slot."""
    if parent_logits.ndim != 3:
        raise ValueError(f"parent_logits must be [B,N,N+1], got {tuple(parent_logits.shape)}")

    root_class = parent_logits.shape[-1] - 1
    candidate_logits = torch.full(
        candidate_indices.shape,
        -1e4,
        dtype=parent_logits.dtype,
        device=parent_logits.device,
    )

    if candidate_indices.shape[-1] == 0:
        return candidate_logits

    if candidate_indices.shape[-1] > 1:
        nonroot_idx = candidate_indices[..., :-1].clamp(min=0, max=root_class - 1)
        nonroot_scores = parent_logits[..., :root_class].gather(-1, nonroot_idx)
        candidate_logits[..., :-1] = nonroot_scores

    candidate_logits[..., -1] = parent_logits[..., root_class]
    candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1e4)
    return candidate_logits


def gather_nonroot_candidate_logits(
    pair_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather logits for non-root parent candidates only."""
    if pair_logits.ndim != 3:
        raise ValueError(f"pair_logits must be [B,N,N], got {tuple(pair_logits.shape)}")

    candidate_logits = torch.full(
        candidate_indices.shape,
        -1e4,
        dtype=pair_logits.dtype,
        device=pair_logits.device,
    )
    if candidate_indices.shape[-1] == 0:
        return candidate_logits

    safe_idx = candidate_indices.clamp(min=0, max=pair_logits.shape[-1] - 1)
    gathered = pair_logits.gather(-1, safe_idx)
    candidate_logits.copy_(gathered)
    candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1e4)
    return candidate_logits


def mask_parent_logits_with_candidates(
    parent_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask the full parent logit tensor so decode uses only the restricted candidates."""
    if parent_logits.ndim != 3:
        raise ValueError(f"parent_logits must be [B,N,N+1], got {tuple(parent_logits.shape)}")

    root_class = parent_logits.shape[-1] - 1
    masked = torch.full_like(parent_logits, -1e4)
    masked[..., root_class] = parent_logits[..., root_class]

    if candidate_indices.shape[-1] > 1:
        for slot in range(candidate_indices.shape[-1] - 1):
            slot_idx = candidate_indices[..., slot]
            valid = candidate_mask[..., slot] & (slot_idx >= 0) & (slot_idx < root_class)
            if not valid.any():
                continue
            batch_idx, node_idx = torch.where(valid)
            parent_idx = slot_idx[valid]
            masked[batch_idx, node_idx, parent_idx] = parent_logits[batch_idx, node_idx, parent_idx]

    return masked