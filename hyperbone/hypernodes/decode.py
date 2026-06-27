"""
Decoder: convert HyperNodeNet predictions into GraphLabel-like structures.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hyperbone.labels.schema import (
    GraphLabel,
    HyperNodeLabel,
    HyperEdgeLabel,
    NodeType,
    EdgeType,
)
from hyperbone.hypernodes.dataset import (
    NODE_TYPES,
    EDGE_TYPES,
    NODE_TYPE_TO_IDX,
    EDGE_TYPE_TO_IDX,
)


@dataclass
class DecodeConfig:
    active_threshold: float = 0.5
    confidence_threshold: float = 0.3
    edge_threshold: float = 0.5
    nms_radius: float = 0.03  # node dedup radius (normalized coords)
    min_nodes: int = 2
    max_nodes: int = 64


def decode_predictions(
    pred: dict[str, torch.Tensor],
    config: DecodeConfig | None = None,
    sample_ids: list[str] | None = None,
) -> list[GraphLabel]:
    """Decode a batch of predictions into GraphLabel list.
    
    Args:
        pred: model output dict with keys active_logits, node_xy, node_xyz,
              node_type_logits, node_confidence, edge_logits, edge_type_logits
        config: decode thresholds
        sample_ids: optional sample IDs for output graphs
    """
    if config is None:
        config = DecodeConfig()

    B = pred["active_logits"].shape[0]
    graphs = []

    for b in range(B):
        active_prob = torch.sigmoid(pred["active_logits"][b])  # [N]
        node_xy = pred["node_xy"][b]  # [N, 2]
        node_xyz = pred["node_xyz"][b]  # [N, 3]
        node_type_logits = pred["node_type_logits"][b]  # [N, T]
        node_conf = pred["node_confidence"][b]  # [N]
        edge_logits = pred["edge_logits"][b]  # [N, N]
        edge_type_logits = pred["edge_type_logits"][b]  # [N, N, E]

        # Select active nodes
        active_mask = active_prob > config.active_threshold
        active_indices = torch.where(active_mask)[0]

        # Filter by confidence
        keep = []
        for idx in active_indices:
            i = idx.item()
            if node_conf[i].item() >= config.confidence_threshold:
                keep.append(i)

        # NMS: remove duplicate nodes within radius
        final_nodes = []
        used = set()
        for i in keep:
            if i in used:
                continue
            xy_i = node_xy[i].detach().cpu().numpy()
            # Check against already-kept nodes
            is_dup = False
            for j in final_nodes:
                xy_j = node_xy[j].detach().cpu().numpy()
                dist = np.linalg.norm(xy_i - xy_j)
                if dist < config.nms_radius:
                    is_dup = True
                    break
            if not is_dup:
                final_nodes.append(i)
                used.add(i)

        # Build graph
        if len(final_nodes) < config.min_nodes:
            # Still create graph but mark as invalid
            graph = GraphLabel(
                sample_id=sample_ids[b] if sample_ids else f"pred_{b}",
                metadata={"valid": False, "reason": "too_few_nodes"},
            )
            graphs.append(graph)
            continue

        # Create nodes
        nodes = []
        slot_to_node_id = {}
        for node_id, slot_idx in enumerate(final_nodes):
            xy = node_xy[slot_idx].detach().cpu().numpy().tolist()
            xyz = node_xyz[slot_idx].detach().cpu().numpy().tolist()
            type_idx = node_type_logits[slot_idx].argmax().item()
            ntype = NODE_TYPES[type_idx] if type_idx < len(NODE_TYPES) else NodeType.UNKNOWN
            conf = node_conf[slot_idx].item()

            node = HyperNodeLabel(
                id=node_id,
                node_type=ntype,
                xy=xy,
                xyz=xyz,
                confidence=conf,
            )
            nodes.append(node)
            slot_to_node_id[slot_idx] = node_id

        # Create edges
        edges = []
        edge_id = 0
        edge_prob = torch.sigmoid(edge_logits)
        for i_idx, si in enumerate(final_nodes):
            for j_idx, sj in enumerate(final_nodes):
                if j_idx <= i_idx:
                    continue
                if edge_prob[si, sj].item() > config.edge_threshold:
                    etype_idx = edge_type_logits[si, sj].argmax().item()
                    etype = EDGE_TYPES[etype_idx] if etype_idx < len(EDGE_TYPES) else EdgeType.UNKNOWN
                    edge = HyperEdgeLabel(
                        id=edge_id,
                        source_node_id=slot_to_node_id[si],
                        target_node_id=slot_to_node_id[sj],
                        edge_type=etype,
                        confidence=edge_prob[si, sj].item(),
                    )
                    edges.append(edge)
                    edge_id += 1

        graph = GraphLabel(
            sample_id=sample_ids[b] if sample_ids else f"pred_{b}",
            nodes=nodes,
            edges=edges,
            metadata={"valid": True},
        )
        graphs.append(graph)

    return graphs


def save_predictions_jsonl(graphs: list[GraphLabel], path: str | Path) -> None:
    """Save decoded predictions as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for g in graphs:
            f.write(g.to_jsonl_line() + "\n")
