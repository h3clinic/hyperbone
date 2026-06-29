"""Augment geometry student cache with v4.1 teacher edge labels.

For each sample and candidate edge, adds:
  - teacher_selected: 1 if v4.1 selected this edge, 0 otherwise
  - teacher_score: v4.1 composite score for this edge

Teacher uses skinning as part of scoring (training supervision only).
Student input tensors remain geometry-only.

Usage:
    python scripts/augment_cache_with_teacher.py \
        --cache outputs/models/hyperbone_track_b_student/cache_train.pt \
        --pt datasets/anymate/Anymate_test.pt \
        --splits-dir outputs/anymate_local_dev/splits \
        --split train \
        --ckpt outputs/models/hyperbone_anymate_static_v2.16_topology_full/best_model.pt \
        --out outputs/models/hyperbone_track_b_student/cache_train_v11.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.datasets.anymate_static_dataset import AnymateStaticRigDataset
from hyperbone.models.hyperbone_static_parent import HyperBoneStaticParentModel
from hyperbone.models.topology_edge_scorer import TopologyEdgeScorer
from hyperbone.rigs.skinning_topology_features import compute_skinning_score_matrices
from hyperbone.rigs.topology_optimizers import (
    hybrid_skinning_cost_optimize,
    _build_hybrid_skinning_candidates,
)
from hyperbone.rigs.undirected_topology import (
    build_undirected_adjacency,
    build_undirected_knn_candidates,
    compute_topology_edge_inputs,
    symmetrize_pair_scores,
)

V41_CONFIG = {
    "candidate_k": 16,
    "neural_weight": 1.0,
    "skin_cosine_weight": 4.0,
    "shared_weight": 4.0,
    "distance_weight": 1.0,
    "degree_penalty": 0.05,
    "long_edge_penalty": 0.25,
    "mutual_bonus": 0.2,
    "max_degree": 4,
}


def load_models(ckpt_path, device, max_nodes=128, feat_dim=512):
    model = HyperBoneStaticParentModel(
        in_channels=3, feat_dim=feat_dim, max_joints=max_nodes,
        predict_skinning=False, backbone="dgcnn", knn_k=16,
        parent_head="pairwise", root_feature_mode="structural",
    ).to(device)
    scorer = TopologyEdgeScorer(
        node_feature_dim=256, edge_feature_dim=23, node_local_dim=8,
        global_context_dim=4, hidden_dim=256, dropout=0.1, num_blocks=3,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
        scorer_state = state.get(
            "topology_scorer_state_dict",
            state.get("edge_refiner_state_dict"),
        )
        if scorer_state is not None:
            scorer.load_state_dict(scorer_state, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    scorer.eval()
    return model, scorer


@torch.no_grad()
def get_teacher_labels(
    dataset, sample_idx, model, scorer, device, student_edge_pairs,
):
    """Run v4.1 on a sample, return teacher labels for student's candidate edges."""
    cfg = V41_CONFIG
    batch = dataset[sample_idx]
    raw_sample = dataset.data[dataset.indices[sample_idx]]

    joint_pos = batch["joint_pos"].unsqueeze(0).to(device)
    active_mask = (batch["joint_active"] > 0.5).unsqueeze(0).to(device)
    adj_matrix = batch["adj_matrix"].unsqueeze(0).to(device)

    pred = model.forward_parent_from_joints(
        joint_pos, active_mask=active_mask, no_backbone_for_gt_nodes=True,
    )
    base_scores = symmetrize_pair_scores(pred["parent_pair_logits"], mode="average")
    node_tokens = pred["node_tokens"]

    gt_adj = build_undirected_adjacency(adj_matrix, active_mask)

    cand = build_undirected_knn_candidates(
        joint_pos, active_mask, k=cfg["candidate_k"],
        gt_adj=gt_adj, force_include_gt=False,
    )
    candidate_mask = cand["candidate_mask"]
    topo_inputs = compute_topology_edge_inputs(
        joint_pos, active_mask, candidate_mask, k=cfg["candidate_k"],
    )
    neural_scores = scorer(
        base_scores, topo_inputs["pair_features"], node_tokens,
        topo_inputs["node_local_features"], topo_inputs["global_context"],
    )

    jp_cpu = joint_pos[0].cpu()
    am_cpu = active_mask[0].cpu()
    cm_cpu = candidate_mask[0].cpu()
    skinning_mats = compute_skinning_score_matrices(jp_cpu, am_cpu, cm_cpu, raw_sample)

    teacher_adj = hybrid_skinning_cost_optimize(
        joint_pos=jp_cpu, active_mask=am_cpu,
        neural_scores=neural_scores[0].cpu(),
        skinning_cosine=skinning_mats["skinning_cosine"],
        max_shared_weight=skinning_mats["max_shared_weight"],
        mode="density_normalized_mst",
        k=cfg["candidate_k"],
        distance_weight=cfg["distance_weight"],
        neural_weight=cfg["neural_weight"],
        skin_cosine_weight=cfg["skin_cosine_weight"],
        shared_weight=cfg["shared_weight"],
        degree_penalty=cfg["degree_penalty"],
        long_edge_penalty=cfg["long_edge_penalty"],
        mutual_bonus=cfg["mutual_bonus"],
        max_degree=cfg["max_degree"],
        candidate_mask=cm_cpu,
    )

    # Build teacher score for student's candidate edges
    # Use the student's candidate mask to get scores
    student_cm = torch.zeros_like(cm_cpu)
    for e_idx in range(student_edge_pairs.shape[0]):
        i, j = int(student_edge_pairs[e_idx, 0]), int(student_edge_pairs[e_idx, 1])
        student_cm[i, j] = True
        student_cm[j, i] = True

    teacher_edges = _build_hybrid_skinning_candidates(
        joint_pos=jp_cpu, active_mask=am_cpu,
        neural_scores=neural_scores[0].cpu(),
        skinning_cosine=skinning_mats["skinning_cosine"],
        max_shared_weight=skinning_mats["max_shared_weight"],
        k=cfg["candidate_k"],
        distance_weight=cfg["distance_weight"],
        neural_weight=cfg["neural_weight"],
        skin_cosine_weight=cfg["skin_cosine_weight"],
        shared_weight=cfg["shared_weight"],
        mutual_bonus=cfg["mutual_bonus"],
        long_edge_penalty=cfg["long_edge_penalty"],
        candidate_mask=student_cm,
    )

    score_map = {}
    for e in teacher_edges:
        key = (min(e.i, e.j), max(e.i, e.j))
        score_map[key] = e.score

    n_edges = student_edge_pairs.shape[0]
    teacher_selected = torch.zeros(n_edges, dtype=torch.float32)
    teacher_score = torch.zeros(n_edges, dtype=torch.float32)

    for e_idx in range(n_edges):
        i, j = int(student_edge_pairs[e_idx, 0]), int(student_edge_pairs[e_idx, 1])
        key = (min(i, j), max(i, j))
        if bool(teacher_adj[i, j].item()):
            teacher_selected[e_idx] = 1.0
        if key in score_map:
            teacher_score[e_idx] = score_map[key]

    return teacher_selected, teacher_score


def main():
    parser = argparse.ArgumentParser(
        description="Augment geometry cache with v4.1 teacher labels")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--pt", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--feat-dim", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Augmenting cache with v4.1 teacher labels", flush=True)
    print(f"Device: {device}", flush=True)

    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    print(f"Loaded cache: {len(cached)} samples", flush=True)

    split_file = f"{args.splits_dir}/{args.split}.jsonl"
    dataset = AnymateStaticRigDataset(args.pt, split_file, max_joints=args.max_nodes)
    print(f"Dataset: {len(dataset)} samples", flush=True)

    model, scorer = load_models(args.ckpt, device, args.max_nodes, args.feat_dim)
    print(f"Loaded v4.1 model from {args.ckpt}", flush=True)

    # Match cache entries to dataset samples
    # The cache builder iterates over split indices in order, skipping n_j<3 or no mesh.
    # We replicate that logic to find the correct dataset sample_idx for each cache entry.
    raw_data = torch.load(args.pt, map_location="cpu", weights_only=False)
    split_indices = []
    with open(split_file) as f:
        for line in f:
            split_indices.append(json.loads(line.strip())["idx"])

    cache_to_dataset = []
    cache_idx = 0
    for si, raw_idx in enumerate(split_indices):
        if cache_idx >= len(cached):
            break
        d = raw_data[raw_idx]
        n_j = int(d["joints_num"])
        has_mesh = "mesh_pc" in d and "mesh_face" in d
        n_verts = d["mesh_pc"].shape[0] if has_mesh else 0
        if n_j < 3 or not has_mesh or n_verts < 10:
            continue
        cache_to_dataset.append(si)
        cache_idx += 1

    del raw_data
    assert len(cache_to_dataset) == len(cached), (
        f"Mismatch: {len(cache_to_dataset)} matchable vs {len(cached)} cached"
    )
    print(f"Matched {len(cache_to_dataset)} cache entries to dataset indices", flush=True)

    augmented = 0
    teacher_sel_rates = []

    for ci in range(len(cached)):
        si = cache_to_dataset[ci]
        entry = cached[ci]

        teacher_selected, teacher_score = get_teacher_labels(
            dataset, si, model, scorer, device, entry["edge_pairs"],
        )

        entry["teacher_selected"] = teacher_selected
        entry["teacher_score"] = teacher_score
        augmented += 1
        sel_rate = float(teacher_selected.sum()) / max(entry["n_edges"], 1)
        teacher_sel_rates.append(sel_rate)

        if (ci + 1) % 50 == 0:
            mean_sel = np.mean(teacher_sel_rates)
            print(f"  {ci+1}/{len(cached)}: teacher sel rate={mean_sel:.4f}", flush=True)

    print(f"\nAugmented {augmented} samples", flush=True)
    print(f"Mean teacher selection rate: {np.mean(teacher_sel_rates):.4f}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, str(out_path))
    print(f"Saved -> {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
