"""Track B v1: Geometry-only student prototype.

Small-scale experiment (200 samples) to test whether local mesh patches
improve edge classification AUC over the Euclidean distance baseline (0.716).

Architecture:
  PointNetLite encodes local mesh patches around each joint.
  Per-edge: concat(patch_tok_i, patch_tok_j, corridor_tok, geom_feats) -> MLP -> score.

Targets:
  GT edge labels (binary).
  Optional: v4.1 teacher scores for distillation.

NO skinning features at any point.

Usage:
    python scripts/train_geometry_student_prototype.py \
        --pt datasets/anymate/Anymate_test.pt \
        --splits-dir outputs/anymate_local_dev/splits \
        --max-samples 200 \
        --epochs 30 \
        --out outputs/models/hyperbone_track_b_student_prototype
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.models.geometry_edge_student import GeometryEdgeStudent


# ---- Mesh patch extraction (no skinning) ----

def estimate_vertex_normals(mesh_face, mesh_pc, n_verts):
    pos = mesh_pc[:, :3]
    normals = np.zeros((n_verts, 3), dtype=np.float32)
    for f in range(mesh_face.shape[0]):
        v0, v1, v2 = int(mesh_face[f, 0]), int(mesh_face[f, 1]), int(mesh_face[f, 2])
        if v0 >= n_verts or v1 >= n_verts or v2 >= n_verts:
            continue
        e1 = pos[v1] - pos[v0]
        e2 = pos[v2] - pos[v0]
        fn = np.cross(e1, e2)
        normals[v0] += fn
        normals[v1] += fn
        normals[v2] += fn
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-8)
    return normals


def extract_local_patch(mesh_pc, vert_normals, vert_tree, center, radius,
                        max_pts=64):
    """Extract a local point cloud patch centered at `center`.

    Returns: [max_pts, 6] array (relative_xyz, normal).
    Points are centered and scaled by radius.
    """
    nbrs = vert_tree.query_ball_point(center, radius)
    if len(nbrs) == 0:
        return np.zeros((max_pts, 6), dtype=np.float32)

    pts = mesh_pc[nbrs, :3].copy()
    nrm = vert_normals[nbrs].copy()

    # Center and scale
    pts -= center
    pts /= max(radius, 1e-6)

    # Subsample or pad
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
        nrm = nrm[idx]
    elif len(pts) < max_pts:
        pad = max_pts - len(pts)
        pts = np.concatenate([pts, np.zeros((pad, 3), dtype=np.float32)])
        nrm = np.concatenate([nrm, np.zeros((pad, 3), dtype=np.float32)])

    return np.concatenate([pts, nrm], axis=-1).astype(np.float32)


def extract_corridor_patch(mesh_pc, vert_normals, vert_tree, pos_i, pos_j,
                           tube_radius, max_pts=64):
    """Extract vertices in a cylinder between joint i and joint j."""
    midpoint = (pos_i + pos_j) / 2.0
    edge_vec = pos_j - pos_i
    edge_len = np.linalg.norm(edge_vec)
    edge_dir = edge_vec / max(edge_len, 1e-8)

    search_r = max(edge_len * 0.6, tube_radius)
    nbrs = vert_tree.query_ball_point(midpoint, search_r)
    if len(nbrs) == 0:
        return np.zeros((max_pts, 6), dtype=np.float32)

    pts = mesh_pc[nbrs, :3]
    nrm = vert_normals[nbrs]

    # Filter: within cylinder
    vecs = pts - pos_i
    proj = np.dot(vecs, edge_dir)
    in_range = (proj >= -edge_len * 0.05) & (proj <= edge_len * 1.05)
    perp = vecs - np.outer(proj, edge_dir)
    perp_dist = np.linalg.norm(perp, axis=1)
    in_tube = in_range & (perp_dist < tube_radius)

    pts = pts[in_tube].copy()
    nrm = nrm[in_tube].copy()

    if len(pts) == 0:
        return np.zeros((max_pts, 6), dtype=np.float32)

    # Center at midpoint, scale by edge length
    pts -= midpoint
    pts /= max(edge_len, 1e-6)

    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
        nrm = nrm[idx]
    elif len(pts) < max_pts:
        pad = max_pts - len(pts)
        pts = np.concatenate([pts, np.zeros((pad, 3), dtype=np.float32)])
        nrm = np.concatenate([nrm, np.zeros((pad, 3), dtype=np.float32)])

    return np.concatenate([pts, nrm], axis=-1).astype(np.float32)


def compute_geom_feats(pos_i, pos_j, joints, n_j, vert_tree, mesh_pc, radius):
    """Compute 16-dim geometry features for edge (i, j)."""
    edge_vec = pos_j - pos_i
    euc_dist = float(np.linalg.norm(edge_vec))
    edge_dir = edge_vec / max(euc_dist, 1e-8)

    # All pairwise distances for normalization
    all_dists = []
    for a in range(n_j):
        for b in range(a + 1, n_j):
            all_dists.append(float(np.linalg.norm(joints[a] - joints[b])))
    median_dist = float(np.median(all_dists)) if all_dists else 1.0
    mean_dist = float(np.mean(all_dists)) if all_dists else 1.0

    rel_dist = euc_dist / max(median_dist, 1e-8)

    # Densities
    density_r = max(euc_dist * 0.4, 0.02)
    den_i = len(vert_tree.query_ball_point(pos_i, density_r))
    den_j = len(vert_tree.query_ball_point(pos_j, density_r))
    midpoint = (pos_i + pos_j) / 2.0
    den_mid = len(vert_tree.query_ball_point(midpoint, density_r))

    # Normalize densities
    total_v = mesh_pc.shape[0]
    nden_i = den_i / max(total_v, 1)
    nden_j = den_j / max(total_v, 1)
    nden_mid = den_mid / max(total_v, 1)
    den_ratio = min(den_i, den_j) / max(max(den_i, den_j), 1)

    # kNN rank features
    joint_dists_i = np.linalg.norm(joints[:n_j] - pos_i, axis=1)
    joint_dists_j = np.linalg.norm(joints[:n_j] - pos_j, axis=1)
    rank_j_in_i = float(np.argsort(np.argsort(joint_dists_i))[np.argmin(np.abs(joint_dists_i - euc_dist))] / max(n_j - 1, 1))
    rank_i_in_j = float(np.argsort(np.argsort(joint_dists_j))[np.argmin(np.abs(joint_dists_j - euc_dist))] / max(n_j - 1, 1))

    # Edge direction components (rotation-equivariant via abs)
    abs_dir = np.abs(edge_dir)

    # Centroid distance
    centroid = joints[:n_j].mean(axis=0)
    dist_to_centroid_i = float(np.linalg.norm(pos_i - centroid))
    dist_to_centroid_j = float(np.linalg.norm(pos_j - centroid))

    feats = np.array([
        euc_dist,
        rel_dist,
        nden_i,
        nden_j,
        nden_mid,
        den_ratio,
        abs_dir[0], abs_dir[1], abs_dir[2],
        rank_j_in_i,
        rank_i_in_j,
        dist_to_centroid_i / max(mean_dist, 1e-8),
        dist_to_centroid_j / max(mean_dist, 1e-8),
        float(n_j) / 128.0,
        euc_dist / max(mean_dist, 1e-8),
        float(total_v) / 10000.0,
    ], dtype=np.float32)
    return feats


# ---- Dataset builder ----

def build_edge_dataset(data, indices, candidate_k=12, max_pts=64):
    """Build edge-level dataset with local mesh patches. No skinning."""
    samples = []

    for si, data_idx in enumerate(indices):
        d = data[data_idx]
        n_j = int(d["joints_num"])

        if n_j < 3 or "mesh_pc" not in d or "mesh_face" not in d:
            continue

        joints = d["joints"][:n_j].numpy()
        conns = d["conns"][:n_j].numpy()
        mesh_pc = d["mesh_pc"].numpy()
        mesh_face = d["mesh_face"].numpy()
        n_verts = mesh_pc.shape[0]

        if n_verts < 10:
            continue

        vert_normals = estimate_vertex_normals(mesh_face, mesh_pc, n_verts)
        vert_tree = cKDTree(mesh_pc[:, :3])

        # GT edges
        gt_edges = set()
        for c in range(n_j):
            p = int(conns[c])
            if 0 <= p < n_j and p != c:
                gt_edges.add((min(c, p), max(c, p)))

        # Candidate edges
        joint_tree = cKDTree(joints)
        candidates = set()
        for i in range(n_j):
            _, nbrs = joint_tree.query(joints[i], k=min(candidate_k + 1, n_j))
            for nb in nbrs:
                if nb != i:
                    candidates.add((min(i, nb), max(i, nb)))
        candidates |= gt_edges

        # Median distance for patch radius
        dists = []
        for i in range(n_j):
            for j in range(i + 1, n_j):
                dists.append(np.linalg.norm(joints[i] - joints[j]))
        median_dist = float(np.median(dists)) if dists else 0.1

        for (i, j) in candidates:
            pos_i = joints[i]
            pos_j = joints[j]
            euc_dist = float(np.linalg.norm(pos_i - pos_j))
            radius = max(euc_dist * 0.5, median_dist * 0.2)
            tube_r = max(euc_dist * 0.3, 0.02)

            patch_i = extract_local_patch(mesh_pc, vert_normals, vert_tree,
                                          pos_i, radius, max_pts)
            patch_j = extract_local_patch(mesh_pc, vert_normals, vert_tree,
                                          pos_j, radius, max_pts)
            corridor = extract_corridor_patch(mesh_pc, vert_normals, vert_tree,
                                              pos_i, pos_j, tube_r, max_pts)
            geom = compute_geom_feats(pos_i, pos_j, joints, n_j, vert_tree,
                                      mesh_pc, radius)

            label = 1.0 if (i, j) in gt_edges else 0.0

            samples.append({
                "patch_i": torch.from_numpy(patch_i),
                "patch_j": torch.from_numpy(patch_j),
                "corridor": torch.from_numpy(corridor),
                "geom_feats": torch.from_numpy(geom),
                "label": torch.tensor(label),
                "sample_idx": si,
                "edge": (i, j),
            })

        if (si + 1) % 20 == 0:
            print(f"  Built edges for {si+1}/{len(indices)} samples "
                  f"({len(samples)} edges)", flush=True)

    return samples


class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["patch_i"], s["patch_j"], s["corridor"], s["geom_feats"], s["label"]


# ---- Training ----

def train_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_loss = 0
    n = 0
    for patch_i, patch_j, corridor, geom, label in loader:
        patch_i = patch_i.to(device)
        patch_j = patch_j.to(device)
        corridor = corridor.to(device)
        geom = geom.to(device)
        label = label.to(device)

        logit = model(patch_i, patch_j, corridor, geom)
        loss = F.binary_cross_entropy_with_logits(
            logit, label, pos_weight=pos_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.shape[0]
        n += label.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    for patch_i, patch_j, corridor, geom, label in loader:
        patch_i = patch_i.to(device)
        patch_j = patch_j.to(device)
        corridor = corridor.to(device)
        geom = geom.to(device)

        logit = model(patch_i, patch_j, corridor, geom)
        all_logits.append(logit.cpu())
        all_labels.append(label)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    if len(np.unique(labels)) < 2:
        return {"roc_auc": 0.0, "pr_auc": 0.0, "n": len(labels)}

    roc = float(roc_auc_score(labels, logits))
    pr = float(average_precision_score(labels, logits))
    return {"roc_auc": roc, "pr_auc": pr, "n": len(labels)}


def main():
    parser = argparse.ArgumentParser(
        description="Track B: Geometry-only student prototype (no skinning)")
    parser.add_argument("--pt", default="datasets/anymate/Anymate_test.pt")
    parser.add_argument("--splits-dir", default="outputs/anymate_local_dev/splits")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--max-pts", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default="outputs/models/hyperbone_track_b_student_prototype")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Track B v1: Geometry-Only Student Prototype", flush=True)
    print("NO skinning features used.", flush=True)
    print(f"Device: {device}", flush=True)

    # Load data
    print("\nLoading dataset...", flush=True)
    data = torch.load(args.pt, map_location="cpu", weights_only=False)

    # Use val for training, test for eval (prototype only)
    val_indices = []
    with open(f"{args.splits_dir}/val.jsonl") as f:
        for line in f:
            val_indices.append(json.loads(line.strip())["idx"])

    test_indices = []
    with open(f"{args.splits_dir}/test.jsonl") as f:
        for line in f:
            test_indices.append(json.loads(line.strip())["idx"])

    n_train = min(args.max_samples, len(val_indices))
    n_test = min(args.max_samples, len(test_indices))
    train_indices = val_indices[:n_train]
    test_indices = test_indices[:n_test]

    # Build edge datasets
    print(f"\nBuilding train edges ({n_train} samples)...", flush=True)
    train_samples = build_edge_dataset(data, train_indices, args.candidate_k, args.max_pts)
    print(f"  Train edges: {len(train_samples)}", flush=True)

    print(f"\nBuilding test edges ({n_test} samples)...", flush=True)
    test_samples = build_edge_dataset(data, test_indices, args.candidate_k, args.max_pts)
    print(f"  Test edges: {len(test_samples)}", flush=True)

    train_labels = [s["label"].item() for s in train_samples]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    print(f"\n  Pos: {n_pos}, Neg: {n_neg}, pos_weight: {pos_weight.item():.2f}", flush=True)

    train_loader = torch.utils.data.DataLoader(
        EdgeDataset(train_samples), batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        EdgeDataset(test_samples), batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Build model
    model = GeometryEdgeStudent(
        patch_in_channels=6,
        patch_out_dim=64,
        geom_feat_dim=16,
        hidden_dim=128,
        dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Baseline: Euclidean distance AUC
    test_labels_np = np.array([s["label"].item() for s in test_samples])
    test_euc = np.array([s["geom_feats"][0].item() for s in test_samples])
    baseline_roc = float(roc_auc_score(test_labels_np, -test_euc))
    baseline_pr = float(average_precision_score(test_labels_np, -test_euc))
    print(f"\nBaseline (Euclidean dist): ROC={baseline_roc:.4f}  PR={baseline_pr:.4f}", flush=True)
    print(f"Reference (Track A skinning_cosine): ROC=0.946", flush=True)

    # Train
    print(f"\n{'Epoch':>6} {'Loss':>10} {'ROC':>10} {'PR':>10} {'vs_base':>10}", flush=True)
    print("-" * 50, flush=True)

    best_roc = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
        metrics = eval_model(model, test_loader, device)
        scheduler.step()

        delta = metrics["roc_auc"] - baseline_roc
        marker = " *" if metrics["roc_auc"] > best_roc else ""
        print(f"  {epoch:4d}   {loss:10.4f} {metrics['roc_auc']:10.4f} "
              f"{metrics['pr_auc']:10.4f} {delta:+10.4f}{marker}", flush=True)

        history.append({
            "epoch": epoch,
            "loss": loss,
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        })

        if metrics["roc_auc"] > best_roc:
            best_roc = metrics["roc_auc"]
            torch.save(model.state_dict(), str(out_dir / "best_student.pt"))

    # Summary
    best_epoch = max(history, key=lambda x: x["roc_auc"])
    print(f"\n{'='*60}", flush=True)
    print("Track B Prototype Results", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Baseline (Euclidean dist):  ROC={baseline_roc:.4f}  PR={baseline_pr:.4f}", flush=True)
    print(f"  Best student (epoch {best_epoch['epoch']}): "
          f"ROC={best_epoch['roc_auc']:.4f}  PR={best_epoch['pr_auc']:.4f}", flush=True)
    print(f"  Delta over baseline:       ROC={best_epoch['roc_auc'] - baseline_roc:+.4f}  "
          f"PR={best_epoch['pr_auc'] - baseline_pr:+.4f}", flush=True)
    print(f"  Reference (skinning_cos):  ROC=0.946", flush=True)

    if best_epoch["roc_auc"] > baseline_roc + 0.02:
        print("\n  >> Mesh patches provide signal above Euclidean baseline.", flush=True)
        print("  >> Proceed to full-scale Track B training.", flush=True)
    else:
        print("\n  >> Mesh patches do NOT meaningfully improve over distance.", flush=True)
        print("  >> Track B may need a deeper representation change.", flush=True)

    # Save report
    report = {
        "track": "B_geometry_student_prototype",
        "skinning_used": False,
        "n_train_samples": n_train,
        "n_test_samples": n_test,
        "n_train_edges": len(train_samples),
        "n_test_edges": len(test_samples),
        "model_params": n_params,
        "baseline_euclidean_roc": baseline_roc,
        "baseline_euclidean_pr": baseline_pr,
        "best_student_roc": best_epoch["roc_auc"],
        "best_student_pr": best_epoch["pr_auc"],
        "best_epoch": best_epoch["epoch"],
        "delta_roc": best_epoch["roc_auc"] - baseline_roc,
        "history": history,
    }
    with open(out_dir / "prototype_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {out_dir / 'prototype_report.json'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
