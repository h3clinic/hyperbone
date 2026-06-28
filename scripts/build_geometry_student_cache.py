"""Track B: Precompute geometry-edge cache for student training.

Extracts local mesh patches, corridor patches, and geometric features
for every candidate edge. No skinning fields in input tensors.

Optionally attaches v4.1 teacher edge labels from cached test/val data.

Output per split: one .pt file with list of per-sample dicts containing:
  - joint_pos, active_mask, gt_adj, candidate_mask
  - patch_i, patch_j, corridor: [n_edges, max_pts, 6] tensors
  - geom_feats: [n_edges, 16]
  - edge_pairs: [n_edges, 2] int tensor
  - gt_labels: [n_edges] float tensor
  - teacher_labels: [n_edges] float tensor (if teacher cache available)

Usage:
    python scripts/build_geometry_student_cache.py \
        --pt datasets/anymate/Anymate_test.pt \
        --splits-dir outputs/anymate_local_dev/splits \
        --split train \
        --candidate-k 12 \
        --max-pts 64 \
        --out outputs/models/hyperbone_track_b_student/cache_train.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



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


def extract_patch(mesh_pc, normals, tree, center, radius, max_pts):
    nbrs = tree.query_ball_point(center, radius)
    if len(nbrs) == 0:
        return np.zeros((max_pts, 6), dtype=np.float32)
    pts = mesh_pc[nbrs, :3].copy()
    nrm = normals[nbrs].copy()
    pts -= center
    pts /= max(radius, 1e-6)
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts, nrm = pts[idx], nrm[idx]
    elif len(pts) < max_pts:
        pad = max_pts - len(pts)
        pts = np.concatenate([pts, np.zeros((pad, 3), dtype=np.float32)])
        nrm = np.concatenate([nrm, np.zeros((pad, 3), dtype=np.float32)])
    return np.concatenate([pts, nrm], axis=-1).astype(np.float32)


def extract_corridor(mesh_pc, normals, tree, pos_i, pos_j, tube_r, max_pts):
    midpoint = (pos_i + pos_j) / 2.0
    edge_vec = pos_j - pos_i
    edge_len = np.linalg.norm(edge_vec)
    edge_dir = edge_vec / max(edge_len, 1e-8)
    nbrs = tree.query_ball_point(midpoint, max(edge_len * 0.6, tube_r))
    if len(nbrs) == 0:
        return np.zeros((max_pts, 6), dtype=np.float32)
    pts = mesh_pc[nbrs, :3]
    nrm = normals[nbrs]
    vecs = pts - pos_i
    proj = np.dot(vecs, edge_dir)
    in_range = (proj >= -edge_len * 0.05) & (proj <= edge_len * 1.05)
    perp = vecs - np.outer(proj, edge_dir)
    in_tube = in_range & (np.linalg.norm(perp, axis=1) < tube_r)
    pts, nrm = pts[in_tube].copy(), nrm[in_tube].copy()
    if len(pts) == 0:
        return np.zeros((max_pts, 6), dtype=np.float32)
    pts -= midpoint
    pts /= max(edge_len, 1e-6)
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts, nrm = pts[idx], nrm[idx]
    elif len(pts) < max_pts:
        pad = max_pts - len(pts)
        pts = np.concatenate([pts, np.zeros((pad, 3), dtype=np.float32)])
        nrm = np.concatenate([nrm, np.zeros((pad, 3), dtype=np.float32)])
    return np.concatenate([pts, nrm], axis=-1).astype(np.float32)


def compute_geom_feats(pos_i, pos_j, joints, n_j, tree, mesh_pc):
    edge_vec = pos_j - pos_i
    euc = float(np.linalg.norm(edge_vec))
    edge_dir = edge_vec / max(euc, 1e-8)
    dists = [float(np.linalg.norm(joints[a] - joints[b]))
             for a in range(n_j) for b in range(a+1, n_j)]
    med_d = float(np.median(dists)) if dists else 1.0
    mean_d = float(np.mean(dists)) if dists else 1.0
    dr = max(euc * 0.4, 0.02)
    di = len(tree.query_ball_point(pos_i, dr))
    dj = len(tree.query_ball_point(pos_j, dr))
    mid = (pos_i + pos_j) / 2.0
    dm = len(tree.query_ball_point(mid, dr))
    tv = mesh_pc.shape[0]
    centroid = joints[:n_j].mean(axis=0)
    abs_dir = np.abs(edge_dir)
    return np.array([
        euc, euc / max(med_d, 1e-8),
        di / max(tv, 1), dj / max(tv, 1), dm / max(tv, 1),
        min(di, dj) / max(max(di, dj), 1),
        abs_dir[0], abs_dir[1], abs_dir[2],
        0.0, 0.0,  # rank placeholders filled below
        float(np.linalg.norm(pos_i - centroid)) / max(mean_d, 1e-8),
        float(np.linalg.norm(pos_j - centroid)) / max(mean_d, 1e-8),
        float(n_j) / 128.0,
        euc / max(mean_d, 1e-8),
        float(tv) / 10000.0,
    ], dtype=np.float32)


def process_sample(d, candidate_k, max_pts):
    n_j = int(d["joints_num"])
    if n_j < 3 or "mesh_pc" not in d or "mesh_face" not in d:
        return None

    joints = d["joints"][:n_j].numpy()
    conns = d["conns"][:n_j].numpy()
    mesh_pc = d["mesh_pc"].numpy()
    mesh_face = d["mesh_face"].numpy()
    n_verts = mesh_pc.shape[0]
    if n_verts < 10:
        return None

    vnormals = estimate_vertex_normals(mesh_face, mesh_pc, n_verts)
    vtree = cKDTree(mesh_pc[:, :3])

    # GT adjacency
    gt_edges = set()
    for c in range(n_j):
        p = int(conns[c])
        if 0 <= p < n_j and p != c:
            gt_edges.add((min(c, p), max(c, p)))

    # Candidate edges
    jtree = cKDTree(joints)
    candidates = set()
    for i in range(n_j):
        _, nbrs = jtree.query(joints[i], k=min(candidate_k + 1, n_j))
        for nb in nbrs:
            if nb != i:
                candidates.add((min(i, nb), max(i, nb)))
    candidates |= gt_edges
    edge_list = sorted(candidates)

    dists = [np.linalg.norm(joints[a] - joints[b])
             for a in range(n_j) for b in range(a+1, n_j)]
    med_d = float(np.median(dists)) if dists else 0.1

    patches_i, patches_j, corridors, geoms, labels, pairs = [], [], [], [], [], []

    for (i, j) in edge_list:
        pi, pj = joints[i], joints[j]
        euc = float(np.linalg.norm(pi - pj))
        radius = max(euc * 0.5, med_d * 0.2)
        tube_r = max(euc * 0.3, 0.02)

        patches_i.append(extract_patch(mesh_pc, vnormals, vtree, pi, radius, max_pts))
        patches_j.append(extract_patch(mesh_pc, vnormals, vtree, pj, radius, max_pts))
        corridors.append(extract_corridor(mesh_pc, vnormals, vtree, pi, pj, tube_r, max_pts))
        geoms.append(compute_geom_feats(pi, pj, joints, n_j, vtree, mesh_pc))
        labels.append(1.0 if (i, j) in gt_edges else 0.0)
        pairs.append([i, j])

    # Build full adjacency matrices for topology eval
    max_nodes = 128
    gt_adj = torch.zeros(max_nodes, max_nodes)
    for (i, j) in gt_edges:
        gt_adj[i, j] = 1.0
        gt_adj[j, i] = 1.0

    active_mask = torch.zeros(max_nodes, dtype=torch.bool)
    active_mask[:n_j] = True

    joint_pos = torch.zeros(max_nodes, 3)
    joint_pos[:n_j] = torch.from_numpy(joints)

    cand_mask = torch.zeros(max_nodes, max_nodes, dtype=torch.bool)
    for (i, j) in edge_list:
        cand_mask[i, j] = True
        cand_mask[j, i] = True

    return {
        "joint_pos": joint_pos,
        "active_mask": active_mask,
        "gt_adj": gt_adj,
        "candidate_mask": cand_mask,
        "patch_i": torch.from_numpy(np.stack(patches_i)),
        "patch_j": torch.from_numpy(np.stack(patches_j)),
        "corridor": torch.from_numpy(np.stack(corridors)),
        "geom_feats": torch.from_numpy(np.stack(geoms)),
        "gt_labels": torch.tensor(labels, dtype=torch.float32),
        "edge_pairs": torch.tensor(pairs, dtype=torch.long),
        "n_joints": n_j,
        "n_edges": len(edge_list),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Track B: Build geometry-edge cache (NO skinning inputs)")
    parser.add_argument("--pt", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--max-pts", type=int, default=64)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    print(f"Track B Cache Builder ({args.split})", flush=True)
    print("NO skinning in input tensors.", flush=True)

    data = torch.load(args.pt, map_location="cpu", weights_only=False)
    indices = []
    with open(f"{args.splits_dir}/{args.split}.jsonl") as f:
        for line in f:
            indices.append(json.loads(line.strip())["idx"])

    if args.max_samples > 0:
        indices = indices[:args.max_samples]
    print(f"  {args.split}: {len(indices)} samples", flush=True)

    np.random.seed(42)
    cached = []
    skipped = 0
    total_edges = 0

    for si, idx in enumerate(indices):
        result = process_sample(data[idx], args.candidate_k, args.max_pts)
        if result is None:
            skipped += 1
            continue
        cached.append(result)
        total_edges += result["n_edges"]
        if (si + 1) % 50 == 0:
            print(f"  {si+1}/{len(indices)} ({len(cached)} cached, "
                  f"{total_edges} edges)", flush=True)

    print(f"\nCached {len(cached)} samples, skipped {skipped}", flush=True)
    print(f"Total edges: {total_edges}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, str(out_path))
    print(f"Saved -> {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
