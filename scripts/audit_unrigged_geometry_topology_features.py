"""Track B geometry signal audit: edge-level features WITHOUT skinning.

For each candidate edge (i,j), compute geometry-only features and measure
separability between GT edges and false candidate edges.

Explicitly excludes: mesh_skins_index, mesh_skins_weight, and any
rig-dependent influence features. This is a Track B audit.

Features computed per candidate edge:
  1. euclidean_dist: ||joint_i - joint_j||
  2. geodesic_dist: Dijkstra on mesh surface graph
  3. geodesic_euclidean_ratio: geodesic / euclidean
  4. curvature_diff: |mean_curvature_i - mean_curvature_j|
  5. curvature_product: mean_curvature_i * mean_curvature_j
  6. normal_alignment: dot(avg_normal_i, edge_direction)
  7. cross_section_area: mesh cross-section area perpendicular to edge
  8. local_density_i: vertex density in radius around joint i
  9. local_density_j: vertex density in radius around joint j
 10. density_ratio: min(di, dj) / max(di, dj)
 11. midpoint_density: vertex density around edge midpoint
 12. tube_density: vertices within cylinder around edge
 13. relative_edge_length: dist / median(all pairwise dists)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, average_precision_score


def build_mesh_graph(mesh_face, mesh_pc, n_verts):
    """Build sparse adjacency from mesh faces with edge-length weights."""
    rows, cols, weights = [], [], []
    pos = mesh_pc[:, :3]
    for f in range(mesh_face.shape[0]):
        v0, v1, v2 = int(mesh_face[f, 0]), int(mesh_face[f, 1]), int(mesh_face[f, 2])
        if v0 >= n_verts or v1 >= n_verts or v2 >= n_verts:
            continue
        if v0 == v1 or v1 == v2 or v0 == v2:
            continue
        for a, b in [(v0, v1), (v1, v2), (v0, v2)]:
            d = float(np.linalg.norm(pos[a] - pos[b]))
            rows.extend([a, b])
            cols.extend([b, a])
            weights.extend([d, d])
    return sparse.csr_matrix((weights, (rows, cols)), shape=(n_verts, n_verts))


def dijkstra_distance(mesh_graph, src_v, dst_v, n_verts):
    """Single-source Dijkstra from src_v, return distance to dst_v."""
    import heapq
    dist_map = {src_v: 0.0}
    heap = [(0.0, src_v)]
    max_explore = min(n_verts, 50000)
    explored = 0
    while heap and explored < max_explore:
        d, u = heapq.heappop(heap)
        if d > dist_map.get(u, float("inf")):
            continue
        explored += 1
        if u == dst_v:
            return d
        start, end = mesh_graph.indptr[u], mesh_graph.indptr[u + 1]
        for idx in range(start, end):
            nb = mesh_graph.indices[idx]
            w = mesh_graph.data[idx]
            nd = d + w
            if nd < dist_map.get(nb, float("inf")):
                dist_map[nb] = nd
                heapq.heappush(heap, (nd, nb))
    return -1.0


def estimate_vertex_curvature(mesh_graph, mesh_pc, n_verts):
    """Estimate mean curvature per vertex via Laplacian approximation.

    For each vertex, curvature ~ ||v - mean(neighbors)||.
    """
    pos = mesh_pc[:, :3]
    curvature = np.zeros(n_verts, dtype=np.float32)
    for v in range(n_verts):
        start, end = mesh_graph.indptr[v], mesh_graph.indptr[v + 1]
        nbrs = mesh_graph.indices[start:end]
        if len(nbrs) == 0:
            continue
        mean_nb = pos[nbrs].mean(axis=0)
        curvature[v] = float(np.linalg.norm(pos[v] - mean_nb))
    return curvature


def estimate_vertex_normals(mesh_face, mesh_pc, n_verts):
    """Estimate per-vertex normals from face normals (area-weighted)."""
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
    norms = np.maximum(norms, 1e-8)
    normals /= norms
    return normals


def compute_geometry_edge_features(
    joint_i, joint_j,
    joints, mesh_pc, mesh_graph, vert_tree, vert_curvature, vert_normals,
    nearest_vert, median_dist, n_verts,
):
    """Compute geometry-only features for one candidate edge."""
    pos_i = joints[joint_i]
    pos_j = joints[joint_j]
    edge_vec = pos_j - pos_i
    euc_dist = float(np.linalg.norm(edge_vec))
    edge_dir = edge_vec / max(euc_dist, 1e-8)

    # 1. Geodesic distance
    src_v = nearest_vert[joint_i]
    dst_v = nearest_vert[joint_j]
    geo_dist = dijkstra_distance(mesh_graph, src_v, dst_v, n_verts)
    geo_ratio = geo_dist / max(euc_dist, 1e-8) if geo_dist > 0 else -1.0

    # 2. Local curvature around each joint
    radius = euc_dist * 0.5 if euc_dist > 1e-6 else 0.05
    nbrs_i = vert_tree.query_ball_point(pos_i, radius)
    nbrs_j = vert_tree.query_ball_point(pos_j, radius)
    curv_i = float(vert_curvature[nbrs_i].mean()) if len(nbrs_i) > 0 else 0.0
    curv_j = float(vert_curvature[nbrs_j].mean()) if len(nbrs_j) > 0 else 0.0
    curv_diff = abs(curv_i - curv_j)
    curv_product = curv_i * curv_j

    # 3. Normal alignment: average normal near joint_i dotted with edge direction
    if len(nbrs_i) > 0:
        avg_normal_i = vert_normals[nbrs_i].mean(axis=0)
        avg_normal_i /= max(np.linalg.norm(avg_normal_i), 1e-8)
        normal_align = abs(float(np.dot(avg_normal_i, edge_dir)))
    else:
        normal_align = 0.0

    # 4. Cross-section: count vertices near the midpoint within a thin slab
    midpoint = (pos_i + pos_j) / 2.0
    slab_half = euc_dist * 0.1 if euc_dist > 1e-6 else 0.01
    tube_radius = euc_dist * 0.3 if euc_dist > 1e-6 else 0.03

    mid_nbrs = vert_tree.query_ball_point(midpoint, tube_radius)
    if len(mid_nbrs) > 0:
        mid_verts = mesh_pc[mid_nbrs, :3]
        projections = np.dot(mid_verts - midpoint, edge_dir)
        in_slab = np.abs(projections) < slab_half
        cross_section_area = float(in_slab.sum())
    else:
        cross_section_area = 0.0

    # 5. Local density
    density_r = max(euc_dist * 0.4, 0.02)
    density_i = len(vert_tree.query_ball_point(pos_i, density_r))
    density_j = len(vert_tree.query_ball_point(pos_j, density_r))
    density_ratio = min(density_i, density_j) / max(max(density_i, density_j), 1)
    midpoint_density = len(vert_tree.query_ball_point(midpoint, density_r))

    # 6. Tube density: vertices within cylinder along edge
    all_in_tube = vert_tree.query_ball_point(midpoint, euc_dist * 0.6)
    tube_count = 0
    if len(all_in_tube) > 0:
        tube_verts = mesh_pc[all_in_tube, :3]
        # Project onto edge axis, keep only those within tube radius
        vecs = tube_verts - pos_i
        proj_len = np.dot(vecs, edge_dir)
        in_range = (proj_len >= -euc_dist * 0.05) & (proj_len <= euc_dist * 1.05)
        perp = vecs - np.outer(proj_len, edge_dir)
        perp_dist = np.linalg.norm(perp, axis=1)
        in_tube = in_range & (perp_dist < tube_radius)
        tube_count = int(in_tube.sum())

    # 7. Relative edge length
    rel_length = euc_dist / max(median_dist, 1e-8)

    return {
        "euclidean_dist": euc_dist,
        "geodesic_dist": geo_dist,
        "geodesic_euclidean_ratio": geo_ratio,
        "curvature_diff": curv_diff,
        "curvature_product": curv_product,
        "normal_alignment": normal_align,
        "cross_section_area": cross_section_area,
        "local_density_i": float(density_i),
        "local_density_j": float(density_j),
        "density_ratio": density_ratio,
        "midpoint_density": float(midpoint_density),
        "tube_density": float(tube_count),
        "relative_edge_length": rel_length,
    }


def compute_auc(labels, scores, kind="roc"):
    labels = np.array(labels)
    scores = np.array(scores)
    if len(np.unique(labels)) < 2:
        return 0.0
    if kind == "roc":
        return float(roc_auc_score(labels, scores))
    return float(average_precision_score(labels, scores))


def main():
    parser = argparse.ArgumentParser(
        description="Track B geometry signal audit (NO skinning features)")
    parser.add_argument("--pt", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Track B Geometry Signal Audit", flush=True)
    print("NO skinning features. Geometry only.", flush=True)
    print("", flush=True)

    print("Loading dataset...", flush=True)
    data = torch.load(args.pt, map_location="cpu", weights_only=False)

    split_path = Path(args.splits_dir) / f"{args.split}.jsonl"
    indices = []
    with open(split_path) as f:
        for line in f:
            indices.append(json.loads(line.strip())["idx"])

    n_use = min(args.max_samples, len(indices))
    sample_indices = indices[:n_use]
    print(f"  Split {args.split}: {len(indices)} total, using {n_use}", flush=True)

    feature_names = [
        "euclidean_dist", "geodesic_dist", "geodesic_euclidean_ratio",
        "curvature_diff", "curvature_product", "normal_alignment",
        "cross_section_area", "local_density_i", "local_density_j",
        "density_ratio", "midpoint_density", "tube_density",
        "relative_edge_length",
    ]

    all_labels = []
    all_features = {f: [] for f in feature_names}

    n_processed = 0
    n_skipped = 0

    for si, data_idx in enumerate(sample_indices):
        d = data[data_idx]
        n_j = int(d["joints_num"])

        if n_j < 3:
            n_skipped += 1
            continue

        if "mesh_pc" not in d or "mesh_face" not in d:
            n_skipped += 1
            continue

        joints = d["joints"][:n_j].numpy()
        conns = d["conns"][:n_j].numpy()
        mesh_pc = d["mesh_pc"].numpy()
        mesh_face = d["mesh_face"].numpy()
        n_verts = mesh_pc.shape[0]

        if n_verts < 10:
            n_skipped += 1
            continue

        # GT adjacency from parent connectivity
        gt_edges = set()
        for c in range(n_j):
            p = int(conns[c])
            if 0 <= p < n_j and p != c:
                gt_edges.add((min(c, p), max(c, p)))

        # Candidate edges (kNN)
        joint_tree = cKDTree(joints)
        candidate_edges = set()
        for i in range(n_j):
            _, nbrs = joint_tree.query(joints[i], k=min(args.candidate_k + 1, n_j))
            for nb in nbrs:
                if nb != i:
                    candidate_edges.add((min(i, nb), max(i, nb)))
        candidate_edges |= gt_edges

        # Precompute mesh structures
        mesh_graph = build_mesh_graph(mesh_face, mesh_pc, n_verts)
        vert_tree = cKDTree(mesh_pc[:, :3])
        vert_curvature = estimate_vertex_curvature(mesh_graph, mesh_pc, n_verts)
        vert_normals = estimate_vertex_normals(mesh_face, mesh_pc, n_verts)

        # Nearest mesh vertex to each joint
        nearest_vert = {}
        for j in range(n_j):
            dists = np.linalg.norm(mesh_pc[:, :3] - joints[j], axis=1)
            nearest_vert[j] = int(np.argmin(dists))

        # Median pairwise joint distance
        all_dists = []
        for i in range(n_j):
            for j in range(i + 1, n_j):
                all_dists.append(float(np.linalg.norm(joints[i] - joints[j])))
        median_dist = float(np.median(all_dists)) if all_dists else 1.0

        for (i, j) in candidate_edges:
            feats = compute_geometry_edge_features(
                i, j, joints, mesh_pc, mesh_graph, vert_tree,
                vert_curvature, vert_normals, nearest_vert, median_dist, n_verts,
            )
            label = 1 if (i, j) in gt_edges else 0
            all_labels.append(label)
            for f in feature_names:
                all_features[f].append(feats[f])

        n_processed += 1
        if (si + 1) % 20 == 0:
            print(f"  Processed {si+1}/{n_use} ({n_processed} ok, {n_skipped} skipped)",
                  flush=True)

    print(f"\nProcessed {n_processed} samples, skipped {n_skipped}", flush=True)
    n_gt = sum(all_labels)
    n_false = len(all_labels) - n_gt
    print(f"Total edges: {len(all_labels)} (GT: {n_gt}, false: {n_false})", flush=True)

    # --- Compute separability ---
    print(f"\n{'='*80}", flush=True)
    print(f"{'Feature':30s} {'GT_mean':>10s} {'False_mean':>10s} "
          f"{'ROC-AUC':>10s} {'PR-AUC':>10s} {'Dir':>8s}", flush=True)
    print(f"{'-'*80}", flush=True)

    results = {}
    labels_arr = np.array(all_labels)

    for f in feature_names:
        vals = np.array(all_features[f])
        gt_vals = vals[labels_arr == 1]
        false_vals = vals[labels_arr == 0]

        gt_mean = float(gt_vals.mean()) if len(gt_vals) > 0 else 0
        false_mean = float(false_vals.mean()) if len(false_vals) > 0 else 0

        valid = np.isfinite(vals) & (vals >= -0.5)
        if valid.sum() > 10 and len(np.unique(labels_arr[valid])) == 2:
            roc = compute_auc(labels_arr[valid], vals[valid], "roc")
            pr = compute_auc(labels_arr[valid], vals[valid], "pr")
            roc_flip = compute_auc(labels_arr[valid], -vals[valid], "roc")
            pr_flip = compute_auc(labels_arr[valid], -vals[valid], "pr")
        else:
            roc = pr = roc_flip = pr_flip = 0.0

        best_roc = max(roc, roc_flip)
        best_pr = max(pr, pr_flip)
        direction = "higher" if roc >= roc_flip else "lower"

        print(f"  {f:30s} {gt_mean:10.4f} {false_mean:10.4f} "
              f"{best_roc:10.4f} {best_pr:10.4f} {direction:>8s}",
              flush=True)

        results[f] = {
            "gt_mean": gt_mean,
            "gt_std": float(gt_vals.std()) if len(gt_vals) > 0 else 0,
            "false_mean": false_mean,
            "false_std": float(false_vals.std()) if len(false_vals) > 0 else 0,
            "roc_auc": best_roc,
            "pr_auc": best_pr,
            "direction": direction,
        }

    # Reference: skinning signal AUCs from Track A audit
    print(f"\n{'='*80}", flush=True)
    print("Reference (Track A skinning signals, NOT computed here):", flush=True)
    print("  skinning_cosine:        ROC-AUC = 0.946", flush=True)
    print("  max_shared_weight:      ROC-AUC = 0.939", flush=True)
    print(f"{'='*80}", flush=True)

    # Sort by ROC-AUC
    ranked = sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True)
    print("\nRanked by ROC-AUC:", flush=True)
    for i, (f, r) in enumerate(ranked):
        marker = " ***" if r["roc_auc"] >= 0.80 else ""
        print(f"  {i+1}. {f:30s} ROC={r['roc_auc']:.4f}  PR={r['pr_auc']:.4f}{marker}",
              flush=True)

    above_80 = sum(1 for _, r in ranked if r["roc_auc"] >= 0.80)
    print(f"\nFeatures with ROC-AUC >= 0.80: {above_80}", flush=True)
    if above_80 == 0:
        print("WARNING: No geometry feature reaches the 0.80 AUC threshold.", flush=True)
        print("Track B may need a fundamentally different approach.", flush=True)

    # Save report
    report = {
        "track": "B_unrigged_geometry",
        "skinning_used": False,
        "n_samples": n_processed,
        "n_gt_edges": n_gt,
        "n_false_edges": n_false,
        "candidate_k": args.candidate_k,
        "features": results,
        "ranked_by_roc_auc": [(f, r["roc_auc"], r["pr_auc"]) for f, r in ranked],
        "reference_skinning_roc_auc": {
            "skinning_cosine": 0.946,
            "max_shared_weight": 0.939,
        },
    }
    report_path = out_dir / "geometry_signal_audit.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {report_path}", flush=True)

    # Save raw features
    raw_path = out_dir / "geometry_features_raw.pt"
    torch.save({
        "labels": torch.tensor(all_labels),
        "features": {f: torch.tensor(all_features[f]) for f in feature_names},
    }, str(raw_path))
    print(f"Raw features -> {raw_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
