"""
Compare HyperBone 2D graph to projected Blender armature joints (GT).

Metrics:
- hyperbone_nodes_per_frame
- gt_joints_per_frame
- centroid_distance_px
- bbox_iou
- nearest_neighbor_distance_px
- coverage_score (% of GT joints within threshold of any HyperBone node)
- topology_note

Usage:
  python -m hyperbone.eval.fox_gt_compare \
    --gt outputs/synthetic_animals/fox_armature_gt.jsonl \
    --hyperbone output/fox3d/pipeline/graphs/graphs.jsonl \
    --out outputs/synthetic_animals/fox_hyperbone
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def bbox_iou(bbox_a: List[int], bbox_b: List[int]) -> float:
    """IoU between two xywh bboxes."""
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix or iy2 <= iy:
        return 0.0

    inter = (ix2 - ix) * (iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nearest_neighbor_distances(
    gt_points: np.ndarray, hb_points: np.ndarray
) -> np.ndarray:
    """For each GT point, find distance to nearest HyperBone node."""
    if len(hb_points) == 0:
        return np.full(len(gt_points), float('inf'))
    # Broadcast distance computation
    # gt_points: (G, 2), hb_points: (H, 2)
    diffs = gt_points[:, None, :] - hb_points[None, :, :]  # (G, H, 2)
    dists = np.linalg.norm(diffs, axis=2)  # (G, H)
    return dists.min(axis=1)  # (G,)


def coverage_score(nn_dists: np.ndarray, threshold_px: float = 40.0) -> float:
    """Fraction of GT joints within threshold of any HyperBone node."""
    if len(nn_dists) == 0:
        return 0.0
    return float(np.mean(nn_dists <= threshold_px))


def compare_frame(gt_record: Dict, hb_record: Dict) -> Dict:
    """Compare a single frame's GT armature to HyperBone graph."""
    # Extract GT projected 2D joints
    gt_joints = gt_record.get("joints", gt_record.get("joints_3d", []))
    gt_points = []
    for j in gt_joints:
        xy = j.get("image_xy")
        if xy and j.get("visible", True):
            gt_points.append(xy)
    gt_points = np.array(gt_points) if gt_points else np.zeros((0, 2))

    # Extract HyperBone node positions
    hb_nodes = hb_record.get("nodes", [])
    hb_points = []
    for n in hb_nodes:
        xy = n.get("xy")
        if xy:
            hb_points.append(xy)
    hb_points = np.array(hb_points) if hb_points else np.zeros((0, 2))

    # Centroids
    gt_centroid = gt_points.mean(axis=0) if len(gt_points) > 0 else np.array([0, 0])
    hb_centroid = hb_points.mean(axis=0) if len(hb_points) > 0 else np.array([0, 0])
    centroid_dist = float(np.linalg.norm(gt_centroid - hb_centroid))

    # Nearest neighbor
    nn_dists = nearest_neighbor_distances(gt_points, hb_points)
    cov = coverage_score(nn_dists, threshold_px=40.0)

    # BBox IoU
    gt_bbox = gt_record.get("bbox_xywh")
    hb_bbox = hb_record.get("bbox_xywh")
    iou = bbox_iou(gt_bbox, hb_bbox) if (gt_bbox and hb_bbox) else 0.0

    return {
        "frame_idx": gt_record.get("frame_idx", 0),
        "gt_joints": len(gt_points),
        "hb_nodes": len(hb_points),
        "centroid_distance_px": round(centroid_dist, 2),
        "bbox_iou": round(iou, 4),
        "nn_mean_px": round(float(nn_dists.mean()), 2) if len(nn_dists) > 0 else None,
        "nn_p95_px": round(float(np.percentile(nn_dists, 95)), 2) if len(nn_dists) > 0 else None,
        "coverage_40px": round(cov, 4),
    }


def run_comparison(gt_path: str, hb_path: str, out_dir: str) -> Dict:
    """Run full GT vs HyperBone comparison."""
    gt_records = load_jsonl(gt_path)
    hb_records = load_jsonl(hb_path)

    # Index by frame
    gt_by_frame = {r["frame_idx"]: r for r in gt_records}
    hb_by_frame = {r["frame_idx"]: r for r in hb_records}

    # Compare matched frames
    common_frames = sorted(set(gt_by_frame.keys()) & set(hb_by_frame.keys()))
    frame_results = []

    for fi in common_frames:
        result = compare_frame(gt_by_frame[fi], hb_by_frame[fi])
        frame_results.append(result)

    # Aggregate
    if frame_results:
        centroid_dists = [r["centroid_distance_px"] for r in frame_results]
        ious = [r["bbox_iou"] for r in frame_results]
        nn_means = [r["nn_mean_px"] for r in frame_results if r["nn_mean_px"] is not None]
        nn_p95s = [r["nn_p95_px"] for r in frame_results if r["nn_p95_px"] is not None]
        coverages = [r["coverage_40px"] for r in frame_results]
        hb_nodes = [r["hb_nodes"] for r in frame_results]
        gt_joints = [r["gt_joints"] for r in frame_results]
    else:
        centroid_dists = ious = nn_means = nn_p95s = coverages = hb_nodes = gt_joints = []

    summary = {
        "gt_path": gt_path,
        "hyperbone_path": hb_path,
        "frames_compared": len(frame_results),
        "gt_frames_total": len(gt_records),
        "hb_frames_total": len(hb_records),
        "avg_gt_joints": round(np.mean(gt_joints), 1) if gt_joints else 0,
        "avg_hb_nodes": round(np.mean(hb_nodes), 1) if hb_nodes else 0,
        "avg_centroid_distance_px": round(np.mean(centroid_dists), 2) if centroid_dists else None,
        "avg_bbox_iou": round(np.mean(ious), 4) if ious else None,
        "avg_nn_distance_px": round(np.mean(nn_means), 2) if nn_means else None,
        "p95_nn_distance_px": round(np.mean(nn_p95s), 2) if nn_p95s else None,
        "avg_coverage_40px": round(np.mean(coverages), 4) if coverages else None,
        "topology_note": (
            "HyperBone graph is a structural 2D contour skeleton extracted via "
            "thinning. GT is a semantic internal armature from Blender. These are "
            "fundamentally different representations — HyperBone traces silhouette "
            "edges while GT describes internal joint locations."
        ),
        "per_frame": frame_results,
    }

    # Write outputs
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "fox_gt_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nGT vs HyperBone Comparison")
    print("=" * 50)
    print(f"  Frames compared:        {summary['frames_compared']}")
    print(f"  Avg GT joints/frame:    {summary['avg_gt_joints']}")
    print(f"  Avg HyperBone nodes:    {summary['avg_hb_nodes']}")
    print(f"  Avg centroid dist (px): {summary['avg_centroid_distance_px']}")
    print(f"  Avg bbox IoU:           {summary['avg_bbox_iou']}")
    print(f"  Avg NN distance (px):   {summary['avg_nn_distance_px']}")
    print(f"  P95 NN distance (px):   {summary['p95_nn_distance_px']}")
    print(f"  Coverage @40px:         {summary['avg_coverage_40px']}")
    print(f"\n  Note: {summary['topology_note'][:80]}...")
    print(f"\n  Output: {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare GT armature to HyperBone graph")
    parser.add_argument("--gt", required=True, help="GT armature JSONL")
    parser.add_argument("--hyperbone", required=True, help="HyperBone graph JSONL")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    run_comparison(args.gt, args.hyperbone, args.out)


if __name__ == "__main__":
    main()
