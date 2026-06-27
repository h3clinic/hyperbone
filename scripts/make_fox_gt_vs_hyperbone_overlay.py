"""
Side-by-side overlay: GT 3D armature (projected) vs HyperBone 2D graph.

Shows both on the same rendered frames so you can see exactly how the
2D contour skeleton differs from internal armature joints.

Usage:
  python scripts/make_fox_gt_vs_hyperbone_overlay.py \
    --video output/fox3d/fox3d_walk.mp4 \
    --gt output/fox3d/fox_armature_gt.jsonl \
    --hyperbone output/fox3d/pipeline/graphs/graphs.jsonl \
    --out output/fox3d/fox_gt_vs_hyperbone_overlay.mp4
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def draw_gt_armature(img, record, color_bone=(255, 200, 0), color_joint=(0, 255, 255)):
    """Draw projected GT armature joints and bones."""
    joints = record.get("joints", [])
    bones = record.get("bones", [])

    joint_xy = {}
    for j in joints:
        if j.get("visible", True):
            xy = j.get("image_xy")
            if xy:
                joint_xy[j["id"]] = (int(xy[0]), int(xy[1]))

    for bone in bones:
        pid = bone.get("parent_id")
        cid = bone.get("child_id")
        if pid in joint_xy and cid in joint_xy:
            cv2.line(img, joint_xy[pid], joint_xy[cid], color_bone, 2)

    for jid, (px, py) in joint_xy.items():
        cv2.circle(img, (px, py), 5, color_joint, -1)
        cv2.circle(img, (px, py), 6, (0, 180, 200), 1)


def draw_hyperbone_graph(img, record, color_edge=(0, 255, 255), color_node=(255, 0, 255)):
    """Draw HyperBone 2D graph (nodes + edges)."""
    nodes = record.get("nodes", [])
    edges = record.get("edges", [])

    node_xy = {}
    for n in nodes:
        nid = n["id"]
        xy = n.get("xy")
        if xy:
            node_xy[nid] = (int(xy[0]), int(xy[1]))

    for edge in edges:
        pid = edge.get("parent")
        cid = edge.get("child")
        if pid in node_xy and cid in node_xy:
            cv2.line(img, node_xy[pid], node_xy[cid], color_edge, 1)

    for nid, (px, py) in node_xy.items():
        cv2.circle(img, (px, py), 2, color_node, -1)


def compute_coverage(gt_record, hb_record, threshold=40.0):
    """Quick coverage metric for display."""
    joints = gt_record.get("joints", [])
    nodes = hb_record.get("nodes", [])

    gt_pts = np.array([j["image_xy"] for j in joints if j.get("visible", True) and j.get("image_xy")])
    hb_pts = np.array([n["xy"] for n in nodes if n.get("xy")])

    if len(gt_pts) == 0 or len(hb_pts) == 0:
        return 0.0

    diffs = gt_pts[:, None, :] - hb_pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    nn_dists = dists.min(axis=1)
    return float(np.mean(nn_dists <= threshold))


def main():
    parser = argparse.ArgumentParser(description="GT vs HyperBone overlay video")
    parser.add_argument("--video", required=True, help="Input rendered video")
    parser.add_argument("--gt", required=True, help="GT armature JSONL")
    parser.add_argument("--hyperbone", required=True, help="HyperBone graph JSONL")
    parser.add_argument("--out", required=True, help="Output overlay video")
    args = parser.parse_args()

    gt_records = load_jsonl(args.gt)
    hb_records = load_jsonl(args.hyperbone)
    gt_by_frame = {r["frame_idx"]: r for r in gt_records}
    hb_by_frame = {r["frame_idx"]: r for r in hb_records}

    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    print(f"Video: {w}x{h} @ {fps}fps, {n_frames} frames")
    print(f"GT records: {len(gt_records)}, HyperBone records: {len(hb_records)}")

    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw GT armature (cyan/yellow = internal bones)
        if fi in gt_by_frame:
            draw_gt_armature(frame, gt_by_frame[fi])

        # Draw HyperBone graph (magenta/yellow-green = 2D contour)
        if fi in hb_by_frame:
            draw_hyperbone_graph(frame, hb_by_frame[fi],
                                 color_edge=(0, 200, 100), color_node=(200, 0, 200))

        # Metrics overlay
        if fi in gt_by_frame and fi in hb_by_frame:
            cov = compute_coverage(gt_by_frame[fi], hb_by_frame[fi])
            gt_n = len([j for j in gt_by_frame[fi]["joints"] if j.get("visible", True)])
            hb_n = len(hb_by_frame[fi].get("nodes", []))
            cv2.putText(frame, f"GT:{gt_n}j  HB:{hb_n}n  Cov@40px:{cov:.0%}",
                        (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        elif fi in gt_by_frame:
            cv2.putText(frame, "GT armature only (no HB this frame)",
                        (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Legend
        cv2.putText(frame, f"Frame {fi}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.circle(frame, (w - 130, 15), 4, (0, 255, 255), -1)
        cv2.putText(frame, "GT Armature", (w - 120, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.circle(frame, (w - 130, 32), 3, (200, 0, 200), -1)
        cv2.putText(frame, "HyperBone 2D", (w - 120, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 200), 1)

        writer.write(frame)
        fi += 1

    cap.release()
    writer.release()
    print(f"\nOverlay saved: {out_path}")
    print(f"  Frames with GT: {len(gt_by_frame)}")
    print(f"  Frames with HyperBone: {len(hb_by_frame)}")
    print(f"  Frames overlapping: {len(set(gt_by_frame) & set(hb_by_frame))}")


if __name__ == "__main__":
    main()
