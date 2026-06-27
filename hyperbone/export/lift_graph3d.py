"""
HyperBone 2D graph to canonical 3D lift.

Converts the 2D HyperBone contour graph into a flat canonical 3D representation.
This is NOT true 3D pose — it's a normalized 2D→canonical mapping for comparison.

depth_source = "flat_canonical" — all z=0.

Usage:
  python -m hyperbone.export.lift_graph3d \
    --graphs output/fox3d/pipeline/graphs/graphs.jsonl \
    --out output/fox3d/graphs3d/hyperbone_graph3d.jsonl
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict


def lift_graph_to_canonical(graph_record: Dict) -> Dict:
    """
    Convert a 2D HyperBone graph record to canonical 3D coordinates.

    Canonical mapping:
      x = (node_x - bbox_x) / bbox_w - 0.5
      y = 0.5 - (node_y - bbox_y) / bbox_h
      z = 0.0
    """
    bbox_xywh = graph_record.get("bbox_xywh", [0, 0, 1, 1])
    bx, by, bw, bh = bbox_xywh
    bw = max(bw, 1)
    bh = max(bh, 1)

    nodes_2d = graph_record.get("nodes", [])
    edges = graph_record.get("edges", [])

    nodes_3d = []
    for node in nodes_2d:
        node_id = node["id"]
        # Use the image-space xy coordinates
        nx, ny = node.get("xy", [0, 0])

        # Canonical mapping
        cx = (nx - bx) / bw - 0.5
        cy = 0.5 - (ny - by) / bh
        cz = 0.0

        nodes_3d.append({
            "id": node_id,
            "image_xy": [nx, ny],
            "canonical_xyz": [round(cx, 5), round(cy, 5), round(cz, 5)],
            "confidence": 1.0,
            "node_type": node.get("type", "unknown"),
        })

    edges_3d = []
    # Build position lookup for length computation
    pos_map = {n["id"]: n["canonical_xyz"] for n in nodes_3d}
    for edge in edges:
        src = edge.get("parent", edge.get("source", 0))
        tgt = edge.get("child", edge.get("target", 0))
        # Compute canonical length
        if src in pos_map and tgt in pos_map:
            import math
            p1 = pos_map[src]
            p2 = pos_map[tgt]
            length = math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
        else:
            length = 0.0

        edges_3d.append({
            "source": src,
            "target": tgt,
            "length_canonical": round(length, 5),
        })

    return {
        "frame_idx": graph_record.get("frame_idx", 0),
        "timestamp_sec": graph_record.get("timestamp_sec", 0.0),
        "object_id": graph_record.get("object_id", ""),
        "object_label": graph_record.get("object_label", ""),
        "skeleton_mapper": graph_record.get("skeleton_mapper", "hyperbone-custom"),
        "depth_source": "flat_canonical",
        "bbox_xywh": bbox_xywh,
        "node_count": len(nodes_3d),
        "edge_count": len(edges_3d),
        "nodes": nodes_3d,
        "edges": edges_3d,
        "accepted": graph_record.get("accepted", False),
    }


def lift_graphs_file(input_path: str, output_path: str) -> int:
    """Process an entire graphs JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            lifted = lift_graph_to_canonical(record)
            fout.write(json.dumps(lifted) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Lift HyperBone 2D graphs to canonical 3D")
    parser.add_argument("--graphs", required=True, help="Input graphs JSONL path")
    parser.add_argument("--out", required=True, help="Output graphs3d JSONL path")
    args = parser.parse_args()

    count = lift_graphs_file(args.graphs, args.out)
    print(f"[Lift3D] Lifted {count} graph records -> {args.out}")
    print(f"[Lift3D] depth_source = flat_canonical (z=0 for all nodes)")
    print(f"[Lift3D] NOTE: This is NOT true 3D. It is a normalized 2D→canonical mapping.")


if __name__ == "__main__":
    main()
