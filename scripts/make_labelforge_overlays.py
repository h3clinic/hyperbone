"""
Generate visual overlays for LabelForge graph labels.

Draws nodes by type, edges by type, confidence, and source labels
for visual QA of the generated dataset.

Usage:
  python scripts/make_labelforge_overlays.py ^
    --graphs outputs/labelforge_v0/graphs/fused_graphs.jsonl ^
    --out outputs/labelforge_v0/overlays ^
    --sample-count 100
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.labels.schema import (
    GraphLabel,
    HyperNodeLabel,
    NodeType,
    EdgeType,
    load_graph_labels,
)

try:
    import cv2
except ImportError:
    cv2 = None


# Color scheme per node type (BGR)
NODE_COLORS = {
    NodeType.ROOT: (0, 0, 255),           # red
    NodeType.CENTER: (0, 100, 255),       # orange
    NodeType.ENDPOINT: (0, 255, 0),       # green
    NodeType.BRANCH: (255, 255, 0),       # cyan
    NodeType.ARTICULATION: (255, 0, 255), # magenta
    NodeType.BEND: (200, 200, 0),         # teal
    NodeType.RIDGE: (100, 100, 255),      # salmon
    NodeType.SEMANTIC_JOINT: (0, 200, 255),  # gold
    NodeType.UNKNOWN: (128, 128, 128),    # gray
}

# Edge colors per type (BGR)
EDGE_COLORS = {
    EdgeType.BONE: (255, 255, 255),       # white
    EdgeType.BRANCH: (0, 200, 0),         # green
    EdgeType.VEIN: (0, 180, 0),           # dark green
    EdgeType.MEDIAL_AXIS: (200, 200, 0),  # cyan
    EdgeType.HINGE_LINK: (255, 0, 255),   # magenta
    EdgeType.RIDGE: (100, 100, 255),      # salmon
    EdgeType.DEFORMATION_LINK: (255, 100, 100),  # light blue
    EdgeType.UNKNOWN: (128, 128, 128),    # gray
}


def get_node_xy(node: HyperNodeLabel, canvas_size: int = 400) -> tuple[int, int]:
    """Get 2D pixel coords for a node, normalizing from xyz or xy."""
    if node.xy is not None:
        return int(node.xy[0]), int(node.xy[1])
    if node.xyz is not None:
        # Project xyz to 2D: use x,y scaled to canvas
        x, y = node.xyz[0], node.xyz[1]
        return int(x), int(y)
    return 0, 0


def normalize_graph_coords(graph: GraphLabel, canvas_size: int = 400, margin: int = 40):
    """Compute normalization to fit graph into canvas."""
    xs, ys = [], []
    for n in graph.nodes:
        if n.xyz is not None:
            xs.append(n.xyz[0])
            ys.append(n.xyz[1])
        elif n.xy is not None:
            xs.append(n.xy[0])
            ys.append(n.xy[1])

    if not xs:
        return lambda n: (canvas_size // 2, canvas_size // 2)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xrange = xmax - xmin if xmax > xmin else 1.0
    yrange = ymax - ymin if ymax > ymin else 1.0
    scale = (canvas_size - 2 * margin) / max(xrange, yrange)

    def project(node):
        if node.xyz is not None:
            x, y = node.xyz[0], node.xyz[1]
        elif node.xy is not None:
            x, y = node.xy[0], node.xy[1]
        else:
            return (canvas_size // 2, canvas_size // 2)
        px = int(margin + (x - xmin) * scale)
        py = int(margin + (y - ymin) * scale)
        return (px, py)

    return project


def render_graph_overlay(graph: GraphLabel, canvas_size: int = 400) -> np.ndarray:
    """Render a graph as a visual overlay image."""
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    img[:] = 30  # dark background

    if not graph.nodes:
        cv2.putText(img, "EMPTY GRAPH", (50, canvas_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        return img

    project = normalize_graph_coords(graph, canvas_size)
    node_positions = {n.id: project(n) for n in graph.nodes}

    # Draw edges first (behind nodes)
    for edge in graph.edges:
        src_pos = node_positions.get(edge.source_node_id)
        tgt_pos = node_positions.get(edge.target_node_id)
        if src_pos and tgt_pos:
            color = EDGE_COLORS.get(edge.edge_type, (128, 128, 128))
            thickness = max(1, int(edge.confidence * 2))
            cv2.line(img, src_pos, tgt_pos, color, thickness)

    # Draw nodes
    for node in graph.nodes:
        pos = node_positions.get(node.id)
        if not pos:
            continue
        color = NODE_COLORS.get(node.node_type, (128, 128, 128))
        radius = max(3, int(node.confidence * 6 + 2))
        cv2.circle(img, pos, radius, color, -1)
        # Outline for clarity
        cv2.circle(img, pos, radius + 1, (255, 255, 255), 1)

    # Draw labels
    source = graph.metadata.get("source", "unknown")
    cv2.putText(img, f"id: {graph.sample_id}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(img, f"src: {source}", (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(img, f"nodes: {graph.node_count()}  edges: {graph.edge_count()}",
                (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    # Legend (bottom)
    y_legend = canvas_size - 50
    seen_types = set()
    for n in graph.nodes:
        if n.node_type not in seen_types:
            seen_types.add(n.node_type)
    x_off = 5
    for nt in seen_types:
        color = NODE_COLORS.get(nt, (128, 128, 128))
        cv2.circle(img, (x_off + 5, y_legend), 4, color, -1)
        cv2.putText(img, nt.value, (x_off + 12, y_legend + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
        x_off += len(nt.value) * 7 + 20

    return img


def main():
    parser = argparse.ArgumentParser(description="Generate LabelForge visual overlays")
    parser.add_argument("--graphs", required=True, help="Path to graphs JSONL")
    parser.add_argument("--out", required=True, help="Output overlay directory")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--canvas-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--by-source", action="store_true",
                        help="Generate per-source grid montages")
    args = parser.parse_args()

    if cv2 is None:
        print("ERROR: opencv-python required for overlays")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graphs from {args.graphs}...")
    graphs = load_graph_labels(Path(args.graphs))
    print(f"  Loaded {len(graphs)} graphs")

    # Sample
    random.seed(args.seed)
    if len(graphs) > args.sample_count:
        sampled = random.sample(graphs, args.sample_count)
    else:
        sampled = graphs

    print(f"  Rendering {len(sampled)} overlays...")
    for i, graph in enumerate(sampled):
        img = render_graph_overlay(graph, canvas_size=args.canvas_size)
        out_path = out_dir / f"{graph.sample_id}.png"
        cv2.imwrite(str(out_path), img)

    print(f"  Saved {len(sampled)} overlays to {out_dir}")

    # Create a grid/montage of first 25
    grid_size = min(25, len(sampled))
    if grid_size > 0:
        cols = 5
        rows = (grid_size + cols - 1) // cols
        cell = args.canvas_size // 2
        grid_img = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)

        for idx in range(grid_size):
            img = render_graph_overlay(sampled[idx], canvas_size=cell)
            r, c = idx // cols, idx % cols
            grid_img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = img

        cv2.imwrite(str(out_dir / "_grid_sample.png"), grid_img)
        print(f"  Grid montage saved: {out_dir / '_grid_sample.png'}")

    # Per-source grids
    if args.by_source:
        print("  Generating per-source grids...")
        source_groups: dict[str, list] = {}
        for g in graphs:
            src = g.metadata.get("source", "unknown")
            # Normalize source names for grouping
            if "procedural" in src:
                key = "procedural"
            elif "rigged" in src:
                key = "rigged_animals"
            elif "medial" in src:
                key = "medial_axis"
            elif "motion" in src:
                key = "motion"
            else:
                key = src
            source_groups.setdefault(key, []).append(g)

        # Also group by classification if available
        class_groups: dict[str, list] = {}
        for g in graphs:
            cls = g.metadata.get("classification", "unknown")
            class_groups.setdefault(cls, []).append(g)

        all_groups = {**source_groups}
        if "review" in class_groups:
            all_groups["review_needed"] = class_groups["review"]

        for group_name, group_graphs in all_groups.items():
            if not group_graphs:
                continue
            n_grid = min(25, len(group_graphs))
            sample = random.sample(group_graphs, n_grid) if len(group_graphs) > n_grid else group_graphs[:n_grid]
            cols = 5
            rows = (n_grid + cols - 1) // cols
            cell = args.canvas_size // 2
            grid_img = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
            for idx in range(n_grid):
                img = render_graph_overlay(sample[idx], canvas_size=cell)
                r, c = idx // cols, idx % cols
                grid_img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = img
            path = out_dir / f"{group_name}_grid.png"
            cv2.imwrite(str(path), grid_img)
            print(f"    {group_name}_grid.png ({n_grid} samples)")


if __name__ == "__main__":
    main()
