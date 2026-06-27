"""
Overlay visualizer for HyperNodeNet predictions.

Creates visual comparison of GT vs predicted graphs.

Usage:
  python scripts/make_hypernode_prediction_overlays.py \
    --predictions outputs/models/hypernode_net_v0/val_predictions.jsonl \
    --targets outputs/models/hypernode_net_v0/val_split.jsonl \
    --out outputs/models/hypernode_net_v0/overlays \
    --sample-count 100
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from hyperbone.labels.schema import load_graph_labels, GraphLabel, NodeType
from hyperbone.hypernodes.dataset import NODE_TYPE_TO_IDX, normalize_positions
from hyperbone.hypernodes.eval import compute_metrics, _get_node_positions, _match_nodes

# Node type colors (BGR)
NODE_COLORS = {
    NodeType.ENDPOINT: (0, 255, 0),       # green
    NodeType.BRANCH: (255, 0, 0),         # blue
    NodeType.ARTICULATION: (0, 0, 255),   # red
    NodeType.BEND: (255, 255, 0),         # cyan
    NodeType.RIDGE: (0, 255, 255),        # yellow
    NodeType.ROOT: (255, 0, 255),         # magenta
    NodeType.CENTER: (128, 128, 255),     # light red
    NodeType.SEMANTIC_JOINT: (255, 128, 0),  # light blue
}
DEFAULT_COLOR = (200, 200, 200)


def draw_graph_on_image(
    img: np.ndarray,
    graph: GraphLabel,
    alpha: float = 1.0,
    node_radius: int = 5,
    edge_width: int = 2,
    label: str = "",
    offset_x: int = 0,
) -> np.ndarray:
    """Draw graph nodes and edges onto an image."""
    H, W = img.shape[:2]
    positions = normalize_positions(graph, max(H, W))

    # Draw edges
    for edge in graph.edges:
        if edge.source_node_id in positions and edge.target_node_id in positions:
            p1 = positions[edge.source_node_id]
            p2 = positions[edge.target_node_id]
            pt1 = (int(p1[0] * (W - 1)) + offset_x, int(p1[1] * (H - 1)))
            pt2 = (int(p2[0] * (W - 1)) + offset_x, int(p2[1] * (H - 1)))
            cv2.line(img, pt1, pt2, (180, 180, 180), edge_width, cv2.LINE_AA)

    # Draw nodes
    for node in graph.nodes:
        if node.id in positions:
            p = positions[node.id]
            pt = (int(p[0] * (W - 1)) + offset_x, int(p[1] * (H - 1)))
            color = NODE_COLORS.get(node.node_type, DEFAULT_COLOR)
            cv2.circle(img, pt, node_radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, node_radius, (0, 0, 0), 1, cv2.LINE_AA)

    # Label
    if label:
        cv2.putText(img, label, (offset_x + 5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def make_comparison_image(
    pred_graph: GraphLabel,
    target_graph: GraphLabel,
    resolution: int = 192,
) -> np.ndarray:
    """Create side-by-side GT vs Predicted comparison."""
    W = resolution * 2
    H = resolution
    img = np.zeros((H, W, 3), dtype=np.uint8) + 30  # dark background

    # Left: GT
    draw_graph_on_image(img, target_graph, label="GT", offset_x=0)
    # Right: Predicted
    draw_graph_on_image(img, pred_graph, label="Pred", offset_x=resolution)

    # Separator line
    cv2.line(img, (resolution, 0), (resolution, H), (100, 100, 100), 1)

    # Info text
    valid = pred_graph.metadata.get("valid", True)
    info = f"N:{len(pred_graph.nodes)}/{len(target_graph.nodes)} E:{len(pred_graph.edges)}/{len(target_graph.edges)}"
    if not valid:
        info = "INVALID"
    cv2.putText(img, info, (5, H - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)

    # Source
    source = target_graph.metadata.get("source", "?")
    cv2.putText(img, source, (resolution + 5, H - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)

    return img


def make_grid(images: list[np.ndarray], cols: int = 5) -> np.ndarray:
    """Arrange images into a grid."""
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    rows = math.ceil(len(images) / cols)
    h, w = images[0].shape[:2]

    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8) + 20
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

    return grid


def main():
    parser = argparse.ArgumentParser(description="HyperNodeNet prediction overlays")
    parser.add_argument("--predictions", required=True, help="val_predictions.jsonl")
    parser.add_argument("--targets", default=None, help="val_split.jsonl (auto-detected)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    pred_graphs = load_graph_labels(Path(args.predictions))
    print(f"Loaded {len(pred_graphs)} predictions")

    # Load targets
    if args.targets:
        target_path = Path(args.targets)
    else:
        # Auto-detect from same directory
        target_path = Path(args.predictions).parent / "val_split.jsonl"

    target_graphs = load_graph_labels(target_path)
    print(f"Loaded {len(target_graphs)} targets")

    # Match by sample_id
    target_by_id = {g.sample_id: g for g in target_graphs}

    pairs = []
    for pg in pred_graphs:
        tg = target_by_id.get(pg.sample_id)
        if tg:
            pairs.append((pg, tg))

    print(f"Matched {len(pairs)} prediction-target pairs")

    # Sample
    rng = random.Random(args.seed)
    if len(pairs) > args.sample_count:
        pairs = rng.sample(pairs, args.sample_count)

    # Generate overlays
    images = []
    for i, (pg, tg) in enumerate(pairs):
        img = make_comparison_image(pg, tg, args.resolution)
        images.append(img)
        cv2.imwrite(str(out_dir / f"pred_{i:04d}.png"), img)

    # Grid
    grid = make_grid(images, cols=5)
    grid_path = out_dir / "overlay_grid.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"Saved {len(images)} overlays + grid to {out_dir}")

    # Metrics summary on sampled
    sampled_preds = [p for p, _ in pairs]
    sampled_targets = [t for _, t in pairs]
    metrics = compute_metrics(sampled_preds, sampled_targets)
    print(f"\nSampled metrics:")
    print(f"  Node F1: {metrics.node_f1:.4f}")
    print(f"  Edge F1: {metrics.edge_f1:.4f}")
    print(f"  Chamfer: {metrics.graph_chamfer:.4f}")
    print(f"  Invalid: {metrics.invalid_graph_rate:.2%}")


if __name__ == "__main__":
    main()
