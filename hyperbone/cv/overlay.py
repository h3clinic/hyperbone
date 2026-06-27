"""Overlay generation — draw skeleton graphs on frames for debugging."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def draw_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    graph: Dict,
    object_id: int = 0,
    quality: Dict = None,
) -> np.ndarray:
    """Draw mask outline, skeleton nodes, edges, and quality info on a copy of the frame.

    Returns BGR image with overlays.
    """
    overlay = frame.copy()

    # Draw mask outline in green
    if mask is not None and mask.sum() > 0:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # Draw bbox if available in quality
    if quality and quality.get("bbox_width", 0) > 0:
        # Reconstruct bbox from nodes extent
        pass

    # Draw edges in cyan
    node_map = {n["id"]: tuple(n["xy"]) for n in nodes}
    for edge in edges:
        p1 = node_map.get(edge["parent"])
        p2 = node_map.get(edge["child"])
        if p1 and p2:
            cv2.line(overlay, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)

    # Draw nodes
    for node in nodes:
        x, y = node["xy"]
        ntype = node.get("type", "")
        if ntype == "junction":
            color = (0, 0, 255)  # red
            radius = 4
        elif ntype == "endpoint":
            color = (255, 0, 255)  # magenta
            radius = 3
        else:
            color = (0, 255, 255)  # yellow
            radius = 3
        cv2.circle(overlay, (x, y), radius, color, -1)
        cv2.putText(
            overlay, str(node["id"]), (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Draw quality label
    if quality:
        accepted = quality.get("accepted", False)
        label = "ACCEPTED" if accepted else "REJECTED"
        color = (0, 200, 0) if accepted else (0, 0, 220)
        cv2.putText(overlay, label, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Draw reject reasons
        if not accepted:
            reasons = quality.get("reject_reasons", [])
            for i, reason in enumerate(reasons[:4]):
                cv2.putText(overlay, reason[:80], (8, 40 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1, cv2.LINE_AA)

        # Draw stats
        stats_text = f"N:{quality.get('skeleton_node_count', 0)} E:{quality.get('skeleton_edge_count', 0)} A:{quality.get('mask_area_ratio', 0):.3f}"
        h = overlay.shape[0]
        cv2.putText(overlay, stats_text, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    return overlay


def save_overlay(
    overlay: np.ndarray, output_dir: str, frame_idx: int, object_id: int
) -> Path:
    """Save overlay image as PNG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"overlay_{frame_idx:06d}_obj{object_id:03d}.png"
    cv2.imwrite(str(path), overlay)
    return path
