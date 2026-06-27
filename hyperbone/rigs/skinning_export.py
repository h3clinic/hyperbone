"""
Skinning weight export for the Anymate pipeline.

Extracts per-vertex joint weights and generates joint influence heatmaps.
This module works both inside Blender (for Blender vertex groups) and
outside Blender (for numpy-based weight processing).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def extract_vertex_weights_blender(mesh_obj, armature_obj) -> Tuple[np.ndarray, List[str]]:
    """
    Extract skinning weights from Blender vertex groups.
    
    Returns:
        weights: [V, J] sparse-ish float32 array
        joint_names: list of joint names corresponding to columns
    """
    mesh = mesh_obj.data
    vertex_count = len(mesh.vertices)

    # Get bone names from armature
    bone_names = [bone.name for bone in armature_obj.pose.bones]
    joint_count = len(bone_names)
    name_to_idx = {name: idx for idx, name in enumerate(bone_names)}

    # Map vertex groups to bone indices
    vg_map = {}
    for vg in mesh_obj.vertex_groups:
        if vg.name in name_to_idx:
            vg_map[vg.index] = name_to_idx[vg.name]

    # Extract weights
    weights = np.zeros((vertex_count, joint_count), dtype=np.float32)

    for vi, vert in enumerate(mesh.vertices):
        for g in vert.groups:
            if g.group in vg_map:
                ji = vg_map[g.group]
                weights[vi, ji] = g.weight

    # Normalize rows to sum to 1 (standard LBS requirement)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    weights = weights / row_sums

    return weights, bone_names


def save_skinning_weights(weights: np.ndarray, joint_names: List[str], output_path: Path):
    """Save skinning weights as compressed NPZ."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        weights=weights,
        joint_names=np.array(joint_names, dtype=object),
    )


def load_skinning_weights(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load skinning weights from NPZ."""
    data = np.load(path, allow_pickle=True)
    return data["weights"], list(data["joint_names"])


def generate_joint_influence_heatmaps(
    weights: np.ndarray,
    vertices_2d: np.ndarray,
    resolution: Tuple[int, int] = (512, 512),
    sigma: float = 5.0,
) -> np.ndarray:
    """
    Generate per-joint influence heatmaps from skinning weights and projected vertices.
    
    Args:
        weights: [V, J] skinning weights
        vertices_2d: [V, 2] projected vertex positions in image space
        resolution: (H, W) output heatmap size
        sigma: Gaussian spread for each vertex
        
    Returns:
        heatmaps: [J, H, W] float32 influence maps
    """
    V, J = weights.shape
    H, W = resolution
    heatmaps = np.zeros((J, H, W), dtype=np.float32)

    # For efficiency, only process vertices with non-negligible weight
    for ji in range(J):
        joint_weights = weights[:, ji]
        active_mask = joint_weights > 0.01
        if not active_mask.any():
            continue

        active_verts = vertices_2d[active_mask]
        active_weights = joint_weights[active_mask]

        for vi in range(len(active_verts)):
            x, y = active_verts[vi]
            w = active_weights[vi]

            # Integer position
            ix, iy = int(round(x)), int(round(y))
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                continue

            # Simple Gaussian splat (truncated for speed)
            radius = int(3 * sigma)
            y_min = max(0, iy - radius)
            y_max = min(H, iy + radius + 1)
            x_min = max(0, ix - radius)
            x_max = min(W, ix + radius + 1)

            yy = np.arange(y_min, y_max)
            xx = np.arange(x_min, x_max)
            yy, xx = np.meshgrid(yy, xx, indexing='ij')

            gauss = np.exp(-((yy - iy) ** 2 + (xx - ix) ** 2) / (2 * sigma ** 2))
            heatmaps[ji, y_min:y_max, x_min:x_max] += w * gauss

    # Normalize each joint heatmap to [0, 1]
    for ji in range(J):
        hmax = heatmaps[ji].max()
        if hmax > 0:
            heatmaps[ji] /= hmax

    return heatmaps


def save_influence_heatmaps(heatmaps: np.ndarray, joint_names: List[str], output_dir: Path):
    """Save per-joint influence heatmaps as individual PNGs and a combined NPZ."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined
    np.savez_compressed(
        output_dir / "influence_maps.npz",
        heatmaps=heatmaps,
        joint_names=np.array(joint_names, dtype=object),
    )

    # Save individual PNGs for visualization
    try:
        import cv2
        for ji, name in enumerate(joint_names):
            hmap = (heatmaps[ji] * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"{name}.png"), hmap)
    except ImportError:
        pass  # cv2 not available, skip PNG export
