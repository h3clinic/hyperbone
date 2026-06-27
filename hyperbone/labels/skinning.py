"""
Skinning weight label extraction.

From a rigged mesh, skinning weights tell you which surface vertices each joint
controls. This creates:
- Joint influence heatmaps (projected influence regions)
- Part segmentation labels
- Surface-to-joint affinity fields

This is stronger than single-point joint projection because it teaches the model
the visual REGION controlled by each joint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from pygltflib import GLTF2
    import struct
except ImportError:
    GLTF2 = None

from hyperbone.labels.schema import (
    HyperNodeLabel,
    NodeType,
    LabelSource,
)


@dataclass
class SkinningInfluence:
    """Skinning influence data for one joint."""
    joint_index: int
    joint_name: str
    influenced_vertices: np.ndarray          # [K, 3] world coords
    weights: np.ndarray                       # [K] weight values
    influence_centroid: np.ndarray            # [3] weighted centroid
    influence_radius: float                   # approximate radius


@dataclass
class SkinningLabels:
    """Complete skinning label output for one frame."""
    influences: list[SkinningInfluence]
    joint_heatmaps: Optional[np.ndarray] = None    # [J, H, W]
    part_segmentation: Optional[np.ndarray] = None  # [H, W] int labels
    joint_influence_xy: Optional[np.ndarray] = None # [J, 2] projected centroids


def extract_skinning_weights_gltf(
    gltf_path: Path,
    weight_threshold: float = 0.1,
) -> list[SkinningInfluence]:
    """
    Extract skinning weight data from GLB file.

    For each joint, collects mesh vertices with weight > threshold
    and computes influence centroid and radius.
    """
    if GLTF2 is None:
        raise ImportError("pygltflib required")

    gltf = GLTF2().load(str(gltf_path))
    if not gltf.skins:
        raise ValueError("No skin found")

    skin = gltf.skins[0]
    num_joints = len(skin.joints)

    # Collect all vertices and their joint weights
    # This requires reading JOINTS_0, WEIGHTS_0, and POSITION accessors
    influences: list[SkinningInfluence] = []

    # Get mesh vertices and weights
    vertices, joint_indices_arr, weights_arr = _read_skinned_mesh(gltf, gltf_path)

    if vertices is None:
        # Fallback: return empty influences
        for ji in range(num_joints):
            node = gltf.nodes[skin.joints[ji]]
            influences.append(SkinningInfluence(
                joint_index=ji,
                joint_name=node.name or f"joint_{ji}",
                influenced_vertices=np.zeros((0, 3)),
                weights=np.zeros(0),
                influence_centroid=np.zeros(3),
                influence_radius=0.0,
            ))
        return influences

    # For each joint, find vertices where it has significant weight
    for ji in range(num_joints):
        node = gltf.nodes[skin.joints[ji]]
        name = node.name or f"joint_{ji}"

        # Find where this joint appears in JOINTS_0 with weight > threshold
        mask = np.zeros(len(vertices), dtype=bool)
        w_values = np.zeros(len(vertices), dtype=np.float32)

        for slot in range(joint_indices_arr.shape[1]):
            slot_mask = (joint_indices_arr[:, slot] == ji) & (weights_arr[:, slot] > weight_threshold)
            mask |= slot_mask
            w_values = np.where(slot_mask, np.maximum(w_values, weights_arr[:, slot]), w_values)

        influenced_verts = vertices[mask]
        w = w_values[mask]

        if len(influenced_verts) == 0:
            centroid = np.zeros(3)
            radius = 0.0
        else:
            # Weighted centroid
            w_norm = w / (w.sum() + 1e-8)
            centroid = (influenced_verts * w_norm[:, None]).sum(axis=0)
            # Radius: weighted std
            dists = np.linalg.norm(influenced_verts - centroid, axis=1)
            radius = float((dists * w_norm).sum())

        influences.append(SkinningInfluence(
            joint_index=ji,
            joint_name=name,
            influenced_vertices=influenced_verts,
            weights=w,
            influence_centroid=centroid,
            influence_radius=radius,
        ))

    return influences


def _read_skinned_mesh(gltf, gltf_path: Path):
    """
    Read POSITION, JOINTS_0, WEIGHTS_0 from the first mesh primitive.

    Returns (vertices, joint_indices, weights) or (None, None, None).
    """
    try:
        import struct

        if not gltf.meshes:
            return None, None, None

        mesh = gltf.meshes[0]
        prim = mesh.primitives[0]

        # Read binary buffer
        buffer = gltf.buffers[0]
        bin_path = Path(gltf_path)

        # For GLB, binary data is in the file
        if bin_path.suffix.lower() == ".glb":
            blob = gltf.binary_blob()
        else:
            uri = gltf.buffers[0].uri
            blob = (bin_path.parent / uri).read_bytes()

        if blob is None:
            return None, None, None

        def read_accessor(accessor_idx):
            accessor = gltf.accessors[accessor_idx]
            bv = gltf.bufferViews[accessor.bufferView]
            offset = (bv.byteOffset or 0) + (accessor.byteOffset or 0)
            count = accessor.count

            # Component type
            comp_type = accessor.componentType
            # 5126=FLOAT, 5123=UNSIGNED_SHORT, 5121=UNSIGNED_BYTE
            if comp_type == 5126:
                fmt = 'f'
                size = 4
            elif comp_type == 5123:
                fmt = 'H'
                size = 2
            elif comp_type == 5121:
                fmt = 'B'
                size = 1
            else:
                return None

            # Type -> num components
            type_map = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
            num_comp = type_map.get(accessor.type, 1)

            stride = bv.byteStride or (size * num_comp)
            data = []
            for i in range(count):
                start = offset + i * stride
                row = struct.unpack_from(f'<{num_comp}{fmt}', blob, start)
                data.append(row)

            return np.array(data, dtype=np.float32 if comp_type == 5126 else np.int32)

        # Position
        pos_idx = prim.attributes.POSITION
        if pos_idx is None:
            return None, None, None
        vertices = read_accessor(pos_idx)

        # Joints
        joints_idx = getattr(prim.attributes, 'JOINTS_0', None)
        if joints_idx is None:
            return None, None, None
        joint_indices = read_accessor(joints_idx)

        # Weights
        weights_idx = getattr(prim.attributes, 'WEIGHTS_0', None)
        if weights_idx is None:
            return None, None, None
        weights = read_accessor(weights_idx)

        return vertices, joint_indices.astype(np.int32), weights

    except Exception:
        return None, None, None


def generate_joint_heatmaps(
    influences: list[SkinningInfluence],
    image_width: int,
    image_height: int,
    focal_length: float,
    camera_position: np.ndarray,
    camera_target: np.ndarray,
    sigma_scale: float = 1.5,
) -> np.ndarray:
    """
    Generate per-joint influence heatmaps by projecting influenced vertices.

    Returns: [num_joints, H, W] float32 heatmaps.
    """
    num_joints = len(influences)
    heatmaps = np.zeros((num_joints, image_height, image_width), dtype=np.float32)

    # Simple look-at camera
    forward = camera_target - camera_position
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, np.array([0, 1, 0]))
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    view_rot = np.stack([right, up, -forward], axis=0)

    cx, cy = image_width / 2, image_height / 2

    for ji, inf in enumerate(influences):
        if len(inf.influenced_vertices) == 0:
            continue

        # Project vertices to image
        cam_coords = (inf.influenced_vertices - camera_position) @ view_rot.T
        z = cam_coords[:, 2]
        valid = z < -0.01  # in front of camera
        if not valid.any():
            continue

        u = (focal_length * cam_coords[valid, 0] / (-z[valid]) + cx).astype(np.int32)
        v = (focal_length * cam_coords[valid, 1] / (-z[valid]) + cy).astype(np.int32)
        w = inf.weights[valid] if len(inf.weights) == len(inf.influenced_vertices) else np.ones(valid.sum())

        # Scatter weights into heatmap
        in_bounds = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
        for ui, vi, wi in zip(u[in_bounds], v[in_bounds], w[in_bounds]):
            heatmaps[ji, vi, ui] += wi

    # Gaussian blur each heatmap for smoother regions
    try:
        from scipy.ndimage import gaussian_filter
        for ji in range(num_joints):
            if heatmaps[ji].max() > 0:
                heatmaps[ji] = gaussian_filter(heatmaps[ji], sigma=sigma_scale)
                heatmaps[ji] /= heatmaps[ji].max() + 1e-8
    except ImportError:
        # Normalize without blur
        for ji in range(num_joints):
            mx = heatmaps[ji].max()
            if mx > 0:
                heatmaps[ji] /= mx

    return heatmaps


def generate_part_segmentation(
    influences: list[SkinningInfluence],
    image_width: int,
    image_height: int,
    focal_length: float,
    camera_position: np.ndarray,
    camera_target: np.ndarray,
) -> np.ndarray:
    """
    Generate part segmentation map where each pixel gets the joint index
    with highest skinning weight.

    Returns: [H, W] int32, 0=background, 1..J = joint ID + 1.
    """
    seg = np.zeros((image_height, image_width), dtype=np.float32)
    seg_idx = np.zeros((image_height, image_width), dtype=np.int32)

    forward = camera_target - camera_position
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, np.array([0, 1, 0]))
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    view_rot = np.stack([right, up, -forward], axis=0)
    cx, cy = image_width / 2, image_height / 2

    for ji, inf in enumerate(influences):
        if len(inf.influenced_vertices) == 0:
            continue

        cam_coords = (inf.influenced_vertices - camera_position) @ view_rot.T
        z = cam_coords[:, 2]
        valid = z < -0.01
        if not valid.any():
            continue

        u = (focal_length * cam_coords[valid, 0] / (-z[valid]) + cx).astype(np.int32)
        v = (focal_length * cam_coords[valid, 1] / (-z[valid]) + cy).astype(np.int32)
        w = inf.weights[valid] if len(inf.weights) == len(inf.influenced_vertices) else np.ones(valid.sum())

        in_bounds = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
        for ui, vi, wi in zip(u[in_bounds], v[in_bounds], w[in_bounds]):
            if wi > seg[vi, ui]:
                seg[vi, ui] = wi
                seg_idx[vi, ui] = ji + 1

    return seg_idx


def skinning_to_node_labels(
    influences: list[SkinningInfluence],
) -> list[HyperNodeLabel]:
    """
    Convert skinning influence data to HyperNodeLabels.

    Uses influence centroid as node position and influence radius.
    """
    nodes = []
    for inf in influences:
        if inf.influence_radius < 1e-6 and len(inf.influenced_vertices) == 0:
            continue

        nodes.append(HyperNodeLabel(
            id=inf.joint_index,
            node_type=NodeType.SEMANTIC_JOINT,
            xyz=inf.influence_centroid.tolist(),
            semantic=inf.joint_name,
            radius=inf.influence_radius,
            confidence=min(1.0, len(inf.influenced_vertices) / 100.0),
            label_sources={LabelSource.SKINNING.value: min(1.0, len(inf.influenced_vertices) / 50.0)},
        ))

    return nodes
