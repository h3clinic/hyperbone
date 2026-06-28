"""Skinning-based edge features for topology optimization.

v4.1: Extract per-edge skinning features from mesh/skinning data.
These features require rigged assets with skinning weights — they are NOT
available for unrigged video or raw object meshes.

Track A: Rigged/skinned asset topology extraction.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from scipy import sparse


def build_bone_to_joint_map(
    bones: np.ndarray,
    joints: np.ndarray,
    n_bones: int,
    n_joints: int,
) -> tuple:
    """Map each bone to its start and end joint indices.

    Args:
        bones: [B, 6] bone start_xyz + end_xyz
        joints: [J, 3] joint positions
        n_bones: number of valid bones
        n_joints: number of valid joints

    Returns:
        bone_start: [n_bones] start joint index per bone
        bone_end: [n_bones] end joint index per bone
    """
    bone_start = np.zeros(n_bones, dtype=np.intp)
    bone_end = np.zeros(n_bones, dtype=np.intp)
    j = joints[:n_joints]
    for b in range(n_bones):
        bone_start[b] = int(np.argmin(np.linalg.norm(j - bones[b, :3], axis=1)))
        bone_end[b] = int(np.argmin(np.linalg.norm(j - bones[b, 3:6], axis=1)))
    return bone_start, bone_end


def build_joint_influence_matrix(
    mesh_skins_index: np.ndarray,
    mesh_skins_weight: np.ndarray,
    n_joints: int,
    n_bones: int,
    bone_start: np.ndarray,
    bone_end: np.ndarray,
) -> tuple:
    """Build per-joint skinning influence over mesh vertices.

    Assigns each bone's influence to BOTH its start and end joints so that
    leaf joints (which have no bone starting from them) still receive
    influence through the parent bone's endpoint.

    Returns:
        influence: sparse [n_joints, n_verts] skinning weight matrix
        joint_verts: dict mapping joint_idx -> set of vertex indices
    """
    n_verts = mesh_skins_index.shape[0]
    k = mesh_skins_index.shape[1]

    rows, cols, data = [], [], []
    joint_verts = defaultdict(set)

    for v in range(n_verts):
        for ki in range(k):
            bone_idx = int(mesh_skins_index[v, ki])
            weight = float(mesh_skins_weight[v, ki])
            if bone_idx < 0 or weight <= 0 or bone_idx >= n_bones:
                continue
            sj = int(bone_start[bone_idx])
            if sj < n_joints:
                rows.append(sj)
                cols.append(v)
                data.append(weight)
                joint_verts[sj].add(v)
            ej = int(bone_end[bone_idx])
            if ej < n_joints and ej != sj:
                rows.append(ej)
                cols.append(v)
                data.append(weight)
                joint_verts[ej].add(v)

    if not rows:
        return sparse.csr_matrix((n_joints, n_verts)), joint_verts

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_joints, n_verts)), joint_verts


def compute_skinning_edge_features(
    joint_pos: np.ndarray,
    influence: sparse.csr_matrix,
    joint_verts: dict,
    mesh_pc: np.ndarray,
    candidate_pairs: list,
) -> Dict[str, np.ndarray]:
    """Compute skinning features for candidate edges.

    Args:
        joint_pos: [J, 3] joint positions
        influence: sparse [J, V] influence matrix
        joint_verts: dict joint_idx -> set of vertex indices
        mesh_pc: [V, 3+] mesh vertex positions
        candidate_pairs: list of (i, j) tuples

    Returns:
        dict of feature_name -> [n_edges] arrays
    """
    n_edges = len(candidate_pairs)
    feats = {
        "skinning_cosine": np.zeros(n_edges, dtype=np.float32),
        "max_shared_weight": np.zeros(n_edges, dtype=np.float32),
        "shared_influence_frac": np.zeros(n_edges, dtype=np.float32),
        "shared_influence_count": np.zeros(n_edges, dtype=np.float32),
        "influence_centroid_dist": np.zeros(n_edges, dtype=np.float32),
        "euclidean_dist": np.zeros(n_edges, dtype=np.float32),
    }

    positions = mesh_pc[:, :3]

    for idx, (i, j) in enumerate(candidate_pairs):
        pos_i = joint_pos[i]
        pos_j = joint_pos[j]
        feats["euclidean_dist"][idx] = float(np.linalg.norm(pos_i - pos_j))

        vi_set = joint_verts.get(i, set())
        vj_set = joint_verts.get(j, set())
        n_vi = len(vi_set)
        n_vj = len(vj_set)

        # Cosine similarity of influence vectors
        vec_i = influence[i].toarray().flatten()
        vec_j = influence[j].toarray().flatten()
        norm_i = np.linalg.norm(vec_i)
        norm_j = np.linalg.norm(vec_j)
        if norm_i > 0 and norm_j > 0:
            feats["skinning_cosine"][idx] = float(np.dot(vec_i, vec_j) / (norm_i * norm_j))

        # Shared influence
        shared = vi_set & vj_set
        feats["shared_influence_count"][idx] = len(shared)
        feats["shared_influence_frac"][idx] = len(shared) / max(min(n_vi, n_vj), 1)

        # Max shared weight product
        if shared:
            shared_list = list(shared)[:1000]
            max_w = max(vec_i[v] * vec_j[v] for v in shared_list)
            feats["max_shared_weight"][idx] = float(max_w)

        # Influence centroid distance
        if n_vi > 0 and n_vj > 0:
            ci = positions[list(vi_set)].mean(axis=0)
            cj = positions[list(vj_set)].mean(axis=0)
            feats["influence_centroid_dist"][idx] = float(np.linalg.norm(ci - cj))
        else:
            feats["influence_centroid_dist"][idx] = feats["euclidean_dist"][idx] * 2

    return feats


def compute_skinning_score_matrices(
    joint_pos_t: torch.Tensor,
    active_mask_t: torch.Tensor,
    candidate_mask_t: torch.Tensor,
    raw_sample: dict,
) -> Dict[str, torch.Tensor]:
    """Compute per-edge skinning score matrices for the topology optimizer.

    Returns separate matrices for each skinning feature so the optimizer
    can weight them independently.

    Args:
        joint_pos_t: [N, 3] normalized joint positions
        active_mask_t: [N] bool active mask
        candidate_mask_t: [N, N] bool candidate edge mask
        raw_sample: dict with raw dataset fields

    Returns:
        dict with keys "skinning_cosine" and "max_shared_weight",
        each [N, N] tensor (higher = more likely true edge)
    """
    N = joint_pos_t.shape[0]
    empty = {"skinning_cosine": torch.zeros(N, N), "max_shared_weight": torch.zeros(N, N)}

    n_j = int(raw_sample["joints_num"])
    n_b = int(raw_sample["bones_num"])
    if n_j < 2 or n_b < 1:
        return empty

    required = ["mesh_skins_index", "mesh_skins_weight", "mesh_pc", "bones", "joints"]
    for key in required:
        if key not in raw_sample:
            return empty

    joints_raw = raw_sample["joints"][:n_j].numpy()
    bones_raw = raw_sample["bones"][:n_b].numpy()
    msi = raw_sample["mesh_skins_index"].numpy()
    msw = raw_sample["mesh_skins_weight"].numpy()
    mesh_pc = raw_sample["mesh_pc"].numpy()

    bone_start, bone_end = build_bone_to_joint_map(bones_raw, joints_raw, n_b, n_j)
    influence, joint_verts = build_joint_influence_matrix(msi, msw, n_j, n_b, bone_start, bone_end)

    joint_pos_np = joint_pos_t[:n_j].numpy()

    active_indices = torch.where(active_mask_t)[0].tolist()
    pairs = []
    for gi in active_indices:
        for gj in active_indices:
            if gi < gj and candidate_mask_t[gi, gj]:
                pairs.append((gi, gj))

    if not pairs:
        return empty

    feats = compute_skinning_edge_features(
        joint_pos_np, influence, joint_verts, mesh_pc, pairs,
    )

    cos_mat = torch.zeros(N, N)
    msw_mat = torch.zeros(N, N)
    for idx, (gi, gj) in enumerate(pairs):
        c = float(feats["skinning_cosine"][idx])
        w = float(feats["max_shared_weight"][idx])
        cos_mat[gi, gj] = c
        cos_mat[gj, gi] = c
        msw_mat[gi, gj] = w
        msw_mat[gj, gi] = w

    return {"skinning_cosine": cos_mat, "max_shared_weight": msw_mat}
