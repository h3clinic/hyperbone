"""
Anymate Static Rig Dataset — loads point clouds and rig targets from .pt file.

Input: point cloud / voxels / mesh surface samples
Target: joint positions, bone connectivity, bone lengths, skinning weights, node types
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from hyperbone.rigs.parent_targets import extract_parent_targets


class AnymateStaticRigDataset(Dataset):
    """
    Dataset for static rig prediction: shape → skeleton + skinning.

    Args:
        pt_path: path to Anymate_test.pt
        split_path: path to train.jsonl / val.jsonl / test.jsonl
        max_joints: maximum joints to pad to
        max_bones: maximum bones to pad to
        pc_points: number of point cloud points to sample
        use_voxels: include voxel grid as input
    """

    def __init__(
        self,
        pt_path: str,
        split_path: str,
        max_joints: int = 128,
        max_bones: int = 128,
        pc_points: int = 2048,
        use_voxels: bool = False,
    ):
        self.max_joints = max_joints
        self.max_bones = max_bones
        self.pc_points = pc_points
        self.use_voxels = use_voxels

        # Load full dataset
        self.data = torch.load(pt_path, map_location="cpu", weights_only=False)

        # Load split indices
        self.indices = []
        with open(split_path) as f:
            for line in f:
                record = json.loads(line.strip())
                self.indices.append(record["idx"])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = self.data[self.indices[idx]]

        # --- Input: Point Cloud ---
        pc = d["pc"]  # [N, 3] or [N, 6] (xyz + normals)
        if pc.dim() == 1:
            pc = pc.unsqueeze(0)
        
        # Sample/pad to fixed size
        n_pts = pc.shape[0]
        if n_pts >= self.pc_points:
            # Random subsample
            perm = torch.randperm(n_pts)[: self.pc_points]
            pc_sampled = pc[perm]
        else:
            # Pad with zeros
            pad = torch.zeros(self.pc_points - n_pts, pc.shape[1])
            pc_sampled = torch.cat([pc, pad], dim=0)

        # Normalize point cloud to unit sphere
        centroid = pc_sampled[:n_pts].mean(dim=0) if n_pts > 0 else torch.zeros(pc.shape[1])
        pc_centered = pc_sampled - centroid
        scale = pc_centered[:min(n_pts, self.pc_points)].norm(dim=-1).max().clamp(min=1e-6)
        pc_norm = pc_centered / scale

        # --- Target: Joints ---
        joints = d["joints"]  # [J, 3]
        n_joints = int(d["joints_num"])
        
        joint_pos = torch.zeros(self.max_joints, 3)
        joint_active = torch.zeros(self.max_joints)
        if n_joints > 0:
            j = min(n_joints, self.max_joints)
            joint_pos[:j] = (joints[:j] - centroid[:3]) / scale
            joint_active[:j] = 1.0

        # --- Target: Connectivity ---
        conns = d["conns"]  # [J] raw parent indices for audit
        conns_np = conns.detach().cpu().numpy() if isinstance(conns, torch.Tensor) else np.asarray(conns)
        # Parent supervision fields come from normalized forest-valid targets.
        normalized_targets = extract_parent_targets(joints[:n_joints].detach().cpu().numpy(), conns_np[:n_joints])
        # Build edge list from parent connections
        edge_src = []
        edge_dst = []
        for child_idx in range(min(n_joints, self.max_joints)):
            parent_idx = int(conns[child_idx])
            if parent_idx >= 0 and parent_idx < n_joints and parent_idx != child_idx:
                edge_src.append(parent_idx)
                edge_dst.append(child_idx)

        n_edges = len(edge_src)

        # Adjacency matrix for edge prediction
        adj_matrix = torch.zeros(self.max_joints, self.max_joints)
        for s, d_idx in zip(edge_src, edge_dst):
            if s < self.max_joints and d_idx < self.max_joints:
                adj_matrix[s, d_idx] = 1.0
                adj_matrix[d_idx, s] = 1.0  # Undirected

        # --- Target: Bones ---
        bones = d["bones"]  # [B, 6] start_xyz, end_xyz
        n_bones = int(d["bones_num"])

        bone_starts = torch.zeros(self.max_bones, 3)
        bone_ends = torch.zeros(self.max_bones, 3)
        bone_lengths = torch.zeros(self.max_bones)
        bone_active = torch.zeros(self.max_bones)

        if n_bones > 0:
            b = min(n_bones, self.max_bones)
            bone_starts[:b] = (bones[:b, :3] - centroid[:3]) / scale
            bone_ends[:b] = (bones[:b, 3:6] - centroid[:3]) / scale
            bone_lengths[:b] = (bone_ends[:b] - bone_starts[:b]).norm(dim=-1)
            bone_active[:b] = 1.0

        # --- Target: Skinning Weights ---
        # skins_index: [V, K] joint indices per vertex
        # skins_weight: [V, K] weights per vertex
        skins_idx = d.get("skins_index", None)
        skins_w = d.get("skins_weight", None)

        # Create per-vertex skinning target for the sampled points
        # Simplified: for each sampled PC point, store top-K joint assignments
        skin_joints = torch.zeros(self.pc_points, 4, dtype=torch.long)  # top-4
        skin_weights = torch.zeros(self.pc_points, 4)

        if skins_idx is not None and skins_w is not None:
            n_skin_verts = skins_idx.shape[0]
            k_influences = min(skins_idx.shape[1], 4)
            
            # Map skinning from mesh vertices to sampled PC points
            if "mesh_skins_index" in d and "mesh_skins_weight" in d:
                mesh_skin_idx = d["mesh_skins_index"]
                mesh_skin_w = d["mesh_skins_weight"]
                n_mesh = mesh_skin_idx.shape[0]
                k_mesh = min(mesh_skin_idx.shape[1], 4)
                # Only use indices that are valid for the mesh skinning array
                if n_pts >= self.pc_points:
                    valid_perm = perm[perm < n_mesh][:self.pc_points]
                    sampled_skin_idx = mesh_skin_idx[valid_perm]
                    sampled_skin_w = mesh_skin_w[valid_perm]
                else:
                    n_use = min(n_pts, n_mesh)
                    sampled_skin_idx = mesh_skin_idx[:n_use]
                    sampled_skin_w = mesh_skin_w[:n_use]
                
                n_valid = min(sampled_skin_idx.shape[0], self.pc_points)
                skin_joints[:n_valid, :k_mesh] = sampled_skin_idx[:n_valid, :k_mesh].long()
                skin_weights[:n_valid, :k_mesh] = sampled_skin_w[:n_valid, :k_mesh].float()

        result = {
            "pc": pc_norm.float(),  # [P, 3+]
            "joint_pos": joint_pos.float(),  # [J, 3]
            "joint_active": joint_active.float(),  # [J]
            "n_joints": torch.tensor(n_joints, dtype=torch.long),
            "conns": torch.full((self.max_joints,), -1, dtype=torch.long),
            "adj_matrix": adj_matrix.float(),  # [J, J]
            "bone_starts": bone_starts.float(),  # [B, 3]
            "bone_ends": bone_ends.float(),  # [B, 3]
            "bone_lengths": bone_lengths.float(),  # [B]
            "bone_active": bone_active.float(),  # [B]
            "n_bones": torch.tensor(n_bones, dtype=torch.long),
            "parent_index": torch.full((self.max_joints,), -1, dtype=torch.long),
            "root_mask": torch.zeros(self.max_joints, dtype=torch.float32),
            "child_count": torch.zeros(self.max_joints, dtype=torch.long),
            "bone_vector_to_parent": torch.zeros(self.max_joints, 3, dtype=torch.float32),
            "depth_from_root": torch.zeros(self.max_joints, dtype=torch.long),
            "valid_parent_mask": torch.zeros(self.max_joints, dtype=torch.float32),
            "skin_joints": skin_joints.long(),  # [P, 4]
            "skin_weights": skin_weights.float(),  # [P, 4]
            "scale": scale.float(),
            "centroid": centroid[:3].float(),
        }

        if n_joints > 0:
            j = min(n_joints, self.max_joints)
            result["conns"][:j] = conns[:j].long()
            result["parent_index"][:j] = torch.from_numpy(normalized_targets.parent_index[:j]).long()
            result["root_mask"][:j] = torch.from_numpy(normalized_targets.root_mask[:j]).float()
            result["child_count"][:j] = torch.from_numpy(normalized_targets.child_count[:j]).long()
            result["bone_vector_to_parent"][:j] = torch.from_numpy(normalized_targets.bone_vector_to_parent[:j]).float()
            result["depth_from_root"][:j] = torch.from_numpy(normalized_targets.depth_from_root[:j]).long()
            result["valid_parent_mask"][:j] = torch.from_numpy(normalized_targets.valid_parent_mask[:j]).float()

        if self.use_voxels and "vox" in d:
            vox = d["vox"]  # [D, H, W] binary voxel grid
            if vox.dim() == 3:
                vox = vox.unsqueeze(0)  # [1, D, H, W]
            result["voxels"] = vox.float()

        return result
