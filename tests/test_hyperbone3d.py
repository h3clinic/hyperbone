"""Tests for HyperBone3D model, dataset, joint map, and training."""
import pytest
import json
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch

from hyperbone.pose3d.joint_map import (
    QUADRUPED_JOINTS, QUADRUPED_BONES, NUM_JOINTS,
    remap_joints_to_canonical, auto_map_joints, FOX_JOINT_MAP,
)
from hyperbone.models.hyperbone3d import HyperBone3D, HyperBone3DLoss


# ─── Joint Map Tests ──────────────────────────────────────────────────────────

class TestJointMap:
    def test_canonical_joint_count(self):
        assert NUM_JOINTS == 19

    def test_bone_edges_valid(self):
        for pi, ci in QUADRUPED_BONES:
            assert 0 <= pi < NUM_JOINTS
            assert 0 <= ci < NUM_JOINTS
            assert pi != ci

    def test_fox_map_covers_canonical(self):
        mapped = set(FOX_JOINT_MAP.values())
        # Should cover all 19 canonical joints
        assert mapped == set(QUADRUPED_JOINTS)

    def test_remap_fox_joints(self):
        source_joints = [
            {"name": "b_Root_00", "world_xyz": [0, 0, 0], "visible": True},
            {"name": "b_Head_05", "world_xyz": [0, 5, 0], "visible": True},
            {"name": "b_Tail01_012", "world_xyz": [0, -3, 0], "visible": True},
        ]
        xyz, vis = remap_joints_to_canonical(source_joints, "Fox.glb")
        # Root should be mapped
        assert vis[0] is True  # root
        assert xyz[0] == (0, 0, 0)
        # Head (index 4)
        assert vis[4] is True
        assert xyz[4] == (0, 5, 0)
        # Tail_1 (index 5)
        assert vis[5] is True

    def test_missing_joints_get_zero(self):
        # Only provide root
        source_joints = [
            {"name": "b_Root_00", "world_xyz": [1, 2, 3], "visible": True},
        ]
        xyz, vis = remap_joints_to_canonical(source_joints, "Fox.glb")
        # Root visible
        assert vis[0] is True
        assert xyz[0] == (1, 2, 3)
        # Others invisible with zero
        assert vis[4] is False  # head
        assert xyz[4] == (0.0, 0.0, 0.0)

    def test_auto_map_generic_names(self):
        names = ["root", "spine", "neck", "head", "leftleg", "rightleg"]
        mapping = auto_map_joints(names)
        assert "root" in mapping
        assert mapping["root"] == "root"


# ─── Model Tests ──────────────────────────────────────────────────────────────

class TestHyperBone3DModel:
    def test_forward_shape(self):
        model = HyperBone3D(num_joints=19, input_channels=5, base_channels=32, hidden_dim=64)
        x = torch.randn(2, 5, 256, 256)
        out = model(x)
        assert out["joint_xyz_canonical"].shape == (2, 19, 3)
        assert out["joint_visibility_logits"].shape == (2, 19)

    def test_forward_small_input(self):
        model = HyperBone3D(num_joints=19, input_channels=5, base_channels=32, hidden_dim=64)
        x = torch.randn(1, 5, 64, 64)
        out = model(x)
        assert out["joint_xyz_canonical"].shape == (1, 19, 3)

    def test_parameter_count(self):
        model = HyperBone3D(base_channels=32, hidden_dim=64)
        params = model.count_parameters()
        assert params > 0
        assert params < 5_000_000  # Should be < 5M params

    def test_gradients_flow(self):
        model = HyperBone3D(base_channels=32, hidden_dim=64)
        x = torch.randn(2, 5, 128, 128, requires_grad=True)
        out = model(x)
        loss = out["joint_xyz_canonical"].sum()
        loss.backward()
        # Check gradients exist on first conv layer
        for p in model.encoder[0].conv[0].parameters():
            assert p.grad is not None
            break


# ─── Loss Tests ───────────────────────────────────────────────────────────────

class TestHyperBone3DLoss:
    def test_loss_computes_without_nan(self):
        bone_edges = torch.tensor(QUADRUPED_BONES, dtype=torch.long)
        criterion = HyperBone3DLoss(bone_edges=bone_edges)

        pred_xyz = torch.randn(4, 19, 3)
        pred_vis = torch.randn(4, 19)
        gt_xyz = torch.randn(4, 19, 3)
        gt_vis = torch.ones(4, 19)

        losses = criterion(pred_xyz, pred_vis, gt_xyz, gt_vis)
        assert not torch.isnan(losses["total"])
        assert not torch.isnan(losses["pos_loss"])
        assert not torch.isnan(losses["vis_loss"])
        assert not torch.isnan(losses["bone_loss"])
        assert losses["total"].item() > 0

    def test_zero_loss_on_perfect_prediction(self):
        bone_edges = torch.tensor(QUADRUPED_BONES, dtype=torch.long)
        criterion = HyperBone3DLoss(bone_edges=bone_edges)

        gt_xyz = torch.randn(2, 19, 3)
        gt_vis = torch.ones(2, 19)
        pred_xyz = gt_xyz.clone()
        # Perfect vis: gt_vis=1 should match large positive logits
        pred_vis = torch.ones(2, 19) * 5.0  # sigmoid(5)≈0.99

        losses = criterion(pred_xyz, pred_vis, gt_xyz, gt_vis)
        assert losses["pos_loss"].item() < 1e-6
        assert losses["bone_loss"].item() < 1e-6

    def test_loss_with_invisible_joints(self):
        criterion = HyperBone3DLoss()

        pred_xyz = torch.randn(2, 19, 3)
        pred_vis = torch.randn(2, 19)
        gt_xyz = torch.randn(2, 19, 3)
        gt_vis = torch.zeros(2, 19)  # All invisible

        losses = criterion(pred_xyz, pred_vis, gt_xyz, gt_vis)
        # Position loss should be ~0 since no visible joints
        assert losses["pos_loss"].item() < 1e-6


# ─── Training Smoke Test ──────────────────────────────────────────────────────

class TestTrainingSmoke:
    def test_overfit_two_samples(self):
        """Model should be able to overfit 2 synthetic samples."""
        torch.manual_seed(42)
        model = HyperBone3D(num_joints=19, input_channels=5, base_channels=32, hidden_dim=64)
        bone_edges = torch.tensor(QUADRUPED_BONES, dtype=torch.long)
        criterion = HyperBone3DLoss(bone_edges=bone_edges)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create 2 synthetic samples
        x1 = torch.randn(1, 5, 64, 64)
        x2 = torch.randn(1, 5, 64, 64)
        gt_xyz1 = torch.randn(1, 19, 3) * 0.3
        gt_xyz2 = torch.randn(1, 19, 3) * 0.3
        gt_vis = torch.ones(1, 19)

        initial_loss = None
        final_loss = None

        for step in range(50):
            model.train()
            for x, gt in [(x1, gt_xyz1), (x2, gt_xyz2)]:
                pred = model(x)
                losses = criterion(pred["joint_xyz_canonical"],
                                   pred["joint_visibility_logits"],
                                   gt, gt_vis)
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

                if initial_loss is None:
                    initial_loss = losses["total"].item()
                final_loss = losses["total"].item()

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, \
            f"Model did not overfit: {initial_loss:.4f} -> {final_loss:.4f}"
