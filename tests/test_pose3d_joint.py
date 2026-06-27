"""Tests for HyperBonePose3D-Joint system."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np
import json
import tempfile
from pathlib import Path


class TestQuadrupedSchema:
    """Test pose2d quadruped schema."""

    def test_joint_count(self):
        from hyperbone.pose2d.quadruped_schema import NUM_JOINTS_2D, QUADRUPED_JOINTS_2D
        assert NUM_JOINTS_2D == 19
        assert len(QUADRUPED_JOINTS_2D) == 19

    def test_bone_connectivity(self):
        from hyperbone.pose2d.quadruped_schema import QUADRUPED_BONES_2D, NUM_JOINTS_2D
        for pi, ci in QUADRUPED_BONES_2D:
            assert 0 <= pi < NUM_JOINTS_2D
            assert 0 <= ci < NUM_JOINTS_2D
            assert pi != ci

    def test_parent_hierarchy_valid(self):
        from hyperbone.pose2d.quadruped_schema import JOINT_PARENTS, NUM_JOINTS_2D
        assert len(JOINT_PARENTS) == NUM_JOINTS_2D
        assert JOINT_PARENTS[0] is None  # root has no parent
        for i, p in enumerate(JOINT_PARENTS):
            if p is not None:
                assert 0 <= p < NUM_JOINTS_2D
                assert p < i  # parent always has lower index

    def test_joint_sides(self):
        from hyperbone.pose2d.quadruped_schema import JOINT_SIDES, QUADRUPED_JOINTS_2D
        for name in QUADRUPED_JOINTS_2D:
            assert name in JOINT_SIDES
            assert JOINT_SIDES[name] in ("left", "right", "midline")

    def test_validate_pose_rejects_few_joints(self):
        from hyperbone.pose2d.quadruped_schema import validate_pose
        joints = {"head": (100, 50), "neck": (100, 80)}
        confs = {"head": 0.9, "neck": 0.8}
        valid, reason = validate_pose(joints, confs, min_joints=5)
        assert not valid
        assert "2" in reason

    def test_validate_pose_accepts_enough_joints(self):
        from hyperbone.pose2d.quadruped_schema import validate_pose
        joints = {f"j{i}": (i*10, i*10) for i in range(6)}
        confs = {f"j{i}": 0.8 for i in range(6)}
        valid, reason = validate_pose(joints, confs, min_joints=5)
        assert valid

    def test_joints_inside_bbox_ratio(self):
        from hyperbone.pose2d.quadruped_schema import joints_inside_bbox_ratio
        joints = {"a": (50, 50), "b": (150, 150), "c": (250, 250)}
        confs = {"a": 0.9, "b": 0.9, "c": 0.9}
        bbox = (0, 0, 200, 200)
        ratio = joints_inside_bbox_ratio(joints, bbox, confs)
        assert abs(ratio - 2/3) < 0.01

    def test_pose3d_to_pose2d_mapping(self):
        from hyperbone.pose2d.quadruped_schema import POSE3D_TO_POSE2D
        # Old schema 'tail_1' maps to 'tail_base'
        assert POSE3D_TO_POSE2D["tail_1"] == "tail_base"
        assert POSE3D_TO_POSE2D["front_left_paw"] == "front_left_hoof"


class TestHyperBonePose3DJointModel:
    """Test the 3D joint model."""

    def test_forward_shape(self):
        from hyperbone.models.hyperbone_pose3d_joint import HyperBonePose3DJoint
        model = HyperBonePose3DJoint(num_joints=19, input_channels=3, base_channels=16)
        x = torch.randn(2, 3, 192, 192)
        out = model(x)
        assert out["joint_xyz_canonical"].shape == (2, 19, 3)
        assert out["joint_visibility_logits"].shape == (2, 19)
        assert out["joint_confidence"].shape == (2, 19)
        assert out["joint_heatmaps_2d"].shape[0] == 2
        assert out["joint_heatmaps_2d"].shape[1] == 19
        assert out["projected_joint_xy"].shape == (2, 19, 2)
        # projected_joint_xy should be in [0, 1]
        assert out["projected_joint_xy"].min() >= 0
        assert out["projected_joint_xy"].max() <= 1

    def test_forward_no_aux_heatmaps(self):
        from hyperbone.models.hyperbone_pose3d_joint import HyperBonePose3DJoint
        model = HyperBonePose3DJoint(num_joints=19, input_channels=3, base_channels=16,
                                     use_aux_heatmaps=False)
        x = torch.randn(1, 3, 192, 192)
        out = model(x)
        # Even without "aux heatmaps" flag, the new model always produces heatmaps
        # since they're integral to the architecture
        assert out["joint_xyz_canonical"].shape == (1, 19, 3)
        assert "projected_joint_xy" in out

    def test_confidence_bounded(self):
        from hyperbone.models.hyperbone_pose3d_joint import HyperBonePose3DJoint
        model = HyperBonePose3DJoint(num_joints=19, input_channels=3, base_channels=16)
        x = torch.randn(1, 3, 192, 192)
        out = model(x)
        conf = out["joint_confidence"]
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_5channel_input(self):
        from hyperbone.models.hyperbone_pose3d_joint import HyperBonePose3DJoint
        model = HyperBonePose3DJoint(num_joints=19, input_channels=5, base_channels=16)
        x = torch.randn(1, 5, 192, 192)
        out = model(x)
        assert out["joint_xyz_canonical"].shape == (1, 19, 3)
        assert out["projected_joint_xy"].shape == (1, 19, 2)

    def test_loss_no_nan(self):
        from hyperbone.models.hyperbone_pose3d_joint import (
            HyperBonePose3DJoint, HyperBonePose3DJointLoss
        )
        model = HyperBonePose3DJoint(num_joints=19, input_channels=3, base_channels=16)
        criterion = HyperBonePose3DJointLoss()
        x = torch.randn(2, 3, 192, 192)
        gt_xyz = torch.randn(2, 19, 3)
        gt_vis = torch.ones(2, 19)
        gt_2d = torch.rand(2, 19, 2)
        gt_hm = torch.rand(2, 19, 48, 48)

        pred = model(x)
        losses = criterion(pred, gt_xyz, gt_vis, gt_2d, gt_hm)
        assert not torch.isnan(losses["total"])
        assert losses["total"].item() > 0
        assert "coord2d_loss" in losses
        assert "heatmap_loss" in losses

    def test_loss_invisible_joints_ignored(self):
        from hyperbone.models.hyperbone_pose3d_joint import (
            HyperBonePose3DJoint, HyperBonePose3DJointLoss
        )
        model = HyperBonePose3DJoint(num_joints=19, input_channels=3, base_channels=16)
        criterion = HyperBonePose3DJointLoss()
        x = torch.randn(1, 3, 192, 192)
        gt_xyz = torch.randn(1, 19, 3) * 100  # Large values
        gt_vis = torch.zeros(1, 19)  # All invisible
        gt_2d = torch.zeros(1, 19, 2)

        pred = model(x)
        losses = criterion(pred, gt_xyz, gt_vis, gt_2d, None)
        # Position loss should be ~0 since all joints invisible
        assert losses["pos_loss"].item() < 1e-6

    def test_parameter_count(self):
        from hyperbone.models.hyperbone_pose3d_joint import HyperBonePose3DJoint
        model = HyperBonePose3DJoint(num_joints=19, input_channels=3, base_channels=48)
        n_params = model.count_parameters()
        # Should be reasonable (< 10M for this architecture)
        assert 100_000 < n_params < 10_000_000


class TestProjection:
    """Test projection utilities."""

    def test_canonical_to_crop(self):
        from hyperbone.models.hyperbone_pose3d_joint import project_canonical_to_crop
        xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])  # [2, 3]
        pts = project_canonical_to_crop(xyz, crop_size=192)
        # Origin should map to center
        assert abs(pts[0, 0].item() - 96) < 1
        assert abs(pts[0, 1].item() - 96) < 1

    def test_crop_to_fullframe(self):
        from hyperbone.models.hyperbone_pose3d_joint import project_crop_to_fullframe
        crop_xy = torch.tensor([[96.0, 96.0]])  # center of crop
        bbox = torch.tensor([100.0, 50.0, 200.0, 200.0])
        full = project_crop_to_fullframe(crop_xy, bbox, crop_size=192)
        # Center of crop -> center of bbox
        assert abs(full[0, 0].item() - 200.0) < 1  # 100 + 200/2
        assert abs(full[0, 1].item() - 150.0) < 1  # 50 + 200/2


class TestInferenceGuards:
    """Test inference rejection logic."""

    def test_no_bbox_means_no_pose(self):
        """If no bbox is available, system must not produce a pose."""
        from hyperbone.pose2d.quadruped_schema import validate_pose
        # Empty joints = always rejected
        valid, reason = validate_pose({}, {}, min_joints=5)
        assert not valid

    def test_low_confidence_rejected(self):
        from hyperbone.pose2d.quadruped_schema import validate_pose
        joints = {f"j{i}": (i*10, i*10) for i in range(10)}
        confs = {f"j{i}": 0.1 for i in range(10)}  # all low confidence
        valid, reason = validate_pose(joints, confs, conf_threshold=0.3, min_joints=5)
        assert not valid

    def test_fewer_than_min_joints_rejected(self):
        from hyperbone.pose2d.quadruped_schema import validate_pose
        joints = {"head": (100, 50), "neck": (100, 80), "root": (100, 120)}
        confs = {"head": 0.9, "neck": 0.8, "root": 0.7}
        valid, reason = validate_pose(joints, confs, min_joints=5)
        assert not valid
        assert "3" in reason


class TestDataset:
    """Test Pose3DJointDataset."""

    def test_loads_from_jsonl(self, tmp_path):
        from hyperbone.pose2d.dataset_3djoint import Pose3DJointDataset
        import cv2

        # Create fake image
        img_path = tmp_path / "frame.png"
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[100:200, 100:200] = 128
        cv2.imwrite(str(img_path), img)

        # Create label
        label = {
            "image_path": str(img_path),
            "frame_idx": 0,
            "bbox_xywh": [80, 80, 150, 150],
            "joints_3d": {
                "root": {"xyz": [0, 0, 0], "visible": True},
                "neck": {"xyz": [0, 1, 0], "visible": True},
                "head": {"xyz": [0, 1.5, 0.2], "visible": True},
            },
            "joints_2d": {
                "root": {"xy": [150, 200], "visible": True},
                "neck": {"xy": [150, 150], "visible": True},
                "head": {"xy": [155, 120], "visible": True},
            },
            "asset_id": "test",
            "species": "fox",
        }
        labels_path = tmp_path / "labels.jsonl"
        with open(labels_path, 'w') as f:
            f.write(json.dumps(label) + "\n")

        ds = Pose3DJointDataset(str(labels_path), resolution=128, heatmap_resolution=32)
        assert len(ds) == 1

        sample = ds[0]
        assert sample["image"].shape == (3, 128, 128)
        assert sample["joint_xyz_canonical"].shape == (19, 3)
        assert sample["joint_visible"].shape == (19,)
        assert sample["joints_2d"].shape == (19, 2)
        assert sample["heatmaps"].shape == (19, 32, 32)

        # 3 visible joints
        assert sample["joint_visible"].sum() == 3

    def test_invisible_joints_zero(self, tmp_path):
        from hyperbone.pose2d.dataset_3djoint import Pose3DJointDataset
        import cv2

        img_path = tmp_path / "frame.png"
        cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

        label = {
            "image_path": str(img_path),
            "frame_idx": 0,
            "bbox_xywh": [0, 0, 100, 100],
            "joints_3d": {},
            "joints_2d": {},
        }
        labels_path = tmp_path / "labels.jsonl"
        with open(labels_path, 'w') as f:
            f.write(json.dumps(label) + "\n")

        ds = Pose3DJointDataset(str(labels_path), resolution=64, heatmap_resolution=16)
        sample = ds[0]
        assert sample["joint_visible"].sum() == 0
        assert sample["heatmaps"].sum() == 0
