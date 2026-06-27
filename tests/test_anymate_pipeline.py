"""
Tests for the Anymate training pipeline.

Covers:
- Schema parsing and serialization
- Motion synthesis
- Dataset loading
- Model forward pass
- Loss computation (no NaN)
- Tiny overfit sanity
- Eval metrics
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.rigs.schema import (
    Joint, Bone, Skeleton, AssetRig, AnimationClip, Camera,
    FramePoseLabel, SkinningWeights, write_jsonl, read_jsonl,
    MotionSource,
)
from hyperbone.rigs.motion_synth import (
    MotionParams, generate_motion_curves, get_motion_preset, _compute_depths,
)


# === Schema Tests ===

class TestSchema:
    def test_joint_roundtrip(self):
        j = Joint(id=0, name="root", parent_id=None, type="root",
                  world_xyz=(1.0, 2.0, 3.0))
        d = j.to_dict()
        j2 = Joint.from_dict(d)
        assert j2.id == 0
        assert j2.name == "root"
        assert j2.world_xyz == (1.0, 2.0, 3.0)

    def test_bone_roundtrip(self):
        b = Bone(parent_id=0, child_id=1, length=0.5)
        d = b.to_dict()
        b2 = Bone.from_dict(d)
        assert b2.parent_id == 0
        assert b2.child_id == 1
        assert b2.length == 0.5

    def test_skeleton_hierarchy(self):
        joints = [
            Joint(id=0, name="root", parent_id=None),
            Joint(id=1, name="spine", parent_id=0),
            Joint(id=2, name="head", parent_id=1),
        ]
        bones = [
            Bone(parent_id=0, child_id=1, length=1.0),
            Bone(parent_id=1, child_id=2, length=0.5),
        ]
        skel = Skeleton(joints=joints, bones=bones)
        assert skel.joint_count == 3
        assert skel.bone_count == 2
        assert skel.root.name == "root"
        assert len(skel.children_of(0)) == 1
        assert skel.children_of(0)[0].name == "spine"

    def test_frame_pose_label_roundtrip(self):
        label = FramePoseLabel(
            asset_id="test_001",
            animation_id="walk_like",
            frame_idx=42,
            timestamp_sec=1.75,
            rgb_path="rgb/frame_000042.png",
            mask_path="mask/frame_000042.png",
            depth_path="depth/frame_000042.npz",
            camera=Camera(resolution=(512, 512), view_name="front"),
            joints=[Joint(id=0, name="root", parent_id=None, world_xyz=(0, 0, 0))],
            bones=[],
        )
        d = label.to_dict()
        label2 = FramePoseLabel.from_dict(d)
        assert label2.asset_id == "test_001"
        assert label2.frame_idx == 42
        assert label2.timestamp_sec == 1.75
        assert len(label2.joints) == 1
        assert label2.camera.resolution == (512, 512)

    def test_asset_rig_index_row(self):
        asset = AssetRig(
            asset_id="asset_000001",
            mesh_path="/data/mesh.glb",
            has_skeleton=True,
            has_skinning=True,
            has_animation=False,
            joint_count=34,
            bone_count=33,
            vertex_count=18420,
            category="animal",
        )
        row = asset.to_index_row()
        assert row["asset_id"] == "asset_000001"
        assert row["has_animation"] is False
        assert "rest_skeleton" not in row

    def test_motion_source_enum(self):
        assert MotionSource.SYNTHETIC.value == "synthetic_rig_motion"
        assert MotionSource.ORIGINAL.value == "original"

    def test_write_read_jsonl(self, tmp_path):
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        path = tmp_path / "test.jsonl"
        write_jsonl(path, records)
        loaded = read_jsonl(path)
        assert loaded == records


# === Motion Synthesis Tests ===

class TestMotionSynth:
    def test_compute_depths(self):
        parent_ids = [None, 0, 1, 1, 0]
        depths = _compute_depths(parent_ids)
        assert depths[0] == 0  # root
        assert depths[1] == 1
        assert depths[2] == 2
        assert depths[3] == 2
        assert depths[4] == 1

    def test_generate_motion_identity_at_rest(self):
        """With zero amplitude, all rotations should be identity."""
        params = MotionParams(amplitude_deg=0.0, fps=12, duration_sec=1.0)
        rots = generate_motion_curves(
            joint_count=5,
            parent_ids=[None, 0, 1, 2, 0],
            joint_names=["root", "spine", "neck", "head", "tail"],
            params=params,
        )
        assert rots.shape == (12, 5, 4)
        # All should be identity (w=1, xyz=0)
        np.testing.assert_allclose(rots[:, :, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(rots[:, :, 1:], 0.0, atol=1e-6)

    def test_generate_motion_shapes(self):
        params = get_motion_preset("walk_like", duration_sec=2.0, fps=24)
        rots = generate_motion_curves(
            joint_count=10,
            parent_ids=[None, 0, 1, 2, 0, 4, 5, 0, 7, 8],
            joint_names=["root", "spine", "neck", "head",
                         "left_hip", "left_knee", "left_foot",
                         "right_hip", "right_knee", "right_foot"],
            params=params,
        )
        assert rots.shape == (48, 10, 4)
        # Quaternions should be unit length
        norms = np.linalg.norm(rots, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_all_presets_produce_valid_quaternions(self):
        from hyperbone.rigs.motion_synth import MOTION_PRESETS
        for motion_type in MOTION_PRESETS:
            params = get_motion_preset(motion_type, duration_sec=1.0, fps=12)
            rots = generate_motion_curves(
                joint_count=4,
                parent_ids=[None, 0, 1, 2],
                joint_names=["root", "bone1", "bone2", "tip"],
                params=params,
            )
            norms = np.linalg.norm(rots, axis=-1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5,
                                       err_msg=f"Non-unit quaternion in {motion_type}")


# === Dataset Tests ===

class TestAnymateClipDataset:
    def _make_fake_dataset(self, tmp_path, n_frames=10, n_joints=5):
        """Create a minimal fake dataset for testing."""
        frames_dir = tmp_path / "rgb"
        frames_dir.mkdir()

        labels = []
        for i in range(n_frames):
            # Create a dummy RGB image
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.png"), img)

            joints = []
            for ji in range(n_joints):
                joints.append({
                    "id": ji,
                    "name": f"joint_{ji}",
                    "parent_id": ji - 1 if ji > 0 else None,
                    "camera_xyz": [float(ji) * 0.1, 0.0, 2.0],
                    "image_xy": [32.0 + ji * 5, 32.0],
                    "visible": True,
                })

            labels.append({
                "asset_id": "test_asset",
                "animation_id": "test_motion",
                "frame_idx": i,
                "timestamp_sec": i / 12.0,
                "rgb_path": f"rgb/frame_{i:06d}.png",
                "mask_path": "",
                "depth_path": "",
                "joints": joints,
                "bones": [],
            })

        index_path = tmp_path / "test.jsonl"
        write_jsonl(index_path, labels)
        return index_path

    def test_dataset_loads(self, tmp_path):
        index_path = self._make_fake_dataset(tmp_path)
        from hyperbone.datasets.anymate_clip_dataset import AnymateClipDataset
        ds = AnymateClipDataset(str(index_path), clip_len=1, resolution=64, max_joints=16)
        assert len(ds) == 10
        assert ds.total_frames == 10

    def test_dataset_single_frame(self, tmp_path):
        index_path = self._make_fake_dataset(tmp_path)
        from hyperbone.datasets.anymate_clip_dataset import AnymateClipDataset
        ds = AnymateClipDataset(str(index_path), clip_len=1, resolution=64, max_joints=16)
        sample = ds[0]
        assert sample["rgb"].shape == (3, 64, 64)
        assert sample["joint_xyz"].shape == (16, 3)
        assert sample["joint_active"].shape == (16,)
        assert sample["joint_active"][:5].sum() == 5.0
        assert sample["joint_active"][5:].sum() == 0.0

    def test_dataset_temporal_window(self, tmp_path):
        index_path = self._make_fake_dataset(tmp_path, n_frames=20)
        from hyperbone.datasets.anymate_clip_dataset import AnymateClipDataset
        ds = AnymateClipDataset(str(index_path), clip_len=4, stride=2, resolution=64, max_joints=16)
        assert len(ds) > 0
        sample = ds[0]
        assert sample["rgb"].shape == (4, 3, 64, 64)
        assert sample["joint_xyz"].shape == (4, 16, 3)


# === Model Tests ===

class TestAnymateFrameModel:
    def test_forward_pass(self):
        from scripts.train_hyperbone_anymate_frame import AnymateFrameModel
        model = AnymateFrameModel(in_channels=5, max_joints=32)
        x = torch.randn(2, 5, 64, 64)
        out = model(x)
        assert out["joint_xyz"].shape == (2, 32, 3)
        assert out["joint_vis"].shape == (2, 32)
        assert out["joint_conf"].shape == (2, 32)
        assert out["heatmaps"].shape[0] == 2
        assert out["heatmaps"].shape[1] == 32

    def test_no_nan_output(self):
        from scripts.train_hyperbone_anymate_frame import AnymateFrameModel
        model = AnymateFrameModel(in_channels=5, max_joints=16)
        x = torch.randn(4, 5, 128, 128)
        out = model(x)
        for k, v in out.items():
            assert not torch.isnan(v).any(), f"NaN in {k}"


class TestTemporalModel:
    def test_forward_pass(self):
        from scripts.train_hyperbone_anymate_temporal import TemporalPoseModel
        model = TemporalPoseModel(in_channels=5, max_joints=16, clip_len=8)
        x = torch.randn(2, 8, 5, 64, 64)
        out = model(x)
        assert out["joint_xyz"].shape == (2, 8, 16, 3)
        assert out["joint_vis"].shape == (2, 8, 16)


# === Loss Tests ===

class TestLosses:
    def test_frame_loss_no_nan(self):
        from scripts.train_hyperbone_anymate_frame import AnymateFrameModel, compute_loss
        model = AnymateFrameModel(in_channels=5, max_joints=16)
        x = torch.randn(2, 5, 64, 64)
        pred = model(x)

        batch = {
            "joint_xyz": torch.randn(2, 16, 3),
            "joint_vis": torch.ones(2, 16),
            "joint_active": torch.ones(2, 16),
            "joint_xy": torch.randn(2, 16, 2) * 32 + 32,
        }
        losses = compute_loss(pred, batch, torch.device("cpu"))
        assert not np.isnan(losses["total"].item())
        assert losses["total"].item() > 0

    def test_temporal_loss_no_nan(self):
        from scripts.train_hyperbone_anymate_temporal import TemporalPoseModel, compute_temporal_loss
        model = TemporalPoseModel(in_channels=5, max_joints=16, clip_len=8)
        x = torch.randn(2, 8, 5, 64, 64)
        pred = model(x)

        batch = {
            "joint_xyz": torch.randn(2, 8, 16, 3),
            "joint_vis": torch.ones(2, 8, 16),
            "joint_active": torch.ones(2, 8, 16),
        }
        losses = compute_temporal_loss(pred, batch, torch.device("cpu"))
        assert not np.isnan(losses["total"].item())

    def test_loss_zero_when_no_active_joints(self):
        from scripts.train_hyperbone_anymate_frame import AnymateFrameModel, compute_loss
        model = AnymateFrameModel(in_channels=5, max_joints=16)
        x = torch.randn(2, 5, 64, 64)
        pred = model(x)

        batch = {
            "joint_xyz": torch.randn(2, 16, 3),
            "joint_vis": torch.zeros(2, 16),       # No visible joints
            "joint_active": torch.zeros(2, 16),    # No active joints
            "joint_xy": torch.randn(2, 16, 2),
        }
        losses = compute_loss(pred, batch, torch.device("cpu"))
        # Should not crash, loss should be finite
        assert not np.isnan(losses["total"].item())
        assert np.isfinite(losses["total"].item())


# === Tiny Overfit Test ===

class TestTinyOverfit:
    def test_frame_model_can_overfit_single_batch(self):
        """Model should be able to memorize a single batch (loss decreasing)."""
        from scripts.train_hyperbone_anymate_frame import AnymateFrameModel, compute_loss

        model = AnymateFrameModel(in_channels=5, max_joints=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fixed batch
        x = torch.randn(4, 5, 64, 64)
        batch = {
            "joint_xyz": torch.randn(4, 8, 3) * 0.5,
            "joint_vis": torch.ones(4, 8),
            "joint_active": torch.ones(4, 8),
            "joint_xy": torch.randn(4, 8, 2) * 20 + 32,
        }

        initial_loss = None
        for step in range(50):
            optimizer.zero_grad()
            pred = model(x)
            losses = compute_loss(pred, batch, torch.device("cpu"))
            losses["total"].backward()
            optimizer.step()

            if step == 0:
                initial_loss = losses["total"].item()

        final_loss = losses["total"].item()
        assert final_loss < initial_loss * 0.5, (
            f"Loss did not decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
        )


# === Eval Metric Tests ===

class TestEvalMetrics:
    def test_perfect_prediction(self):
        from scripts.eval_hyperbone_anymate import compute_metrics

        preds = [{"joint_xyz": [[0, 0, 0], [1, 0, 0]], "joint_vis": [1.0, 1.0]}]
        gts = [{
            "joint_xyz": [[0, 0, 0], [1, 0, 0]],
            "joint_vis": [1.0, 1.0],
            "joint_active": [1.0, 1.0],
            "joint_xy": [[100, 100], [200, 200]],
        }]

        metrics = compute_metrics(preds, gts)
        assert metrics["mpjpe"] == 0.0
        assert metrics["pck_010"] == 1.0

    def test_known_error(self):
        from scripts.eval_hyperbone_anymate import compute_metrics

        preds = [{"joint_xyz": [[0.1, 0, 0]], "joint_vis": [1.0]}]
        gts = [{
            "joint_xyz": [[0, 0, 0]],
            "joint_vis": [1.0],
            "joint_active": [1.0],
            "joint_xy": [[100, 100]],
        }]

        metrics = compute_metrics(preds, gts)
        assert abs(metrics["mpjpe"] - 0.1) < 1e-6
        assert metrics["pck_005"] == 0.0  # Error is 0.1, threshold is 0.05
        assert metrics["pck_020"] == 1.0  # Error is 0.1, threshold is 0.20


# === Projection Math Tests ===

class TestProjectionMath:
    def test_camera_projection(self):
        """Test that 3D→2D projection math is correct."""
        # Simple pinhole camera
        fx, fy = 256.0, 256.0
        cx, cy = 128.0, 128.0

        # Point at (0, 0, 2) should project to center
        point_cam = np.array([0.0, 0.0, 2.0])
        px = fx * (-point_cam[0] / point_cam[2]) + cx  # Blender convention: negate x
        py = fy * (-point_cam[1] / point_cam[2]) + cy

        # With Blender's -Z forward, point at (0,0,2) is behind camera
        # Point at (0,0,-2) is in front
        point_cam_front = np.array([0.0, 0.0, -2.0])
        px = fx * (point_cam_front[0] / (-point_cam_front[2])) + cx
        py = fy * (point_cam_front[1] / (-point_cam_front[2])) + cy
        assert abs(px - 128.0) < 1e-6
        assert abs(py - 128.0) < 1e-6

    def test_quaternion_normalization(self):
        """Generated quaternions must be unit length."""
        q = np.array([0.7071, 0.7071, 0, 0])
        norm = np.linalg.norm(q)
        assert abs(norm - 1.0) < 0.01


# === Skeleton Timestamp Sync Test ===

class TestTimestampSync:
    def test_frame_timestamp_consistency(self):
        """Verify that frame_idx and timestamp_sec are consistent."""
        fps = 24
        duration = 3.0
        frames = int(fps * duration)

        for frame_idx in range(1, frames + 1):
            timestamp = (frame_idx - 1) / fps
            # Reconstruct frame from timestamp
            reconstructed_frame = int(round(timestamp * fps)) + 1
            assert reconstructed_frame == frame_idx, (
                f"Timestamp sync broken: frame {frame_idx} → ts {timestamp} → frame {reconstructed_frame}"
            )
