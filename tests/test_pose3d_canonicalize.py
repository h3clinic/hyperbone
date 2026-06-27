"""Tests for hyperbone.pose3d.canonicalize — skeleton normalization."""
import pytest
import numpy as np
from hyperbone.pose3d.schema import Joint3D, Bone3D, PoseFrame3D
from hyperbone.pose3d.canonicalize import (
    find_root_joint,
    compute_torso_length,
    canonicalize_frame,
)


def _make_frame(joints, bones=None):
    return PoseFrame3D(
        asset_id="test", frame_idx=0, timestamp_sec=0.0,
        animation_name="Walk", joints=joints, bones=bones or [],
    )


class TestFindRootJoint:
    def test_finds_by_type(self):
        joints = [
            Joint3D(id=0, name="spine", parent_id=None, type="spine"),
            Joint3D(id=1, name="pelvis", parent_id=None, type="root"),
        ]
        frame = _make_frame(joints)
        root = find_root_joint(frame)
        assert root.id == 1

    def test_finds_by_name(self):
        joints = [
            Joint3D(id=0, name="b_Root", parent_id=None, type="unknown"),
            Joint3D(id=1, name="b_Spine", parent_id=0, type="unknown"),
        ]
        frame = _make_frame(joints)
        root = find_root_joint(frame)
        assert root.id == 0

    def test_fallback_to_parentless(self):
        joints = [
            Joint3D(id=0, name="bone_A", parent_id=None, type="unknown"),
            Joint3D(id=1, name="bone_B", parent_id=0, type="unknown"),
        ]
        frame = _make_frame(joints)
        root = find_root_joint(frame)
        assert root.id == 0

    def test_empty_joints(self):
        frame = _make_frame([])
        root = find_root_joint(frame)
        assert root is None


class TestTorsoLength:
    def test_root_to_neck(self):
        joints = [
            Joint3D(id=0, name="root", parent_id=None, type="root",
                    world_xyz=(0, 0, 0)),
            Joint3D(id=1, name="neck", parent_id=0, type="neck",
                    world_xyz=(0, 5, 0)),
        ]
        frame = _make_frame(joints)
        length = compute_torso_length(frame)
        assert abs(length - 5.0) < 0.01

    def test_fallback_max_distance(self):
        joints = [
            Joint3D(id=0, name="base", parent_id=None, type="root",
                    world_xyz=(0, 0, 0)),
            Joint3D(id=1, name="tip", parent_id=0, type="unknown",
                    world_xyz=(3, 4, 0)),
        ]
        frame = _make_frame(joints)
        length = compute_torso_length(frame)
        assert abs(length - 5.0) < 0.01  # 3-4-5 triangle

    def test_empty_returns_one(self):
        frame = _make_frame([])
        length = compute_torso_length(frame)
        assert length == 1.0


class TestCanonicalizeFrame:
    def test_canonical_xyz_within_range(self):
        joints = [
            Joint3D(id=0, name="root", parent_id=None, type="root",
                    world_xyz=(10, 0, 5)),
            Joint3D(id=1, name="neck", parent_id=0, type="neck",
                    world_xyz=(10, 5, 5)),
            Joint3D(id=2, name="tail", parent_id=0, type="tail",
                    world_xyz=(10, -3, 5)),
        ]
        frame = _make_frame(joints)
        result = canonicalize_frame(frame)

        # Root should be at origin in canonical space
        assert result is not None
        canonical = result["canonical_joints"]
        root_c = canonical[0]
        assert abs(root_c["canonical_xyz"][0]) < 0.01
        assert abs(root_c["canonical_xyz"][1]) < 0.01
        assert abs(root_c["canonical_xyz"][2]) < 0.01

    def test_scale_normalization(self):
        # Two frames with different scales should normalize similarly
        joints_small = [
            Joint3D(id=0, name="root", parent_id=None, type="root",
                    world_xyz=(0, 0, 0)),
            Joint3D(id=1, name="neck", parent_id=0, type="neck",
                    world_xyz=(0, 1, 0)),
        ]
        joints_big = [
            Joint3D(id=0, name="root", parent_id=None, type="root",
                    world_xyz=(0, 0, 0)),
            Joint3D(id=1, name="neck", parent_id=0, type="neck",
                    world_xyz=(0, 100, 0)),
        ]
        r1 = canonicalize_frame(_make_frame(joints_small))
        r2 = canonicalize_frame(_make_frame(joints_big))

        # After normalization, neck should be at the same canonical position
        neck1 = r1["canonical_joints"][1]["canonical_xyz"]
        neck2 = r2["canonical_joints"][1]["canonical_xyz"]
        assert abs(neck1[1] - neck2[1]) < 0.01

    def test_returns_metadata(self):
        joints = [
            Joint3D(id=0, name="root", parent_id=None, type="root",
                    world_xyz=(5, 5, 5)),
            Joint3D(id=1, name="head", parent_id=0, type="head",
                    world_xyz=(5, 8, 5)),
        ]
        frame = _make_frame(joints)
        result = canonicalize_frame(frame)

        assert "canonical_root_joint" in result
        assert "scale_factor" in result
        assert result["scale_factor"] > 0
