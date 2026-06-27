"""Tests for hyperbone.pose3d.schema — 3D pose data structures."""
import pytest
import json
from hyperbone.pose3d.schema import (
    Joint3D, Bone3D, PoseFrame3D, JointType,
)


class TestJoint3D:
    def test_create_joint(self):
        j = Joint3D(id=0, name="spine", parent_id=None, type="root",
                    world_xyz=(1.0, 2.0, 3.0))
        assert j.id == 0
        assert j.name == "spine"
        assert j.parent_id is None
        assert j.type == "root"
        assert j.world_xyz == (1.0, 2.0, 3.0)

    def test_to_dict(self):
        j = Joint3D(id=1, name="neck", parent_id=0, type="neck",
                    world_xyz=(0.5, 1.0, 0.0), image_xy=(320.0, 240.0))
        d = j.to_dict()
        assert d["id"] == 1
        assert d["name"] == "neck"
        assert d["parent_id"] == 0
        assert d["world_xyz"] == [0.5, 1.0, 0.0]
        assert d["image_xy"] == [320.0, 240.0]

    def test_from_dict(self):
        d = {"id": 2, "name": "head", "parent_id": 1, "type": "head",
             "world_xyz": [0, 5, 0], "camera_xyz": [0, 0, -5],
             "image_xy": [300, 200], "visible": True, "confidence": 0.9}
        j = Joint3D.from_dict(d)
        assert j.id == 2
        assert j.name == "head"
        assert j.world_xyz == (0, 5, 0)
        assert j.confidence == 0.9

    def test_from_dict_defaults(self):
        d = {"id": 0, "name": "root"}
        j = Joint3D.from_dict(d)
        assert j.parent_id is None
        assert j.type == JointType.UNKNOWN.value
        assert j.visible is True
        assert j.confidence == 1.0

    def test_roundtrip(self):
        j = Joint3D(id=5, name="tail_01", parent_id=0, type="tail",
                    world_xyz=(1, 2, 3), camera_xyz=(4, 5, 6),
                    local_xyz=(0.1, 0.2, 0.3), image_xy=(100, 200),
                    visible=False, confidence=0.75)
        d = j.to_dict()
        j2 = Joint3D.from_dict(d)
        assert j2.id == j.id
        assert j2.name == j.name
        assert j2.world_xyz == j.world_xyz
        assert j2.visible == j.visible


class TestBone3D:
    def test_create_bone(self):
        b = Bone3D(parent_id=0, child_id=1, name="spine_to_neck",
                   length=5.0, rest_length=5.0)
        assert b.parent_id == 0
        assert b.child_id == 1
        assert b.name == "spine_to_neck"

    def test_to_dict(self):
        b = Bone3D(parent_id=0, child_id=1, name="test",
                   length=3.5, rest_length=3.5)
        d = b.to_dict()
        assert d["parent_id"] == 0
        assert d["child_id"] == 1
        assert d["length"] == 3.5

    def test_from_dict(self):
        d = {"parent_id": 2, "child_id": 3, "name": "leg",
             "length": 10.0, "rest_length": 10.0}
        b = Bone3D.from_dict(d)
        assert b.parent_id == 2
        assert b.child_id == 3


class TestPoseFrame3D:
    def test_create_frame(self):
        joints = [
            Joint3D(id=0, name="root", parent_id=None, type="root",
                    world_xyz=(0, 0, 0)),
            Joint3D(id=1, name="spine", parent_id=0, type="spine",
                    world_xyz=(0, 1, 0)),
        ]
        bones = [Bone3D(parent_id=0, child_id=1, name="root_spine",
                        length=1.0, rest_length=1.0)]
        frame = PoseFrame3D(
            asset_id="fox",
            frame_idx=0,
            timestamp_sec=0.0,
            animation_name="Walk",
            joints=joints,
            bones=bones,
            resolution=(640, 480),
        )
        assert frame.asset_id == "fox"
        assert len(frame.joints) == 2
        assert len(frame.bones) == 1

    def test_to_dict(self):
        joints = [Joint3D(id=0, name="root", parent_id=None)]
        frame = PoseFrame3D(
            asset_id="wolf", frame_idx=5, timestamp_sec=0.2,
            animation_name="Run", joints=joints,
        )
        d = frame.to_dict()
        assert d["asset_id"] == "wolf"
        assert d["frame_idx"] == 5
        assert len(d["joints"]) == 1

    def test_from_dict(self):
        d = {
            "asset_id": "deer",
            "frame_idx": 10,
            "timestamp_sec": 0.4,
            "animation_name": "Walk",
            "joints": [
                {"id": 0, "name": "pelvis", "parent_id": None,
                 "world_xyz": [0, 0, 0], "image_xy": [320, 240]},
            ],
            "bones": [
                {"parent_id": 0, "child_id": 1, "name": "test", "length": 2.0}
            ],
            "resolution": [640, 480],
        }
        frame = PoseFrame3D.from_dict(d)
        assert frame.asset_id == "deer"
        assert frame.joints[0].name == "pelvis"

    def test_parent_child_hierarchy_valid(self):
        joints = [
            Joint3D(id=0, name="root", parent_id=None),
            Joint3D(id=1, name="spine", parent_id=0),
            Joint3D(id=2, name="neck", parent_id=1),
            Joint3D(id=3, name="head", parent_id=2),
        ]
        frame = PoseFrame3D(
            asset_id="test", frame_idx=0, timestamp_sec=0.0,
            animation_name="test", joints=joints,
        )
        # Verify: every non-None parent_id references an existing joint
        ids = {j.id for j in frame.joints}
        for j in frame.joints:
            if j.parent_id is not None:
                assert j.parent_id in ids, f"Joint {j.name} parent {j.parent_id} not found"

    def test_serialization_roundtrip(self):
        joints = [
            Joint3D(id=0, name="root", parent_id=None, type="root",
                    world_xyz=(1, 2, 3), image_xy=(100, 200)),
            Joint3D(id=1, name="neck", parent_id=0, type="neck",
                    world_xyz=(4, 5, 6), image_xy=(150, 180)),
        ]
        bones = [Bone3D(parent_id=0, child_id=1, name="test", length=5.0)]
        frame = PoseFrame3D(
            asset_id="fox", frame_idx=7, timestamp_sec=0.3,
            animation_name="Walk", joints=joints, bones=bones,
            resolution=(640, 480),
        )
        d = frame.to_dict()
        serialized = json.dumps(d)
        restored_d = json.loads(serialized)
        frame2 = PoseFrame3D.from_dict(restored_d)
        assert frame2.asset_id == frame.asset_id
        assert frame2.frame_idx == frame.frame_idx
        assert len(frame2.joints) == len(frame.joints)
        assert frame2.joints[0].world_xyz == frame.joints[0].world_xyz


class TestJointType:
    def test_enum_values(self):
        assert JointType.ROOT.value == "root"
        assert JointType.TAIL.value == "tail"
        assert JointType.PAW.value == "paw"
        assert JointType.UNKNOWN.value == "unknown"
