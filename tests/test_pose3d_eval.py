"""Tests for hyperbone.pose3d.eval — 3D pose evaluation metrics."""
import pytest
import numpy as np
from hyperbone.pose3d.schema import Joint3D, Bone3D, PoseFrame3D
from hyperbone.pose3d.eval import (
    mpjpe,
    pck_3d,
    mean_bone_length_error,
    reprojection_error,
)


def _joint(id, xyz, name="j", image_xy=(0, 0), parent_id=None):
    return Joint3D(id=id, name=name, parent_id=parent_id,
                   world_xyz=tuple(xyz), image_xy=tuple(image_xy))


def _frame(joints, bones=None):
    return PoseFrame3D(
        asset_id="test", frame_idx=0, timestamp_sec=0.0,
        animation_name="test", joints=joints, bones=bones or [],
    )


class TestMPJPE:
    def test_zero_error(self):
        pred = [_joint(0, [1, 2, 3]), _joint(1, [4, 5, 6])]
        gt = [_joint(0, [1, 2, 3]), _joint(1, [4, 5, 6])]
        assert mpjpe(pred, gt) == 0.0

    def test_known_error(self):
        pred = [_joint(0, [0, 0, 0])]
        gt = [_joint(0, [3, 4, 0])]  # distance = 5
        assert abs(mpjpe(pred, gt) - 5.0) < 0.001

    def test_average_across_joints(self):
        pred = [_joint(0, [0, 0, 0]), _joint(1, [0, 0, 0])]
        gt = [_joint(0, [3, 4, 0]), _joint(1, [0, 0, 5])]
        # Errors: 5.0 and 5.0
        assert abs(mpjpe(pred, gt) - 5.0) < 0.001

    def test_missing_joints(self):
        pred = [_joint(0, [0, 0, 0]), _joint(2, [1, 1, 1])]
        gt = [_joint(0, [1, 0, 0]), _joint(1, [5, 5, 5])]
        # Only joint 0 matches: error = 1.0
        assert abs(mpjpe(pred, gt) - 1.0) < 0.001

    def test_empty_returns_inf(self):
        assert mpjpe([], [_joint(0, [1, 2, 3])]) == float('inf')
        assert mpjpe([_joint(0, [1, 2, 3])], []) == float('inf')


class TestPCK3D:
    def test_all_correct(self):
        pred = [_joint(0, [0, 0, 0]), _joint(1, [1, 1, 1])]
        gt = [_joint(0, [0, 0, 0]), _joint(1, [1, 1, 1])]
        assert pck_3d(pred, gt, threshold=0.1) == 1.0

    def test_none_correct(self):
        pred = [_joint(0, [0, 0, 0])]
        gt = [_joint(0, [100, 100, 100])]
        assert pck_3d(pred, gt, threshold=0.1) == 0.0

    def test_threshold_sensitivity(self):
        pred = [_joint(0, [0, 0, 0]), _joint(1, [0, 0, 0])]
        gt = [_joint(0, [0.05, 0, 0]), _joint(1, [0.15, 0, 0])]
        # At threshold 0.1: joint 0 correct (0.05 < 0.1), joint 1 not (0.15 > 0.1)
        assert abs(pck_3d(pred, gt, threshold=0.1) - 0.5) < 0.001
        # At threshold 0.2: both correct
        assert pck_3d(pred, gt, threshold=0.2) == 1.0


class TestBoneLengthError:
    def test_zero_error(self):
        joints_p = [_joint(0, [0, 0, 0], name="a"), _joint(1, [1, 0, 0], name="b", parent_id=0)]
        joints_g = [_joint(0, [0, 0, 0], name="a"), _joint(1, [1, 0, 0], name="b", parent_id=0)]
        bones = [Bone3D(parent_id=0, child_id=1, name="a_b", length=1.0)]
        pred = _frame(joints_p, bones)
        gt = _frame(joints_g, bones)
        err = mean_bone_length_error(pred, gt)
        assert err < 0.001

    def test_length_difference(self):
        joints_p = [_joint(0, [0, 0, 0], name="a"), _joint(1, [2, 0, 0], name="b", parent_id=0)]
        joints_g = [_joint(0, [0, 0, 0], name="a"), _joint(1, [1, 0, 0], name="b", parent_id=0)]
        bones = [Bone3D(parent_id=0, child_id=1, name="a_b", length=1.0)]
        pred = _frame(joints_p, bones)
        gt = _frame(joints_g, bones)
        err = mean_bone_length_error(pred, gt)
        assert abs(err - 1.0) < 0.001  # |2-1| = 1

    def test_empty_bones(self):
        pred = _frame([_joint(0, [0, 0, 0], name="a")])
        gt = _frame([_joint(0, [0, 0, 0], name="a")])
        err = mean_bone_length_error(pred, gt)
        assert err == float('inf')


class TestReprojectionError:
    def test_zero_error(self):
        pred = [_joint(0, [0, 0, 0], image_xy=(100, 200))]
        gt = [_joint(0, [0, 0, 0], image_xy=(100, 200))]
        err = reprojection_error(pred, gt)
        assert err == 0.0

    def test_known_distance(self):
        pred = [_joint(0, [0, 0, 0], image_xy=(100, 200))]
        gt = [_joint(0, [0, 0, 0], image_xy=(103, 204))]
        err = reprojection_error(pred, gt)
        assert abs(err - 5.0) < 0.001  # 3-4-5

    def test_multiple_joints(self):
        pred = [
            _joint(0, [0, 0, 0], image_xy=(0, 0)),
            _joint(1, [0, 0, 0], image_xy=(10, 0)),
        ]
        gt = [
            _joint(0, [0, 0, 0], image_xy=(3, 4)),  # error=5
            _joint(1, [0, 0, 0], image_xy=(10, 0)), # error=0
        ]
        err = reprojection_error(pred, gt)
        assert abs(err - 2.5) < 0.001  # mean of 5 and 0
