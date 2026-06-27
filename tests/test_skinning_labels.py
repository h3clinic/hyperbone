"""Tests for skinning weight label extraction."""
import pytest
import numpy as np

from hyperbone.labels.schema import NodeType, LabelSource
from hyperbone.labels.skinning import (
    SkinningInfluence,
    SkinningLabels,
    generate_joint_heatmaps,
    skinning_to_node_labels,
)


class TestSkinningInfluence:
    def test_create(self):
        si = SkinningInfluence(
            joint_index=0,
            joint_name="knee",
            influenced_vertices=np.random.rand(50, 3),
            weights=np.random.rand(50),
            influence_centroid=np.array([1.0, 2.0, 3.0]),
            influence_radius=0.5,
        )
        assert si.joint_index == 0
        assert si.influenced_vertices.shape == (50, 3)


class TestJointHeatmaps:
    def test_generate_heatmaps_shape(self):
        """Heatmap generation should produce correct shape."""
        # Vertices in 3D world space, in front of camera
        influences = [
            SkinningInfluence(
                joint_index=0, joint_name="hip",
                influenced_vertices=np.array([
                    [0.0, 0.5, 0.0], [0.1, 0.55, 0.0], [0.05, 0.52, 0.0]
                ]),
                weights=np.array([0.8, 0.6, 0.7]),
                influence_centroid=np.array([0.05, 0.52, 0.0]),
                influence_radius=0.1,
            ),
            SkinningInfluence(
                joint_index=1, joint_name="knee",
                influenced_vertices=np.array([
                    [0.3, 0.2, 0.0], [0.32, 0.22, 0.0]
                ]),
                weights=np.array([0.9, 0.7]),
                influence_centroid=np.array([0.31, 0.21, 0.0]),
                influence_radius=0.05,
            ),
        ]
        cam_pos = np.array([0.0, 0.5, 3.0])
        cam_target = np.array([0.0, 0.5, 0.0])
        heatmaps = generate_joint_heatmaps(
            influences, image_width=256, image_height=256,
            focal_length=500.0, camera_position=cam_pos, camera_target=cam_target,
        )
        assert heatmaps.shape == (2, 256, 256)
        assert heatmaps.max() <= 1.0
        assert heatmaps.min() >= 0.0

    def test_heatmap_nonzero_for_visible(self):
        """Heatmap should have non-zero values when vertices project into view."""
        influences = [
            SkinningInfluence(
                joint_index=0, joint_name="test",
                influenced_vertices=np.array([
                    [0.0, 0.5, 0.0], [0.02, 0.52, 0.0], [-0.02, 0.48, 0.0],
                    [0.01, 0.51, 0.0], [-0.01, 0.49, 0.0],
                ]),
                weights=np.array([0.9, 0.8, 0.7, 0.85, 0.75]),
                influence_centroid=np.array([0.0, 0.5, 0.0]),
                influence_radius=0.05,
            ),
        ]
        cam_pos = np.array([0.0, 0.5, 2.0])
        cam_target = np.array([0.0, 0.5, 0.0])
        heatmaps = generate_joint_heatmaps(
            influences, image_width=200, image_height=200,
            focal_length=300.0, camera_position=cam_pos, camera_target=cam_target,
        )
        assert heatmaps[0].max() > 0  # should have some signal


class TestSkinningToNodes:
    def test_generates_semantic_joint_nodes(self):
        """Skinning influences should produce semantic joint node labels."""
        influences = [
            SkinningInfluence(
                joint_index=0, joint_name="front_left_knee",
                influenced_vertices=np.random.rand(20, 3),
                weights=np.random.rand(20),
                influence_centroid=np.array([0.5, 0.3, 0.1]),
                influence_radius=0.1,
            ),
        ]
        nodes = skinning_to_node_labels(influences)
        assert len(nodes) == 1
        assert nodes[0].node_type == NodeType.SEMANTIC_JOINT
        assert nodes[0].semantic == "front_left_knee"
        assert LabelSource.SKINNING.value in nodes[0].label_sources
        assert nodes[0].radius == 0.1
