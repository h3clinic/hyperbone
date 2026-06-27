"""Tests for motion-derived articulation labeling."""
import pytest
import numpy as np

from hyperbone.labels.schema import NodeType, EdgeType, LabelSource
from hyperbone.labels.from_motion import (
    MotionArticulationConfig,
    detect_articulations_from_tracks,
)


def _make_hinge_tracks(T=20, N=100):
    """
    Create synthetic point tracks for a simple hinge motion.

    Left half of points stays still, right half rotates around a pivot.
    The pivot (midpoint) should be detected as an articulation.
    """
    tracks = np.zeros((T, N, 2))
    # Points spread across [0, 200] x [80, 120]
    initial_x = np.linspace(10, 190, N)
    initial_y = np.full(N, 100.0) + np.random.randn(N) * 5

    pivot_x = 100.0
    left_mask = initial_x < pivot_x
    right_mask = ~left_mask

    for t in range(T):
        angle = t * 0.05  # small rotation per frame
        tracks[t, :, 0] = initial_x
        tracks[t, :, 1] = initial_y
        # Rotate right points around pivot
        rx = initial_x[right_mask] - pivot_x
        ry = initial_y[right_mask] - 100.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        tracks[t, right_mask, 0] = pivot_x + rx * cos_a - ry * sin_a
        tracks[t, right_mask, 1] = 100.0 + rx * sin_a + ry * cos_a

    return tracks


def _make_rigid_tracks(T=20, N=50):
    """All points move rigidly together — no articulation."""
    tracks = np.zeros((T, N, 2))
    initial = np.random.rand(N, 2) * 100 + 50
    for t in range(T):
        tracks[t] = initial + t * np.array([2.0, 1.0])  # uniform translation
    return tracks


class TestMotionArticulation:
    def test_hinge_creates_articulation_node(self):
        """A hinge motion should produce at least one articulation node."""
        tracks = _make_hinge_tracks(T=30, N=100)
        config = MotionArticulationConfig(
            num_sample_points=100,
            min_frames=5,
            motion_threshold=0.05,
            max_clusters=4,
        )
        graph = detect_articulations_from_tracks(tracks, config=config, sample_id="hinge")
        articulations = graph.nodes_by_type(NodeType.ARTICULATION)
        assert len(articulations) >= 1

    def test_rigid_motion_no_articulation(self):
        """Purely rigid motion should produce no high-confidence articulation nodes."""
        tracks = _make_rigid_tracks(T=20, N=50)
        config = MotionArticulationConfig(
            min_frames=5,
            motion_threshold=0.1,
            max_clusters=4,
        )
        graph = detect_articulations_from_tracks(tracks, config=config, sample_id="rigid")
        # Rigid motion may produce low-confidence false positives,
        # but none should be high confidence
        high_conf = [n for n in graph.nodes_by_type(NodeType.ARTICULATION)
                     if n.confidence > 0.5]
        assert len(high_conf) == 0

    def test_insufficient_frames_returns_empty(self):
        """Less than min_frames should return empty graph."""
        tracks = np.random.rand(3, 50, 2) * 100
        config = MotionArticulationConfig(min_frames=5)
        graph = detect_articulations_from_tracks(tracks, config=config, sample_id="short")
        assert graph.node_count() == 0
        assert "insufficient_frames" in graph.metadata.get("error", "")

    def test_articulation_has_motion_source(self):
        tracks = _make_hinge_tracks(T=25, N=80)
        config = MotionArticulationConfig(min_frames=5, motion_threshold=0.05)
        graph = detect_articulations_from_tracks(tracks, config=config, sample_id="src")
        for node in graph.nodes:
            assert LabelSource.MOTION_ARTICULATION.value in node.label_sources

    def test_deformation_link_edges(self):
        tracks = _make_hinge_tracks(T=25, N=80)
        config = MotionArticulationConfig(min_frames=5, motion_threshold=0.05)
        graph = detect_articulations_from_tracks(tracks, config=config, sample_id="edges")
        if graph.edge_count() > 0:
            for edge in graph.edges:
                assert edge.edge_type == EdgeType.DEFORMATION_LINK

    def test_metadata_records_clusters(self):
        tracks = _make_hinge_tracks(T=20, N=60)
        config = MotionArticulationConfig(min_frames=5, max_clusters=4)
        graph = detect_articulations_from_tracks(tracks, config=config, sample_id="meta")
        assert "num_clusters" in graph.metadata
        assert graph.metadata["num_frames"] == 20
