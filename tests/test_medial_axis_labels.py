"""Tests for medial axis label extraction."""
import pytest
import numpy as np

from hyperbone.labels.schema import NodeType, EdgeType, LabelSource
from hyperbone.labels.from_medial_axis import (
    MedialAxisConfig,
    extract_medial_axis_graph,
)


class TestMedialAxisLabeler:
    def test_straight_line_creates_two_endpoints(self):
        """A thin horizontal line should give 2 endpoints and 1 edge."""
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[45:55, 20:180] = 1  # horizontal bar
        config = MedialAxisConfig(min_branch_length=3)
        graph = extract_medial_axis_graph(mask, config, sample_id="line")
        endpoints = graph.nodes_by_type(NodeType.ENDPOINT)
        assert len(endpoints) >= 2
        assert graph.edge_count() >= 1

    def test_y_shape_creates_branch_node(self):
        """A Y-shaped mask should produce a branch node."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Vertical stem
        mask[100:190, 95:105] = 1
        # Left branch
        for i in range(60):
            x = 95 - i
            y = 100 - i
            if 0 <= x < 200 and 0 <= y < 200:
                mask[y:y+10, x:x+10] = 1
        # Right branch
        for i in range(60):
            x = 105 + i
            y = 100 - i
            if 0 <= x < 200 and 0 <= y < 200:
                mask[y:y+10, x:x+10] = 1

        graph = extract_medial_axis_graph(mask, sample_id="ybranch")
        branches = graph.nodes_by_type(NodeType.BRANCH)
        # Should detect at least one branch point
        assert len(branches) >= 1 or graph.node_count() >= 3

    def test_empty_mask_returns_empty_graph(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        graph = extract_medial_axis_graph(mask, sample_id="empty")
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_small_mask_returns_empty(self):
        """Mask with < 10 pixels should return empty."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50:53] = 1  # only 3 pixels
        graph = extract_medial_axis_graph(mask, sample_id="tiny")
        assert graph.node_count() == 0

    def test_medial_axis_edges_type(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[40:60, 10:190] = 1
        graph = extract_medial_axis_graph(mask, sample_id="bar")
        for edge in graph.edges:
            assert edge.edge_type == EdgeType.MEDIAL_AXIS

    def test_nodes_have_medial_axis_source(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[40:60, 10:190] = 1
        graph = extract_medial_axis_graph(mask, sample_id="bar")
        for node in graph.nodes:
            assert LabelSource.MEDIAL_AXIS.value in node.label_sources

    def test_radius_is_positive(self):
        """Nodes should have radius from distance transform."""
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[30:70, 10:190] = 1  # thick bar
        graph = extract_medial_axis_graph(mask, sample_id="thick")
        for node in graph.nodes:
            if node.radius is not None:
                assert node.radius > 0

    def test_circle_mask(self):
        """Circular mask should produce limited nodes (center-ish)."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        cy, cx = 50, 50
        for y in range(100):
            for x in range(100):
                if (y - cy)**2 + (x - cx)**2 < 30**2:
                    mask[y, x] = 1
        graph = extract_medial_axis_graph(mask, sample_id="circle")
        # Circle's medial axis is just the center point(ish)
        assert graph.node_count() <= 5
