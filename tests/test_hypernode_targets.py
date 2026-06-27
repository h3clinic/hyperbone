"""Tests for HyperNodeNet target generation."""
import pytest
import numpy as np

from hyperbone.labels.schema import (
    GraphLabel,
    HyperNodeLabel,
    HyperEdgeLabel,
    NodeType,
    EdgeType,
)
from hyperbone.hypernodes.targets import (
    generate_node_heatmaps,
    generate_all_node_heatmap,
    generate_radius_map,
    generate_node_type_map,
    generate_edge_affinity_map,
    generate_targets,
)
from hyperbone.hypernodes.dataset import NUM_NODE_TYPES


def _make_graph(num_nodes=5) -> GraphLabel:
    nodes = []
    for i in range(num_nodes):
        angle = 2 * 3.14159 * i / num_nodes
        nodes.append(HyperNodeLabel(
            id=i,
            node_type=[NodeType.ENDPOINT, NodeType.BRANCH, NodeType.ARTICULATION,
                       NodeType.BEND, NodeType.RIDGE][i % 5],
            xy=[0.2 + 0.3 * np.cos(angle), 0.2 + 0.3 * np.sin(angle)],
            confidence=0.9,
            radius=0.05,
            label_sources={"procedural": 0.9},
        ))
    edges = []
    for i in range(num_nodes - 1):
        edges.append(HyperEdgeLabel(
            id=i, source_node_id=i, target_node_id=i + 1,
            edge_type=EdgeType.BRANCH, confidence=0.8,
        ))
    return GraphLabel(sample_id="target_test", nodes=nodes, edges=edges)


class TestNodeHeatmaps:
    def test_shape(self):
        g = _make_graph(5)
        hm = generate_node_heatmaps(g, 64)
        assert hm.shape == (NUM_NODE_TYPES, 64, 64)

    def test_peaks_at_node_locations(self):
        """Heatmap has peaks where nodes are."""
        g = _make_graph(3)
        hm = generate_node_heatmaps(g, 64, sigma=3.0)
        # At least one channel should have a clear peak
        assert hm.max() > 0.9

    def test_empty_graph(self):
        g = GraphLabel(sample_id="empty")
        hm = generate_node_heatmaps(g, 64)
        assert hm.max() == 0.0

    def test_correct_channel(self):
        """Endpoint node should only appear in endpoint channel."""
        node = HyperNodeLabel(
            id=0, node_type=NodeType.ENDPOINT,
            xy=[0.5, 0.5], confidence=1.0, label_sources={"test": 1.0},
        )
        g = GraphLabel(sample_id="single", nodes=[node])
        hm = generate_node_heatmaps(g, 64, sigma=3.0)
        # Endpoint is index 0
        assert hm[0].max() > 0.9
        # Other channels should be zero
        assert hm[1:].max() == 0.0


class TestAllNodeHeatmap:
    def test_shape(self):
        g = _make_graph(3)
        hm = generate_all_node_heatmap(g, 64)
        assert hm.shape == (1, 64, 64)

    def test_has_peaks(self):
        g = _make_graph(3)
        hm = generate_all_node_heatmap(g, 64, sigma=3.0)
        assert hm.max() > 0.9


class TestRadiusMap:
    def test_shape(self):
        g = _make_graph(3)
        rm = generate_radius_map(g, 64)
        assert rm.shape == (1, 64, 64)

    def test_has_values(self):
        g = _make_graph(3)
        rm = generate_radius_map(g, 64)
        assert rm.max() > 0


class TestNodeTypeMap:
    def test_shape(self):
        g = _make_graph(3)
        tm = generate_node_type_map(g, 64)
        assert tm.shape == (1, 64, 64)

    def test_nonzero(self):
        g = _make_graph(3)
        tm = generate_node_type_map(g, 64)
        assert tm.max() > 0


class TestEdgeAffinityMap:
    def test_shape(self):
        g = _make_graph(3)
        am = generate_edge_affinity_map(g, 64)
        assert am.shape == (1, 64, 64)

    def test_has_lines(self):
        g = _make_graph(3)
        am = generate_edge_affinity_map(g, 64)
        assert am.max() > 0


class TestGenerateTargets:
    def test_all_keys(self):
        g = _make_graph(5)
        targets = generate_targets(g, 64)
        assert "node_heatmaps" in targets
        assert "all_node_heatmap" in targets
        assert "radius_map" in targets
        assert "node_type_map" in targets
        assert "edge_affinity_map" in targets
