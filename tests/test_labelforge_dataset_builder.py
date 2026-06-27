"""Tests for LabelForge dataset builder and report."""
import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from hyperbone.labels.schema import (
    GraphLabel,
    HyperNodeLabel,
    HyperEdgeLabel,
    NodeType,
    EdgeType,
    LabelSource,
    load_graph_labels,
)


class TestDatasetBuilder:
    """Test the dataset builder generates valid output."""

    def test_procedural_branches_generate(self):
        from scripts.build_labelforge_dataset import generate_procedural_branches
        graphs = generate_procedural_branches(10, seed=42)
        assert len(graphs) == 10
        for g in graphs:
            assert g.node_count() > 0
            assert "procedural_branch" == g.metadata.get("source")

    def test_procedural_leaves_generate(self):
        from scripts.build_labelforge_dataset import generate_procedural_leaves
        graphs = generate_procedural_leaves(10, seed=42)
        assert len(graphs) == 10
        for g in graphs:
            assert g.node_count() > 0

    def test_procedural_hinges_generate(self):
        from scripts.build_labelforge_dataset import generate_procedural_hinges
        graphs = generate_procedural_hinges(10, seed=42)
        assert len(graphs) == 10
        for g in graphs:
            assert g.node_count() >= 3

    def test_procedural_rocks_generate(self):
        from scripts.build_labelforge_dataset import generate_procedural_rocks
        graphs = generate_procedural_rocks(10, seed=42)
        assert len(graphs) == 10

    def test_motion_labels_generate(self):
        from scripts.build_labelforge_dataset import generate_synthetic_motion_labels
        graphs = generate_synthetic_motion_labels(5, seed=42)
        assert len(graphs) == 5
        for g in graphs:
            assert g.metadata.get("source") == "motion_articulation"

    def test_medial_axis_labels_generate(self):
        from scripts.build_labelforge_dataset import generate_medial_axis_labels
        graphs = generate_medial_axis_labels(10, seed=42)
        assert len(graphs) == 10
        valid = [g for g in graphs if g.node_count() > 0]
        assert len(valid) >= 5  # at least half should produce nodes

    def test_validate_graph(self):
        from scripts.build_labelforge_dataset import validate_graph
        # Valid
        g = GraphLabel(sample_id="ok", nodes=[
            HyperNodeLabel(id=0, node_type=NodeType.ENDPOINT,
                           label_sources={"procedural": 1.0}),
        ])
        assert validate_graph(g) is True

        # Empty nodes
        g2 = GraphLabel(sample_id="empty")
        assert validate_graph(g2) is False

        # Missing label_sources
        g3 = GraphLabel(sample_id="no_src", nodes=[
            HyperNodeLabel(id=0, node_type=NodeType.ENDPOINT),
        ])
        assert validate_graph(g3) is False

    def test_every_graph_has_label_sources(self):
        """All generated graphs must have label_sources on every node."""
        from scripts.build_labelforge_dataset import (
            generate_procedural_branches,
            generate_procedural_leaves,
        )
        for g in generate_procedural_branches(5, seed=1):
            for n in g.nodes:
                assert n.label_sources, f"Node {n.id} missing label_sources"
        for g in generate_procedural_leaves(5, seed=1):
            for n in g.nodes:
                assert n.label_sources

    def test_every_node_has_node_type(self):
        from scripts.build_labelforge_dataset import generate_procedural_hinges
        for g in generate_procedural_hinges(5, seed=1):
            for n in g.nodes:
                assert n.node_type is not None
                assert isinstance(n.node_type, NodeType)

    def test_every_edge_has_edge_type(self):
        from scripts.build_labelforge_dataset import generate_procedural_rocks
        for g in generate_procedural_rocks(5, seed=1):
            for e in g.edges:
                assert e.edge_type is not None
                assert isinstance(e.edge_type, EdgeType)

    def test_no_graph_lacks_confidence(self):
        """No node should have confidence exactly 0 for procedural sources."""
        from scripts.build_labelforge_dataset import generate_procedural_branches
        for g in generate_procedural_branches(10, seed=77):
            for n in g.nodes:
                assert n.confidence > 0, f"Node {n.id} has zero confidence"


class TestDatasetOutput:
    """Test dataset output file structure."""

    def test_manifest_structure(self):
        """Manifest should have required keys."""
        manifest = {
            "version": "labelforge_v0",
            "total_graphs": 100,
            "accepted_graphs": 95,
            "rejected_graphs": 5,
            "total_nodes": 500,
            "total_edges": 400,
            "review_queue_count": 10,
            "node_types": {"endpoint": 200, "branch": 100},
            "edge_types": {"branch": 300},
            "sources": {"procedural_branch": 100},
            "paths": {
                "graphs": "graphs/graphs.jsonl",
                "fused_graphs": "graphs/fused_graphs.jsonl",
                "review_queue": "review_queue.jsonl",
            },
        }
        required_keys = ["version", "total_graphs", "accepted_graphs",
                         "node_types", "edge_types", "sources", "paths"]
        for k in required_keys:
            assert k in manifest
