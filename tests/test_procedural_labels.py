"""Tests for procedural label generators."""
import pytest
import numpy as np

from hyperbone.labels.schema import NodeType, EdgeType, LabelSource
from hyperbone.labels.procedural.branches import BranchParams, generate_branch_graph
from hyperbone.labels.procedural.leaves import LeafParams, generate_leaf_graph
from hyperbone.labels.procedural.hinge_objects import HingeParams, generate_hinge_graph
from hyperbone.labels.procedural.rocks import RockParams, generate_rock_graph


class TestBranchGenerator:
    def test_y_branch_structure(self):
        """A minimal Y-branch should have 1 branch node and 3 endpoints."""
        params = BranchParams(max_depth=1, branch_prob=1.0, seed=42)
        graph = generate_branch_graph(params)
        assert graph.node_count() >= 3
        assert graph.edge_count() >= 2
        # Should have at least one branch or root node
        branch_nodes = graph.nodes_by_type(NodeType.BRANCH) + graph.nodes_by_type(NodeType.ROOT)
        assert len(branch_nodes) >= 1

    def test_endpoints_exist(self):
        params = BranchParams(max_depth=2, branch_prob=0.8, seed=123)
        graph = generate_branch_graph(params)
        endpoints = graph.nodes_by_type(NodeType.ENDPOINT)
        assert len(endpoints) >= 2

    def test_all_nodes_have_procedural_source(self):
        params = BranchParams(max_depth=2, seed=7)
        graph = generate_branch_graph(params)
        for node in graph.nodes:
            assert LabelSource.PROCEDURAL.value in node.label_sources
            assert node.label_sources[LabelSource.PROCEDURAL.value] == 1.0

    def test_edges_are_branch_type(self):
        params = BranchParams(max_depth=2, seed=7)
        graph = generate_branch_graph(params)
        for edge in graph.edges:
            assert edge.edge_type == EdgeType.BRANCH

    def test_deterministic_with_seed(self):
        p = BranchParams(max_depth=3, seed=99)
        g1 = generate_branch_graph(p)
        g2 = generate_branch_graph(p)
        assert g1.node_count() == g2.node_count()
        assert g1.edge_count() == g2.edge_count()


class TestLeafGenerator:
    def test_midrib_plus_veins(self):
        params = LeafParams(num_secondary_veins=4, seed=42)
        graph = generate_leaf_graph(params)
        # Midrib + secondary veins should give meaningful structure
        assert graph.node_count() >= 6
        assert graph.edge_count() >= 5

    def test_has_branch_nodes_from_secondary_veins(self):
        params = LeafParams(num_secondary_veins=4, tertiary_prob=0.0, seed=10)
        graph = generate_leaf_graph(params)
        branches = graph.nodes_by_type(NodeType.BRANCH)
        # Each secondary vein branches off midrib
        assert len(branches) >= 2

    def test_vein_edges(self):
        params = LeafParams(seed=5)
        graph = generate_leaf_graph(params)
        vein_edges = [e for e in graph.edges if e.edge_type == EdgeType.VEIN]
        assert len(vein_edges) == graph.edge_count()

    def test_2d_leaf_z_zero(self):
        """Leaf is 2D, all z coords should be 0."""
        params = LeafParams(seed=1)
        graph = generate_leaf_graph(params)
        for node in graph.nodes:
            if node.xyz:
                assert node.xyz[2] == 0.0


class TestHingeGenerator:
    def test_basic_hinge_chain(self):
        """3 segments = 2 hinges, plus endpoints."""
        params = HingeParams(num_segments=3, seed=42)
        graph = generate_hinge_graph(params)
        articulations = graph.nodes_by_type(NodeType.ARTICULATION)
        # Should have articulation nodes between segments
        assert len(articulations) >= 2

    def test_endpoints_at_extremes(self):
        params = HingeParams(num_segments=2, seed=42)
        graph = generate_hinge_graph(params)
        endpoints = graph.nodes_by_type(NodeType.ENDPOINT)
        assert len(endpoints) >= 1
        # Total structure: endpoints + articulations
        assert graph.node_count() >= 3

    def test_hinge_link_edges(self):
        params = HingeParams(num_segments=4, seed=42)
        graph = generate_hinge_graph(params)
        hinge_edges = [e for e in graph.edges if e.edge_type == EdgeType.HINGE_LINK]
        assert len(hinge_edges) >= 1

    def test_custom_angles(self):
        import math
        params = HingeParams(num_segments=3, joint_angles=[math.pi/4, math.pi/3], seed=42)
        graph = generate_hinge_graph(params)
        assert graph.node_count() >= 3


class TestRockGenerator:
    def test_center_plus_ridges(self):
        params = RockParams(num_ridges=4, ridge_nodes_per_ridge=3, seed=42)
        graph = generate_rock_graph(params)
        centers = graph.nodes_by_type(NodeType.CENTER)
        assert len(centers) == 1
        ridges = graph.nodes_by_type(NodeType.RIDGE)
        assert len(ridges) >= 4

    def test_ridge_edges(self):
        params = RockParams(num_ridges=3, seed=42)
        graph = generate_rock_graph(params)
        ridge_edges = [e for e in graph.edges if e.edge_type == EdgeType.RIDGE]
        assert len(ridge_edges) >= 3

    def test_confidence_is_approximate(self):
        """Rocks have lower confidence since ridges are approximate."""
        params = RockParams(seed=42)
        graph = generate_rock_graph(params)
        for node in graph.nodes:
            assert node.confidence <= 1.0
            # Rock procedural should use ~0.75
            assert node.confidence > 0.0

    def test_deterministic(self):
        p = RockParams(num_ridges=5, seed=77)
        g1 = generate_rock_graph(p)
        g2 = generate_rock_graph(p)
        assert g1.node_count() == g2.node_count()
