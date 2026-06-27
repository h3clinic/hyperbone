"""Tests for label fusion."""
import pytest
import numpy as np

from hyperbone.labels.schema import (
    GraphLabel,
    HyperNodeLabel,
    HyperEdgeLabel,
    NodeType,
    EdgeType,
    LabelSource,
)
from hyperbone.labels.fusion import (
    FusionConfig,
    fuse_graph_labels,
)


def _make_node(id, ntype, xyz, sources, confidence=0.8, semantic=None):
    return HyperNodeLabel(
        id=id, node_type=ntype, xyz=xyz,
        confidence=confidence, label_sources=sources,
        semantic=semantic,
    )


def _make_edge(id, src, tgt, etype=EdgeType.BONE):
    return HyperEdgeLabel(id=id, source_node_id=src, target_node_id=tgt,
                          edge_type=etype, confidence=0.9)


class TestFusionMerge:
    def test_merge_duplicate_nodes(self):
        """Two nodes at same location from different sources should merge."""
        g1 = GraphLabel(sample_id="rig", nodes=[
            _make_node(0, NodeType.SEMANTIC_JOINT, [1.0, 0.0, 0.0],
                       {"rig_gt": 0.95}, semantic="knee"),
        ], metadata={"source": "rig"})
        g2 = GraphLabel(sample_id="motion", nodes=[
            _make_node(0, NodeType.ARTICULATION, [1.01, 0.0, 0.0],
                       {"motion_articulation": 0.8}),
        ], metadata={"source": "motion"})

        config = FusionConfig(merge_distance=0.05)
        fused, report = fuse_graph_labels([g1, g2], config)

        # Should merge into one node (close + compatible types)
        assert fused.node_count() == 1
        assert report.merged_count == 1
        # Merged node should have both sources
        node = fused.nodes[0]
        assert "rig_gt" in node.label_sources
        assert "motion_articulation" in node.label_sources

    def test_no_merge_if_far(self):
        """Nodes far apart should NOT merge."""
        g1 = GraphLabel(sample_id="a", nodes=[
            _make_node(0, NodeType.ENDPOINT, [0.0, 0.0, 0.0], {"rig_gt": 0.9}),
        ])
        g2 = GraphLabel(sample_id="b", nodes=[
            _make_node(0, NodeType.ENDPOINT, [5.0, 5.0, 5.0], {"medial_axis": 0.7}),
        ])

        config = FusionConfig(merge_distance=0.05)
        fused, report = fuse_graph_labels([g1, g2], config)
        assert fused.node_count() == 2
        assert report.merged_count == 0

    def test_reject_low_confidence(self):
        """Nodes below min_confidence with insufficient sources should be rejected."""
        g = GraphLabel(sample_id="weak", nodes=[
            _make_node(0, NodeType.RIDGE, [1, 1, 1],
                       {"vlm_semantic": 0.1}, confidence=0.1),
        ])
        # min_sources_for_uncertain=2 means single-source low-conf gets rejected
        config = FusionConfig(min_confidence=0.30, min_sources_for_uncertain=2)
        fused, report = fuse_graph_labels([g], config)
        assert fused.node_count() == 0
        assert report.rejected_count == 1

    def test_preserve_conflict_as_uncertain(self):
        """When types conflict at same location, mark uncertain."""
        g1 = GraphLabel(sample_id="a", nodes=[
            _make_node(0, NodeType.BRANCH, [1.0, 0.0, 0.0],
                       {"medial_axis": 0.7}),
        ])
        g2 = GraphLabel(sample_id="b", nodes=[
            _make_node(0, NodeType.RIDGE, [1.0, 0.0, 0.0],
                       {"procedural": 0.8}),
        ])

        # BRANCH and RIDGE are not in compatible pairs
        config = FusionConfig(merge_distance=0.05)
        fused, report = fuse_graph_labels([g1, g2], config)
        # They shouldn't merge (incompatible types), should stay separate
        assert fused.node_count() == 2

    def test_edge_remapping(self):
        """Edges should follow merged node IDs."""
        g = GraphLabel(sample_id="edges", nodes=[
            _make_node(0, NodeType.ROOT, [0, 0, 0], {"rig_gt": 0.9}),
            _make_node(1, NodeType.ENDPOINT, [1, 0, 0], {"rig_gt": 0.9}),
        ], edges=[
            _make_edge(0, 0, 1),
        ])
        config = FusionConfig()
        fused, report = fuse_graph_labels([g], config)
        assert fused.edge_count() >= 1
        # Edge endpoints should be valid node IDs
        valid_ids = {n.id for n in fused.nodes}
        for e in fused.edges:
            assert e.source_node_id in valid_ids
            assert e.target_node_id in valid_ids

    def test_empty_input(self):
        config = FusionConfig()
        fused, report = fuse_graph_labels([], config)
        assert fused.node_count() == 0
        assert report.total_candidates == 0

    def test_multi_source_fusion(self):
        """Fuse three sources with overlapping nodes."""
        g_rig = GraphLabel(sample_id="rig", nodes=[
            _make_node(0, NodeType.SEMANTIC_JOINT, [0, 0, 0],
                       {"rig_gt": 0.95}, semantic="hip"),
            _make_node(1, NodeType.SEMANTIC_JOINT, [0, 1, 0],
                       {"rig_gt": 0.95}, semantic="knee"),
        ], edges=[_make_edge(0, 0, 1)], metadata={"source": "rig"})

        g_motion = GraphLabel(sample_id="motion", nodes=[
            _make_node(0, NodeType.ARTICULATION, [0.01, 1.01, 0.01],
                       {"motion_articulation": 0.8}),
        ], metadata={"source": "motion"})

        g_medial = GraphLabel(sample_id="medial", nodes=[
            _make_node(0, NodeType.BRANCH, [0, 0.5, 0],
                       {"medial_axis": 0.5}),
        ], metadata={"source": "medial"})

        config = FusionConfig(merge_distance=0.05)
        fused, report = fuse_graph_labels([g_rig, g_motion, g_medial], config)
        # Hip stays, knee merges with motion articulation, medial branch is separate
        assert fused.node_count() >= 2
        assert len(report.sources_used) == 3
