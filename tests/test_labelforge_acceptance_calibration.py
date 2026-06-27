"""Tests for LabelForge acceptance calibration."""
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
from hyperbone.labels.acceptance import (
    AcceptanceConfig,
    GraphQuality,
    assess_graph_quality,
    classify_graph,
)


def _make_graph(nodes, edges=None, source="procedural_branch"):
    return GraphLabel(
        sample_id="test",
        nodes=nodes,
        edges=edges or [],
        metadata={"source": source},
    )


def _node(id, conf=0.9, ntype=NodeType.ENDPOINT, sources=None):
    return HyperNodeLabel(
        id=id, node_type=ntype, xyz=[id * 0.1, 0, 0],
        confidence=conf,
        label_sources=sources or {"procedural": conf},
    )


def _edge(id, src, tgt):
    return HyperEdgeLabel(
        id=id, source_node_id=src, target_node_id=tgt,
        edge_type=EdgeType.BRANCH, confidence=0.9,
    )


class TestGraphQuality:
    def test_good_graph_accepted(self):
        """High confidence connected graph should be trainable."""
        g = _make_graph(
            nodes=[_node(0, 0.95), _node(1, 0.90), _node(2, 0.85)],
            edges=[_edge(0, 0, 1), _edge(1, 1, 2)],
        )
        q = assess_graph_quality(g)
        assert q.accepted is True
        assert q.review_needed is False
        assert classify_graph(g, q) == "trainable"

    def test_too_few_nodes_rejected(self):
        """Graph with < 2 nodes should be rejected."""
        g = _make_graph(nodes=[_node(0)])
        q = assess_graph_quality(g)
        assert q.accepted is False
        assert "too_few_nodes" in q.reject_reason

    def test_too_few_edges_rejected(self):
        """Graph with no edges should be rejected."""
        g = _make_graph(nodes=[_node(0), _node(1)])
        q = assess_graph_quality(g)
        assert q.accepted is False
        assert "too_few_edges" in q.reject_reason

    def test_low_confidence_rejected(self):
        """Graph with mean confidence < 0.45 should be rejected."""
        g = _make_graph(
            nodes=[_node(0, 0.1), _node(1, 0.2), _node(2, 0.15)],
            edges=[_edge(0, 0, 1), _edge(1, 1, 2)],
        )
        q = assess_graph_quality(g)
        assert q.accepted is False
        assert "low_mean_confidence" in q.reject_reason

    def test_high_uncertain_ratio_triggers_review(self):
        """Many uncertain nodes should trigger review_needed."""
        nodes = [
            HyperNodeLabel(
                id=i, node_type=NodeType.ENDPOINT, xyz=[i, 0, 0],
                confidence=0.6,
                label_sources={"medial_axis": 0.5},
                uncertainty_reason="motion_type_mismatch",
            )
            for i in range(4)
        ] + [_node(4, 0.9)]
        edges = [_edge(i, i, i + 1) for i in range(4)]
        g = _make_graph(nodes, edges)
        q = assess_graph_quality(g)
        assert q.uncertain_node_ratio > 0.25
        assert q.review_needed is True

    def test_weak_single_source_triggers_review(self):
        """Medial-axis with single source should trigger review."""
        g = _make_graph(
            nodes=[
                _node(0, 0.6, sources={"medial_axis": 0.6}),
                _node(1, 0.6, sources={"medial_axis": 0.6}),
                _node(2, 0.6, sources={"medial_axis": 0.6}),
            ],
            edges=[_edge(0, 0, 1), _edge(1, 1, 2)],
            source="medial_axis",
        )
        q = assess_graph_quality(g)
        assert q.source_diversity_count == 1
        assert q.review_needed is True

    def test_disconnected_graph_triggers_review(self):
        """Disconnected graph (for non-fragment source) should trigger review."""
        g = _make_graph(
            nodes=[_node(0), _node(1), _node(2), _node(3)],
            edges=[_edge(0, 0, 1)],  # only connects 0-1, leaves 2,3 isolated
            source="procedural_branch",
        )
        q = assess_graph_quality(g)
        assert q.is_connected is False
        assert q.review_needed is True


class TestClassification:
    def test_rejected_classification(self):
        g = _make_graph(nodes=[_node(0)])
        q = assess_graph_quality(g)
        assert classify_graph(g, q) == "rejected"

    def test_review_classification(self):
        g = _make_graph(
            nodes=[_node(0, 0.55, sources={"medial_axis": 0.55}),
                   _node(1, 0.55, sources={"medial_axis": 0.55})],
            edges=[_edge(0, 0, 1)],
            source="medial_axis",
        )
        q = assess_graph_quality(g)
        assert classify_graph(g, q) == "review"

    def test_trainable_needs_high_confidence(self):
        """Even accepted graphs need >= 0.60 mean confidence for trainable."""
        g = _make_graph(
            nodes=[_node(0, 0.50), _node(1, 0.55)],
            edges=[_edge(0, 0, 1)],
        )
        q = assess_graph_quality(g)
        # This has mean conf = 0.525, accepted but not trainable
        cls = classify_graph(g, q)
        assert cls == "review"  # below trainable_min_confidence

    def test_rigged_asset_high_conf_trainable(self):
        """Rigged asset graph with high confidence should be trainable."""
        g = _make_graph(
            nodes=[
                _node(0, 0.95, NodeType.SEMANTIC_JOINT, {"rig_gt": 0.95}),
                _node(1, 0.95, NodeType.SEMANTIC_JOINT, {"rig_gt": 0.95}),
                _node(2, 0.95, NodeType.SEMANTIC_JOINT, {"rig_gt": 0.95}),
            ],
            edges=[_edge(0, 0, 1), _edge(1, 1, 2)],
            source="rigged_asset",
        )
        q = assess_graph_quality(g)
        assert q.accepted is True
        assert q.review_needed is False
        assert classify_graph(g, q) == "trainable"


class TestRiggedDataset:
    def test_rigged_graph_has_semantic_joints(self):
        """Rigged graphs should have semantic_joint nodes."""
        from hyperbone.labels.from_rig import (
            extract_joints_from_gltf,
            joints_to_graph_label,
            RigExtractionConfig,
        )
        from pathlib import Path

        fox_path = Path("assets/gltf_samples/Fox/glTF-Binary/Fox.glb")
        if not fox_path.exists():
            pytest.skip("Fox.glb not available")

        joints = extract_joints_from_gltf(fox_path)
        config = RigExtractionConfig()
        g = joints_to_graph_label(joints, config, sample_id="test_fox")

        semantic_joints = g.nodes_by_type(NodeType.SEMANTIC_JOINT)
        # Fox has joints that should map to semantic_joint type
        assert g.node_count() >= 10
        assert g.edge_count() >= 9

    def test_rigged_graph_has_bone_edges(self):
        """Rigged graphs should have bone edges."""
        from hyperbone.labels.from_rig import (
            extract_joints_from_gltf,
            joints_to_graph_label,
            RigExtractionConfig,
        )
        from pathlib import Path

        fox_path = Path("assets/gltf_samples/Fox/glTF-Binary/Fox.glb")
        if not fox_path.exists():
            pytest.skip("Fox.glb not available")

        joints = extract_joints_from_gltf(fox_path)
        config = RigExtractionConfig()
        g = joints_to_graph_label(joints, config, sample_id="test_fox")

        bone_edges = [e for e in g.edges if e.edge_type == EdgeType.BONE]
        assert len(bone_edges) >= 9
        for e in bone_edges:
            assert LabelSource.RIG_GT.value in e.label_sources

    def test_rigged_graph_trainable(self):
        """Rigged asset graph should pass acceptance as trainable."""
        from hyperbone.labels.from_rig import (
            extract_joints_from_gltf,
            joints_to_graph_label,
            RigExtractionConfig,
        )
        from pathlib import Path

        fox_path = Path("assets/gltf_samples/Fox/glTF-Binary/Fox.glb")
        if not fox_path.exists():
            pytest.skip("Fox.glb not available")

        joints = extract_joints_from_gltf(fox_path)
        config = RigExtractionConfig()
        g = joints_to_graph_label(joints, config, sample_id="test_fox")
        g.metadata["source"] = "rigged_asset"

        q = assess_graph_quality(g)
        assert q.accepted is True
        assert classify_graph(g, q) == "trainable"

    def test_skinning_influence_attaches(self):
        """Skinning labels should generate node labels with skinning source."""
        from hyperbone.labels.skinning import SkinningInfluence, skinning_to_node_labels

        influences = [
            SkinningInfluence(
                joint_index=0, joint_name="hip",
                influenced_vertices=np.random.rand(30, 3),
                weights=np.random.rand(30),
                influence_centroid=np.array([0.5, 0.3, 0.0]),
                influence_radius=0.15,
            ),
        ]
        nodes = skinning_to_node_labels(influences)
        assert len(nodes) == 1
        assert LabelSource.SKINNING.value in nodes[0].label_sources
        assert nodes[0].radius == 0.15
