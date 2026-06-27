"""Tests for HyperBone LabelForge schema."""
import json
import tempfile
from pathlib import Path

import pytest

from hyperbone.labels.schema import (
    NodeType,
    EdgeType,
    LabelSource,
    HyperNodeLabel,
    HyperEdgeLabel,
    GraphLabel,
    LabelFusionReport,
    SOURCE_WEIGHTS,
    save_graph_labels,
    load_graph_labels,
)


class TestNodeType:
    def test_all_types_exist(self):
        expected = {"root", "center", "endpoint", "branch", "articulation",
                    "bend", "ridge", "semantic_joint", "unknown"}
        assert {t.value for t in NodeType} == expected

    def test_enum_is_string(self):
        assert NodeType.ROOT == "root"
        assert isinstance(NodeType.BRANCH, str)


class TestEdgeType:
    def test_all_types_exist(self):
        expected = {"bone", "branch", "vein", "medial_axis", "hinge_link",
                    "ridge", "deformation_link", "unknown"}
        assert {t.value for t in EdgeType} == expected


class TestLabelSource:
    def test_all_sources_have_weights(self):
        for src in LabelSource:
            assert src in SOURCE_WEIGHTS

    def test_weights_ordered(self):
        assert SOURCE_WEIGHTS[LabelSource.MANUAL_REVIEW] > SOURCE_WEIGHTS[LabelSource.RIG_GT]
        assert SOURCE_WEIGHTS[LabelSource.RIG_GT] > SOURCE_WEIGHTS[LabelSource.MEDIAL_AXIS]
        assert SOURCE_WEIGHTS[LabelSource.MEDIAL_AXIS] > SOURCE_WEIGHTS[LabelSource.VLM_SEMANTIC]


class TestHyperNodeLabel:
    def test_create_minimal(self):
        n = HyperNodeLabel(id=0, node_type=NodeType.ENDPOINT)
        assert n.id == 0
        assert n.confidence == 0.0
        assert n.accepted is None

    def test_effective_confidence_single_source(self):
        n = HyperNodeLabel(
            id=0, node_type=NodeType.SEMANTIC_JOINT,
            label_sources={"rig_gt": 0.9}
        )
        # weight 0.95 * 0.9 / 0.95 = 0.9
        assert abs(n.effective_confidence() - 0.9) < 1e-6

    def test_effective_confidence_multi_source(self):
        n = HyperNodeLabel(
            id=0, node_type=NodeType.ARTICULATION,
            label_sources={"rig_gt": 0.8, "motion_articulation": 0.9}
        )
        ec = n.effective_confidence()
        assert 0.8 < ec < 0.9  # weighted average

    def test_effective_confidence_empty(self):
        n = HyperNodeLabel(id=0, node_type=NodeType.UNKNOWN)
        assert n.effective_confidence() == 0.0

    def test_to_dict_roundtrip(self):
        n = HyperNodeLabel(
            id=5, node_type=NodeType.BRANCH,
            xyz=[1.0, 2.0, 3.0], semantic="vein_fork",
            confidence=0.8, label_sources={"procedural": 1.0},
        )
        d = n.to_dict()
        assert d["node_type"] == "branch"
        n2 = HyperNodeLabel.from_dict(d)
        assert n2.id == 5
        assert n2.node_type == NodeType.BRANCH
        assert n2.xyz == [1.0, 2.0, 3.0]


class TestHyperEdgeLabel:
    def test_create(self):
        e = HyperEdgeLabel(id=0, source_node_id=1, target_node_id=2,
                           edge_type=EdgeType.BONE, confidence=0.95)
        assert e.length is None

    def test_to_dict_roundtrip(self):
        e = HyperEdgeLabel(
            id=3, source_node_id=0, target_node_id=1,
            edge_type=EdgeType.VEIN, confidence=0.9, length=1.5,
        )
        d = e.to_dict()
        e2 = HyperEdgeLabel.from_dict(d)
        assert e2.edge_type == EdgeType.VEIN
        assert e2.length == 1.5


class TestGraphLabel:
    def test_empty_graph(self):
        g = GraphLabel(sample_id="test")
        assert g.node_count() == 0
        assert g.edge_count() == 0

    def test_nodes_by_type(self):
        nodes = [
            HyperNodeLabel(id=0, node_type=NodeType.ENDPOINT),
            HyperNodeLabel(id=1, node_type=NodeType.BRANCH),
            HyperNodeLabel(id=2, node_type=NodeType.ENDPOINT),
        ]
        g = GraphLabel(sample_id="test", nodes=nodes)
        assert len(g.nodes_by_type(NodeType.ENDPOINT)) == 2
        assert len(g.nodes_by_type(NodeType.BRANCH)) == 1

    def test_uncertain_nodes(self):
        nodes = [
            HyperNodeLabel(id=0, node_type=NodeType.ENDPOINT),
            HyperNodeLabel(id=1, node_type=NodeType.BRANCH,
                           uncertainty_reason="low_conf"),
        ]
        g = GraphLabel(sample_id="test", nodes=nodes)
        assert len(g.uncertain_nodes()) == 1

    def test_jsonl_roundtrip(self):
        g = GraphLabel(
            sample_id="frame_001",
            image_path="img.png",
            nodes=[HyperNodeLabel(id=0, node_type=NodeType.ROOT, xyz=[0, 0, 0])],
            edges=[HyperEdgeLabel(id=0, source_node_id=0, target_node_id=0,
                                  edge_type=EdgeType.BONE)],
            metadata={"source": "test"},
        )
        line = g.to_jsonl_line()
        g2 = GraphLabel.from_jsonl_line(line)
        assert g2.sample_id == "frame_001"
        assert g2.nodes[0].node_type == NodeType.ROOT

    def test_save_load_file(self):
        graphs = [
            GraphLabel(sample_id=f"s{i}", nodes=[
                HyperNodeLabel(id=0, node_type=NodeType.CENTER)
            ]) for i in range(3)
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = Path(f.name)
        save_graph_labels(graphs, path)
        loaded = load_graph_labels(path)
        assert len(loaded) == 3
        assert loaded[2].sample_id == "s2"
        path.unlink()


class TestLabelFusionReport:
    def test_to_dict(self):
        r = LabelFusionReport(total_candidates=10, merged_count=3)
        d = r.to_dict()
        assert d["total_candidates"] == 10
        assert d["merged_count"] == 3
