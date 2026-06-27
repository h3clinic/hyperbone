"""Tests for LabelForge report generation."""
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
    save_graph_labels,
)
from hyperbone.labels.review_queue import ReviewItem, save_review_queue
from hyperbone.report.labelforge_report import (
    DatasetReport,
    generate_report,
    write_report_markdown,
    write_report_json,
)


def _make_test_graphs(n=10):
    """Create minimal test graphs."""
    graphs = []
    for i in range(n):
        g = GraphLabel(
            sample_id=f"test_{i:03d}",
            nodes=[
                HyperNodeLabel(id=0, node_type=NodeType.ROOT, confidence=0.9,
                               xyz=[0, 0, 0], label_sources={"procedural": 1.0}),
                HyperNodeLabel(id=1, node_type=NodeType.ENDPOINT, confidence=0.8,
                               xyz=[1, 0, 0], label_sources={"procedural": 1.0}),
            ],
            edges=[
                HyperEdgeLabel(id=0, source_node_id=0, target_node_id=1,
                               edge_type=EdgeType.BRANCH, confidence=0.9),
            ],
            metadata={"source": "procedural_branch"},
        )
        graphs.append(g)
    return graphs


class TestReportGeneration:
    def test_report_counts(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "graphs.jsonl"
            graphs = _make_test_graphs(20)
            save_graph_labels(graphs, p)

            report = generate_report(p)
            assert report.total_graphs == 20
            assert report.accepted_graphs == 20
            assert report.rejected_graphs == 0
            assert report.total_nodes == 40
            assert report.total_edges == 20

    def test_report_source_counts(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "graphs.jsonl"
            save_graph_labels(_make_test_graphs(5), p)
            report = generate_report(p)
            assert "procedural_branch" in report.sources
            assert report.sources["procedural_branch"] == 5

    def test_report_node_types(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "graphs.jsonl"
            save_graph_labels(_make_test_graphs(3), p)
            report = generate_report(p)
            assert "root" in report.node_types
            assert "endpoint" in report.node_types

    def test_verdict_fail_low_count(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "graphs.jsonl"
            save_graph_labels(_make_test_graphs(10), p)
            report = generate_report(p)
            assert report.verdict == "FAIL"

    def test_verdict_partial(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "graphs.jsonl"
            save_graph_labels(_make_test_graphs(600), p)
            report = generate_report(p)
            assert report.verdict == "PARTIAL"

    def test_verdict_ready(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "graphs.jsonl"
            save_graph_labels(_make_test_graphs(2100), p)
            report = generate_report(p)
            assert "READY" in report.verdict

    def test_review_queue_included(self):
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "graphs.jsonl"
            rp = Path(td) / "review.jsonl"
            save_graph_labels(_make_test_graphs(10), gp)
            items = [ReviewItem(sample_id="x", node_id=0, reason="low_conf",
                                priority=0.5)]
            save_review_queue(items, rp)
            report = generate_report(gp, review_queue_path=rp)
            assert report.review_queue_count == 1

    def test_write_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "graphs.jsonl"
            save_graph_labels(_make_test_graphs(5), gp)
            report = generate_report(gp)
            md_path = Path(td) / "report.md"
            write_report_markdown(report, md_path)
            assert md_path.exists()
            content = md_path.read_text()
            assert "Verdict" in content
            assert "FAIL" in content

    def test_write_json(self):
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "graphs.jsonl"
            save_graph_labels(_make_test_graphs(5), gp)
            report = generate_report(gp)
            jp = Path(td) / "report.json"
            write_report_json(report, jp)
            data = json.loads(jp.read_text())
            assert data["total_graphs"] == 5

    def test_known_weaknesses(self):
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "graphs.jsonl"
            save_graph_labels(_make_test_graphs(5), gp)
            report = generate_report(gp)
            assert len(report.known_weaknesses) > 0
            # Should flag missing animal/motion data
            weak_text = " ".join(report.known_weaknesses)
            assert "animal" in weak_text.lower() or "motion" in weak_text.lower()
