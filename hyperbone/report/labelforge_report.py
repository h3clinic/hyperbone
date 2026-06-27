"""
LabelForge dataset report generator.

Produces a structured report of dataset composition, quality metrics,
and training readiness verdict.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from hyperbone.labels.schema import (
    GraphLabel,
    NodeType,
    EdgeType,
    LabelSource,
    load_graph_labels,
)
from hyperbone.labels.review_queue import ReviewItem, load_review_queue


@dataclass
class DatasetReport:
    """Complete report of a LabelForge dataset."""
    total_graphs: int = 0
    accepted_graphs: int = 0
    rejected_graphs: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    avg_nodes_per_graph: float = 0.0
    avg_edges_per_graph: float = 0.0
    avg_confidence: float = 0.0
    node_types: dict[str, int] = field(default_factory=dict)
    edge_types: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)
    review_queue_count: int = 0
    review_queue_pct: float = 0.0
    known_weaknesses: list[str] = field(default_factory=list)
    verdict: str = "UNKNOWN"
    verdict_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def generate_report(
    graphs_path: Path,
    review_queue_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> DatasetReport:
    """
    Generate a dataset report from graph labels and review queue.

    Args:
        graphs_path: path to fused_graphs.jsonl
        review_queue_path: path to review_queue.jsonl
        manifest_path: optional path to dataset_manifest.json
    """
    report = DatasetReport()

    # Load graphs
    graphs = load_graph_labels(graphs_path)
    report.total_graphs = len(graphs)

    if not graphs:
        report.verdict = "FAIL"
        report.verdict_reasons = ["no graphs found"]
        return report

    # Count accepted/rejected
    accepted = 0
    rejected = 0
    for g in graphs:
        has_rejected_node = any(n.accepted is False for n in g.nodes)
        if has_rejected_node:
            rejected += 1
        else:
            accepted += 1

    report.accepted_graphs = accepted
    report.rejected_graphs = rejected

    # Node/edge counts
    total_nodes = 0
    total_edges = 0
    node_types = Counter()
    edge_types = Counter()
    sources = Counter()
    confidences = []

    for g in graphs:
        total_nodes += g.node_count()
        total_edges += g.edge_count()
        for n in g.nodes:
            node_types[n.node_type.value] += 1
            if n.confidence > 0:
                confidences.append(n.confidence)
        for e in g.edges:
            edge_types[e.edge_type.value] += 1
        src = g.metadata.get("source", "unknown")
        sources[src] += 1

    report.total_nodes = total_nodes
    report.total_edges = total_edges
    report.avg_nodes_per_graph = total_nodes / len(graphs) if graphs else 0
    report.avg_edges_per_graph = total_edges / len(graphs) if graphs else 0
    report.avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    report.node_types = dict(node_types.most_common())
    report.edge_types = dict(edge_types.most_common())
    report.sources = dict(sources.most_common())

    # Review queue
    if review_queue_path and review_queue_path.exists():
        items = load_review_queue(review_queue_path)
        report.review_queue_count = len(items)
        report.review_queue_pct = len(items) / len(graphs) if graphs else 0
    else:
        report.review_queue_count = 0
        report.review_queue_pct = 0.0

    # Known weaknesses
    weaknesses = []
    if "rigged_asset" not in sources:
        weaknesses.append("no rigged asset labels (no real animal joints)")
    if "motion_articulation" not in sources:
        weaknesses.append("no motion-derived articulation labels")
    if "animal_dataset" not in sources:
        weaknesses.append("no real animal dataset labels (AP-10K/Animal3D)")
    if total_nodes < 5000:
        weaknesses.append(f"low node volume ({total_nodes})")
    if report.avg_confidence < 0.5:
        weaknesses.append(f"low average confidence ({report.avg_confidence:.2f})")
    if len(sources) < 3:
        weaknesses.append(f"limited source diversity ({len(sources)} sources)")
    report.known_weaknesses = weaknesses

    # Verdict
    reasons = []
    if accepted < 500:
        report.verdict = "FAIL"
        reasons.append(f"only {accepted} accepted graphs (need 500+)")
    elif accepted < 2000:
        report.verdict = "PARTIAL"
        reasons.append(f"{accepted} accepted graphs (need 2000+ for READY)")
    elif report.review_queue_pct > 0.30:
        report.verdict = "PARTIAL (review queue > 30%)"
        reasons.append(f"review queue {report.review_queue_pct:.0%} > 30%")
    else:
        report.verdict = "READY_FOR_SMOKE_TRAINING"
        reasons.append(f"{accepted} accepted, review {report.review_queue_pct:.0%}")

    report.verdict_reasons = reasons
    return report


def write_report_markdown(report: DatasetReport, path: Path) -> None:
    """Write report as markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LabelForge Dataset Report",
        "",
        f"## Verdict: **{report.verdict}**",
        "",
        "Reasons: " + "; ".join(report.verdict_reasons),
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total graphs | {report.total_graphs} |",
        f"| Accepted | {report.accepted_graphs} |",
        f"| Rejected | {report.rejected_graphs} |",
        f"| Total nodes | {report.total_nodes} |",
        f"| Total edges | {report.total_edges} |",
        f"| Avg nodes/graph | {report.avg_nodes_per_graph:.1f} |",
        f"| Avg edges/graph | {report.avg_edges_per_graph:.1f} |",
        f"| Avg confidence | {report.avg_confidence:.3f} |",
        f"| Review queue | {report.review_queue_count} ({report.review_queue_pct:.0%}) |",
        "",
        "## Sources",
        "",
        "| Source | Count |",
        "|--------|-------|",
    ]
    for src, cnt in report.sources.items():
        lines.append(f"| {src} | {cnt} |")

    lines.extend([
        "",
        "## Node Types",
        "",
        "| Type | Count |",
        "|------|-------|",
    ])
    for nt, cnt in report.node_types.items():
        lines.append(f"| {nt} | {cnt} |")

    lines.extend([
        "",
        "## Edge Types",
        "",
        "| Type | Count |",
        "|------|-------|",
    ])
    for et, cnt in report.edge_types.items():
        lines.append(f"| {et} | {cnt} |")

    if report.known_weaknesses:
        lines.extend([
            "",
            "## Known Weaknesses",
            "",
        ])
        for w in report.known_weaknesses:
            lines.append(f"- {w}")

    lines.append("")
    path.write_text("\n".join(lines))


def write_report_json(report: DatasetReport, path: Path) -> None:
    """Write report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)


# numpy import for avg computation
import numpy as np
