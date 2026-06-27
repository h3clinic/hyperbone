"""Summary report generation for batch runs."""

from pathlib import Path
from typing import Dict, List
from collections import Counter


def generate_summary(
    batch_stats: Dict,
    output_dir: str,
    filename: str = "summary.md",
) -> Path:
    """Write a human-readable markdown summary of a batch run.

    Args:
        batch_stats: Dict with keys:
            videos_processed, videos_failed, frames_sampled,
            objects_detected, accepted_count, rejected_count,
            reject_reasons (list of strings), node_counts (list),
            edge_counts (list), errors (list), output_dir
        output_dir: Where to write the summary.
        filename: Output filename.

    Returns:
        Path to the written summary file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename

    total = batch_stats.get("accepted_count", 0) + batch_stats.get("rejected_count", 0)
    acceptance_rate = (
        batch_stats["accepted_count"] / total * 100 if total > 0 else 0
    )

    # Top rejection reasons
    reason_counter = Counter(batch_stats.get("reject_reasons", []))
    top_reasons = reason_counter.most_common(10)

    # Averages
    node_counts = batch_stats.get("node_counts", [])
    edge_counts = batch_stats.get("edge_counts", [])
    avg_nodes = sum(node_counts) / len(node_counts) if node_counts else 0
    avg_edges = sum(edge_counts) / len(edge_counts) if edge_counts else 0

    lines = [
        "# HyperBone Batch Run Summary",
        "",
        "## Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Videos processed | {batch_stats.get('videos_processed', 0)} |",
        f"| Videos failed | {batch_stats.get('videos_failed', 0)} |",
        f"| Frames sampled | {batch_stats.get('frames_sampled', 0)} |",
        f"| Objects detected | {batch_stats.get('objects_detected', 0)} |",
        f"| **Accepted graphs** | **{batch_stats.get('accepted_count', 0)}** |",
        f"| Rejected graphs | {batch_stats.get('rejected_count', 0)} |",
        f"| **Acceptance rate** | **{acceptance_rate:.1f}%** |",
        f"| Average node count | {avg_nodes:.1f} |",
        f"| Average edge count | {avg_edges:.1f} |",
        "",
    ]

    if top_reasons:
        lines += [
            "## Top Rejection Reasons",
            "",
            "| Reason | Count |",
            "|--------|-------|",
        ]
        for reason, count in top_reasons:
            # Truncate long reasons for table readability
            short = reason[:60] + "..." if len(reason) > 60 else reason
            lines.append(f"| {short} | {count} |")
        lines.append("")

    if batch_stats.get("errors"):
        lines += [
            "## Errors",
            "",
        ]
        for err in batch_stats["errors"][:20]:
            lines.append(f"- {err}")
        lines.append("")

    lines += [
        "## Output Paths",
        "",
        f"- Output directory: `{batch_stats.get('output_dir', output_dir)}`",
        f"- Accepted graphs: `accepted_graphs.jsonl`",
        f"- Rejected graphs: `rejected_graphs.jsonl`",
        f"- Dataset index: `dataset_index.jsonl`",
        f"- NPZ data: per-video `npz/` folders",
        "",
        "## Next Recommended Action",
        "",
    ]

    if acceptance_rate < 10:
        lines.append("- Acceptance rate is very low. Consider relaxing thresholds or improving mask generation (integrate SAM2).")
    elif acceptance_rate < 50:
        lines.append("- Acceptance rate is moderate. Review rejected overlays to tune thresholds before scaling up.")
    else:
        lines.append("- Acceptance rate is good. Ready to scale to more videos or integrate SAM2 for better masks.")

    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
