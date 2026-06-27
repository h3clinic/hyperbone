"""Dataset QA report — generates quality analysis for a pseudo-label run."""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def generate_dataset_qa(output_dir: str, run_name: Optional[str] = None) -> Dict:
    """Generate a dataset QA report from a completed pipeline run.

    Args:
        output_dir: Root output directory (e.g. outputs/sam2_m23_repair_100f)
        run_name: Optional label for the run

    Returns:
        QA metrics dict (also written to disk as JSON + markdown)
    """
    out = Path(output_dir)

    # Find quality.jsonl files
    quality_files = list(out.rglob("quality.jsonl"))
    if not quality_files:
        raise FileNotFoundError(f"No quality.jsonl found in {out}")

    records = []
    for qf in quality_files:
        for line in qf.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("No quality records found")

    # Basic stats
    n_total = len(records)
    n_accepted = sum(1 for r in records if r.get("accepted", False))
    n_rejected = n_total - n_accepted

    # Rejection reasons
    all_reasons = []
    for r in records:
        all_reasons.extend(r.get("reject_reasons", []))
    reason_counter = Counter(all_reasons)
    top_reasons = reason_counter.most_common(10)

    # Node/edge stats
    node_counts = [r.get("skeleton_node_count", 0) for r in records]
    edge_counts = [r.get("skeleton_edge_count", 0) for r in records]

    # Repair stats
    comp_before = [r.get("components_before_repair", 1) for r in records if "components_before_repair" in r]
    comp_after = [r.get("components_after_repair", 1) for r in records if "components_after_repair" in r]
    bridges = [r.get("bridges_added", 0) for r in records if "bridges_added" in r]
    merges = [r.get("nodes_merged", 0) for r in records if "nodes_merged" in r]
    spurs = [r.get("spurs_pruned", 0) for r in records if "spurs_pruned" in r]

    # LCR stats
    lcr_values = [r.get("largest_component_ratio", 0) for r in records]

    # Mask area stats
    area_ratios = [r.get("mask_area_ratio", 0) for r in records if "mask_area_ratio" in r]

    # Per-frame stats
    frames = set()
    per_frame_accepted = Counter()
    for r in records:
        fi = r.get("frame_idx", 0)
        frames.add(fi)
        if r.get("accepted", False):
            per_frame_accepted[fi] += 1

    n_frames = len(frames)
    accepted_per_frame = list(per_frame_accepted.values()) if per_frame_accepted else [0]

    # Mode info
    cleanup_applied = any(r.get("mask_cleanup_applied", False) for r in records)
    repair_applied = any(r.get("graph_repair_applied", False) for r in records)

    # Disk usage
    disk_bytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())

    qa = {
        "run_name": run_name or out.name,
        "output_dir": str(out),
        "frames_processed": n_frames,
        "objects_detected": n_total,
        "accepted_graphs": n_accepted,
        "rejected_graphs": n_rejected,
        "acceptance_rate_pct": round(n_accepted / n_total * 100, 1) if n_total > 0 else 0,
        "top_rejection_reasons": [{"reason": r, "count": c} for r, c in top_reasons],
        "node_count_mean": round(float(np.mean(node_counts)), 1),
        "node_count_median": int(np.median(node_counts)),
        "node_count_min": int(np.min(node_counts)) if node_counts else 0,
        "node_count_max": int(np.max(node_counts)) if node_counts else 0,
        "edge_count_mean": round(float(np.mean(edge_counts)), 1),
        "edge_count_median": int(np.median(edge_counts)),
        "components_before_repair_mean": round(float(np.mean(comp_before)), 2) if comp_before else None,
        "components_after_repair_mean": round(float(np.mean(comp_after)), 2) if comp_after else None,
        "bridges_added_total": int(np.sum(bridges)) if bridges else 0,
        "bridges_added_mean": round(float(np.mean(bridges)), 2) if bridges else 0,
        "nodes_merged_total": int(np.sum(merges)) if merges else 0,
        "spurs_pruned_total": int(np.sum(spurs)) if spurs else 0,
        "largest_component_ratio_mean": round(float(np.mean(lcr_values)), 3) if lcr_values else None,
        "mask_area_ratio_mean": round(float(np.mean(area_ratios)), 4) if area_ratios else None,
        "mask_area_ratio_median": round(float(np.median(area_ratios)), 4) if area_ratios else None,
        "accepted_per_frame_mean": round(float(np.mean(accepted_per_frame)), 1),
        "accepted_per_frame_max": int(np.max(accepted_per_frame)),
        "mask_cleanup_enabled": cleanup_applied,
        "graph_repair_enabled": repair_applied,
        "disk_usage_mb": round(disk_bytes / (1024 * 1024), 1),
    }

    # Write JSON
    qa_json_path = out / "dataset_qa.json"
    qa_json_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")

    # Write markdown
    qa_md = _format_qa_markdown(qa)
    qa_md_path = out / "dataset_qa.md"
    qa_md_path.write_text(qa_md, encoding="utf-8")

    print(f"[DatasetQA] Report: {qa_md_path}")
    print(f"[DatasetQA] JSON:   {qa_json_path}")

    return qa


def _format_qa_markdown(qa: Dict) -> str:
    lines = [
        f"# Dataset QA Report: {qa['run_name']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Frames processed | {qa['frames_processed']} |",
        f"| Objects detected | {qa['objects_detected']} |",
        f"| **Accepted graphs** | **{qa['accepted_graphs']}** |",
        f"| Rejected graphs | {qa['rejected_graphs']} |",
        f"| **Acceptance rate** | **{qa['acceptance_rate_pct']}%** |",
        f"| Mask cleanup enabled | {qa['mask_cleanup_enabled']} |",
        f"| Graph repair enabled | {qa['graph_repair_enabled']} |",
        f"| Disk usage | {qa['disk_usage_mb']} MB |",
        "",
        "## Graph Statistics",
        "",
        "| Metric | Mean | Median | Min | Max |",
        "|--------|------|--------|-----|-----|",
        f"| Node count | {qa['node_count_mean']} | {qa['node_count_median']} | {qa['node_count_min']} | {qa['node_count_max']} |",
        f"| Edge count | {qa['edge_count_mean']} | {qa['edge_count_median']} | — | — |",
        "",
        "## Repair Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Avg components before repair | {qa['components_before_repair_mean']} |",
        f"| Avg components after repair | {qa['components_after_repair_mean']} |",
        f"| Bridges added (total) | {qa['bridges_added_total']} |",
        f"| Bridges added (mean/object) | {qa['bridges_added_mean']} |",
        f"| Nodes merged (total) | {qa['nodes_merged_total']} |",
        f"| Spurs pruned (total) | {qa['spurs_pruned_total']} |",
        f"| Largest component ratio (mean) | {qa['largest_component_ratio_mean']} |",
        "",
        "## Mask Statistics",
        "",
        f"| Mask area ratio mean | {qa['mask_area_ratio_mean']} |",
        f"| Mask area ratio median | {qa['mask_area_ratio_median']} |",
        "",
        "## Per-Frame",
        "",
        f"| Accepted per frame (mean) | {qa['accepted_per_frame_mean']} |",
        f"| Accepted per frame (max) | {qa['accepted_per_frame_max']} |",
        "",
        "## Top Rejection Reasons",
        "",
        "| Reason | Count |",
        "|--------|-------|",
    ]
    for item in qa["top_rejection_reasons"]:
        lines.append(f"| {item['reason']} | {item['count']} |")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate dataset QA report")
    parser.add_argument("--dir", required=True, help="Pipeline output directory")
    parser.add_argument("--name", default=None, help="Run name label")
    args = parser.parse_args()
    generate_dataset_qa(args.dir, args.name)
