"""Compare HyperBone-owned mapper vs Grounded-SAM2 path.

If SAM2 is not available, compares against previously saved output.
"""

import sys, json, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.io.video import get_video_info, sample_frames
from hyperbone.objects.proposals import ObjectProposal
from hyperbone.cv.hyperbone_graph import map_proposal_to_skeleton
from collections import Counter


def load_graph_jsonl(path: str):
    """Load graph records from a JSONL file."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_records(records, label: str) -> dict:
    """Summarize a list of graph records."""
    if not records:
        return {"label": label, "count": 0}

    accepted = [r for r in records if r.get("accepted", False)]
    rejected = [r for r in records if not r.get("accepted", False)]

    node_counts = [r.get("node_count", 0) for r in records]
    edge_counts = [r.get("edge_count", 0) for r in records]
    runtimes = [r.get("runtime_ms", 0) for r in records]

    reject_reasons = []
    for r in rejected:
        reject_reasons.extend(r.get("reject_reasons", []))

    reason_counter = Counter(reject_reasons)

    labels_accepted = Counter(r.get("object_label", "?") for r in accepted)
    labels_rejected = Counter(r.get("object_label", "?") for r in rejected)

    return {
        "label": label,
        "count": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / len(records) * 100 if records else 0,
        "avg_nodes": float(np.mean(node_counts)) if node_counts else 0,
        "avg_edges": float(np.mean(edge_counts)) if edge_counts else 0,
        "avg_runtime_ms": float(np.mean(runtimes)) if runtimes else 0,
        "top_reject_reasons": reason_counter.most_common(5),
        "labels_accepted": labels_accepted.most_common(10),
        "labels_rejected": labels_rejected.most_common(10),
    }


def print_comparison(summary_a: dict, summary_b: dict):
    """Print side-by-side comparison."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON: {summary_a['label']} vs {summary_b['label']}")
    print(f"{'='*60}\n")

    header = f"{'Metric':<25} {'Path A':>15} {'Path B':>15}"
    print(header)
    print("-" * 55)

    metrics = [
        ("Objects proposed", "count"),
        ("Accepted", "accepted"),
        ("Rejected", "rejected"),
        ("Acceptance rate (%)", "acceptance_rate"),
        ("Avg nodes", "avg_nodes"),
        ("Avg edges", "avg_edges"),
        ("Avg runtime (ms)", "avg_runtime_ms"),
    ]

    for name, key in metrics:
        va = summary_a.get(key, 0)
        vb = summary_b.get(key, 0)
        if isinstance(va, float):
            print(f"{name:<25} {va:>15.1f} {vb:>15.1f}")
        else:
            print(f"{name:<25} {va:>15} {vb:>15}")

    print(f"\n--- Top reject reasons: {summary_a['label']} ---")
    for reason, count in summary_a.get("top_reject_reasons", []):
        print(f"  {count:>3}x  {reason}")

    print(f"\n--- Top reject reasons: {summary_b['label']} ---")
    for reason, count in summary_b.get("top_reject_reasons", []):
        print(f"  {count:>3}x  {reason}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare HyperBone-owned mapper vs Grounded-SAM2 path"
    )
    parser.add_argument("--owned-output", required=True,
                        help="Path to HyperBone-owned mapper output (graphs.jsonl)")
    parser.add_argument("--sam2-output", default=None,
                        help="Path to Grounded-SAM2 output (graphs.jsonl or quality.jsonl)")
    parser.add_argument("--sam2-previous", default=None,
                        help="Path to previously saved SAM2 output directory")
    parser.add_argument("--out", default="outputs/comparison",
                        help="Output directory for comparison report")

    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load owned path results
    owned_path = Path(args.owned_output)
    if owned_path.is_dir():
        owned_jsonl = owned_path / "graphs" / "graphs.jsonl"
    else:
        owned_jsonl = owned_path

    if not owned_jsonl.exists():
        print(f"ERROR: Owned output not found: {owned_jsonl}")
        sys.exit(1)

    owned_records = load_graph_jsonl(str(owned_jsonl))
    owned_summary = summarize_records(owned_records, "HyperBone-owned")

    # Load SAM2 path results (if available)
    sam2_records = []
    sam2_label = "Grounded-SAM2"

    if args.sam2_output:
        sam2_path = Path(args.sam2_output)
        if sam2_path.is_dir():
            # Try graphs.jsonl first, then quality.jsonl
            candidates = [
                sam2_path / "graphs" / "graphs.jsonl",
                sam2_path / "graphs.jsonl",
                sam2_path / "quality.jsonl",
            ]
            for c in candidates:
                if c.exists():
                    sam2_records = load_graph_jsonl(str(c))
                    break
        elif sam2_path.exists():
            sam2_records = load_graph_jsonl(str(sam2_path))
    elif args.sam2_previous:
        prev = Path(args.sam2_previous)
        candidates = [
            prev / "videos" / "HyperVid" / "quality.jsonl",
            prev / "quality.jsonl",
            prev / "graphs.jsonl",
        ]
        for c in candidates:
            if c.exists():
                sam2_records = load_graph_jsonl(str(c))
                sam2_label = f"Grounded-SAM2 (saved: {c.parent.name})"
                break

    if not sam2_records:
        print("[Compare] No SAM2 output available. Showing owned-mapper stats only.")
        print_comparison(owned_summary, {"label": "(no SAM2 data)", "count": 0,
                                         "accepted": 0, "rejected": 0,
                                         "acceptance_rate": 0, "avg_nodes": 0,
                                         "avg_edges": 0, "avg_runtime_ms": 0,
                                         "top_reject_reasons": [],
                                         "labels_accepted": [],
                                         "labels_rejected": []})
    else:
        sam2_summary = summarize_records(sam2_records, sam2_label)
        print_comparison(owned_summary, sam2_summary)

        # Save comparison JSON
        comparison = {
            "owned": {k: v for k, v in owned_summary.items() if k != "top_reject_reasons"},
            "sam2": {k: v for k, v in sam2_summary.items() if k != "top_reject_reasons"},
        }
        with open(out / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2, default=str)

    # Save owned summary
    with open(out / "owned_summary.json", "w") as f:
        json.dump({k: v for k, v in owned_summary.items()
                   if k != "top_reject_reasons"}, f, indent=2, default=str)

    print(f"\n[Compare] Output: {out}")


if __name__ == "__main__":
    main()
