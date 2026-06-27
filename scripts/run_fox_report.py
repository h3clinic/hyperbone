"""Run Fox tracking report generation."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.tracking.simple_graph_track import compute_tracking_metrics
from hyperbone.report.fox_tracking_report import generate_fox_tracking_report


def main():
    base = Path("outputs/synthetic/fox_hyperbone")
    graphs_path = base / "graphs" / "graphs.jsonl"

    # Load graph records
    with open(graphs_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Compute tracking metrics
    tracking_metrics = compute_tracking_metrics(records)

    # Print metrics
    print("[Fox Tracking Metrics]")
    for k, v in tracking_metrics.items():
        if k != "per_frame_metrics":
            print(f"  {k}: {v}")

    # Load manifests
    asset_manifest = None
    asset_path = Path("assets/gltf_samples/Fox/asset_manifest.json")
    if asset_path.exists():
        with open(asset_path) as f:
            asset_manifest = json.load(f)

    render_manifest = None
    render_path = Path("outputs/synthetic/fox_render_manifest.json")
    if render_path.exists():
        with open(render_path) as f:
            render_manifest = json.load(f)

    # Build stats compatible with report generator
    runtimes = [r.get("runtime_ms", 0) for r in records]
    reject_reasons = []
    for r in records:
        reject_reasons.extend(r.get("reject_reasons", []))

    stats = {
        "video_path": str(Path("outputs/synthetic/fox_5s.mp4").resolve()),
        "resolution": "640x480",
        "proposal_source": "manual_synthetic",
        "text_prompt": "",
        "frames_processed": len(set(r["frame_idx"] for r in records)),
        "objects_proposed": len(records),
        "accepted_count": sum(1 for r in records if r.get("accepted")),
        "rejected_count": sum(1 for r in records if not r.get("accepted")),
        "reject_reasons": reject_reasons,
        "runtimes_ms": runtimes,
        "labels_accepted": [r["object_label"] for r in records if r.get("accepted")],
        "labels_rejected": [r["object_label"] for r in records if not r.get("accepted")],
        "node_counts": [r.get("node_count", 0) for r in records],
        "edge_counts": [r.get("edge_count", 0) for r in records],
        "output_dir": str(base),
    }

    # Generate report
    report_path = generate_fox_tracking_report(
        stats, tracking_metrics, str(base),
        asset_manifest=asset_manifest,
        render_manifest=render_manifest,
    )
    print(f"\n[Report] Written: {report_path}")

    # Verify all records are hyperbone-custom
    all_owned = all(r["skeleton_mapper"] == "hyperbone-custom" for r in records)
    print(f"\n[Verify] All skeleton_mapper='hyperbone-custom': {all_owned}")
    print(f"[Verify] Accepted: {stats['accepted_count']}/{stats['objects_proposed']}")
    print(f"[Verify] Topology stability: {tracking_metrics['topology_stability_score']}")

    # Verdict
    if stats["accepted_count"] >= 5 and tracking_metrics["topology_stability_score"] >= 0.50:
        print("\n[VERDICT] PASS")
    elif stats["accepted_count"] > 0:
        print("\n[VERDICT] PARTIAL")
    else:
        print("\n[VERDICT] FAIL")


if __name__ == "__main__":
    main()
