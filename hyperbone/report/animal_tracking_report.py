"""Animal tracking report generator for public asset tests."""

import json
import numpy as np
from pathlib import Path
from typing import Dict
from collections import Counter


def generate_animal_tracking_report(
    stats: Dict,
    tracking_metrics: Dict,
    output_dir: str,
    animal_name: str = "wolf",
    asset_manifest: Dict = None,
    render_manifest: Dict = None,
) -> Path:
    """Generate the animal tracking validation report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "animal_tracking_summary.md"

    accepted = stats.get("accepted_count", 0)
    rejected = stats.get("rejected_count", 0)
    total = accepted + rejected
    acceptance_rate = accepted / total * 100 if total > 0 else 0

    runtimes = stats.get("runtimes_ms", [])
    avg_runtime = float(np.mean(runtimes)) if runtimes else 0

    topo_score = tracking_metrics.get("topology_stability_score", 0)
    centroid_mean = tracking_metrics.get("centroid_jump_px_mean", 0)
    centroid_p95 = tracking_metrics.get("centroid_jump_px_p95", 0)

    # Verdict
    if accepted >= 5 and topo_score >= 0.50:
        verdict = "PASS"
        verdict_detail = f"At least 5 accepted graphs ({accepted}) and topology stability >= 0.50 ({topo_score:.3f})"
    elif accepted > 0:
        verdict = "PARTIAL"
        verdict_detail = f"Graphs exist but unstable or insufficient (accepted={accepted}, stability={topo_score:.3f})"
    else:
        verdict = "FAIL"
        verdict_detail = "No accepted graphs"

    reason_counter = Counter(stats.get("reject_reasons", []))

    lines = [
        f"# HyperBone Animal Tracking — {animal_name.title()} Smoke Test",
        "",
        "## Asset",
        "",
    ]

    if asset_manifest:
        lines.extend([
            f"- **Source**: {asset_manifest.get('source_name', 'N/A')}",
            f"- **License**: {asset_manifest.get('license', 'N/A')}",
            f"- **Attribution**: {asset_manifest.get('attribution_note', 'N/A')}",
            f"- **Selected animal**: {asset_manifest.get('selected_animal', animal_name)}",
            f"- **Model path**: `{asset_manifest.get('selected_model_path', 'N/A')}`",
            "",
        ])
    else:
        lines.extend([
            "- Source: Quaternius Animated Animal Pack",
            "- License: Public Domain / CC0",
            f"- Animal: {animal_name}",
            "",
        ])

    if render_manifest:
        lines.extend([
            "## Render",
            "",
            f"- Animation: {render_manifest.get('animation_used', 'N/A')}",
            f"- Animation source: {render_manifest.get('animation_source', 'N/A')}",
            f"- FPS: {render_manifest.get('fps', 'N/A')}",
            f"- Duration: {render_manifest.get('duration_sec', 'N/A')}s",
            f"- Resolution: {render_manifest.get('resolution', 'N/A')}",
            "",
        ])

    lines.extend([
        "## Pipeline",
        "",
        "```",
        "Synthetic bbox proposal → HyperBone crop → custom mask → Zhang-Suen thinning → custom graph → quality gate",
        "```",
        "",
        "## Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Video | `{stats.get('video_path', 'N/A')}` |",
        f"| Frames processed | {stats.get('frames_processed', 0)} |",
        f"| Proposals | {stats.get('objects_proposed', 0)} |",
        f"| **Accepted** | **{accepted}** |",
        f"| Rejected | {rejected} |",
        f"| **Acceptance rate** | **{acceptance_rate:.1f}%** |",
        f"| Avg runtime/object | {avg_runtime:.1f} ms |",
        "",
        "## Tracking Stability",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Topology stability** | **{topo_score:.4f}** |",
        f"| Centroid jump mean | {centroid_mean:.1f} px |",
        f"| Centroid jump P95 | {centroid_p95:.1f} px |",
        f"| Node count mean ± std | {tracking_metrics.get('node_count_mean', 0):.1f} ± {tracking_metrics.get('node_count_std', 0):.1f} |",
        f"| Edge count mean ± std | {tracking_metrics.get('edge_count_mean', 0):.1f} ± {tracking_metrics.get('edge_count_std', 0):.1f} |",
        f"| Skeleton length mean ± std | {tracking_metrics.get('skeleton_length_mean', 0):.1f} ± {tracking_metrics.get('skeleton_length_std', 0):.1f} |",
        "",
    ])

    if reason_counter:
        lines.extend([
            "## Top Rejection Reasons",
            "",
            "| Reason | Count |",
            "|--------|-------|",
        ])
        for reason, count in reason_counter.most_common(10):
            lines.append(f"| {reason} | {count} |")
        lines.append("")

    lines.extend([
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        verdict_detail,
        "",
        "## Ownership",
        "",
        "> All skeleton graphs produced by HyperBone-owned mapper.",
        "> skeleton_mapper = hyperbone-custom",
        "> No SAM2, no CoTracker, no Depth Anything.",
        "",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path
