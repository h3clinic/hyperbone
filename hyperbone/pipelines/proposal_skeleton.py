"""Proposal-driven skeleton pipeline — HyperBone-owned mapper.

Pipeline flow:
  load video → sample frames → get proposals → for each proposal:
    crop bbox → resize → custom mask → custom thinning → graph extraction →
    graph repair → quality scoring → export graph JSONL → overlay

DINO is only used for object labels + bboxes. Skeleton mapping is HyperBone-owned.
"""

import json
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

from hyperbone.io.video import get_video_info, sample_frames
from hyperbone.objects.proposals import ObjectProposal
from hyperbone.objects.manual_loader import load_proposals_jsonl, clip_proposal_to_frame
from hyperbone.cv.hyperbone_graph import map_proposal_to_skeleton
from hyperbone.quality.score import score_graph


def run_proposal_skeleton(
    video_path: str,
    output_dir: str,
    proposals_path: Optional[str] = None,
    proposal_source: str = "manual",
    text_prompt: str = "person. hand. arm. leg. head. torso.",
    sample_fps: float = 1.0,
    skip_start_sec: float = 0.0,
    skip_end_sec: float = 0.0,
    max_side: int = 384,
    thinning_algorithm: str = "zhang-suen",
    min_branch_length: int = 10,
    mask_method: str = "combined",
    device: str = "cuda",
    grounding_model: str = "IDEA-Research/grounding-dino-tiny",
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    max_proposals_per_frame: int = 20,
) -> Dict:
    """Run the HyperBone-owned proposal→skeleton pipeline.

    Args:
        video_path: Path to input video.
        output_dir: Where to write outputs.
        proposals_path: Path to manual proposals JSONL (if proposal_source='manual').
        proposal_source: 'manual' or 'groundingdino'.
        text_prompt: DINO text prompt (if proposal_source='groundingdino').
        sample_fps: Frame sampling rate.
        skip_start_sec: Skip this many seconds from the start.
        skip_end_sec: Skip this many seconds from the end.
        max_side: Max side for crop resize.
        thinning_algorithm: 'zhang-suen' or 'guo-hall'.
        min_branch_length: Prune branches shorter than this.
        mask_method: Custom mask method.
        device: Device for DINO (if used).
        grounding_model: HuggingFace model ID for DINO.
        box_threshold: DINO box confidence threshold.
        text_threshold: DINO text confidence threshold.
        max_proposals_per_frame: Max proposals per frame from DINO.

    Returns:
        Dict with run statistics.
    """
    info = get_video_info(video_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Output paths
    graph_dir = out / "graphs"
    graph_dir.mkdir(exist_ok=True)
    overlay_dir = out / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    accepted_dir = out / "accepted"
    accepted_dir.mkdir(exist_ok=True)
    rejected_dir = out / "rejected"
    rejected_dir.mkdir(exist_ok=True)

    # Load proposals source
    manual_proposals: List[ObjectProposal] = []
    dino_adapter = None

    if proposal_source == "manual":
        if not proposals_path:
            raise ValueError("--proposals path required for proposal_source='manual'")
        manual_proposals = load_proposals_jsonl(proposals_path)
        print(f"[HyperBone] Loaded {len(manual_proposals)} manual proposals from {proposals_path}")
    elif proposal_source == "groundingdino":
        from hyperbone.objects.dino_adapter import DINOProposalAdapter
        dino_adapter = DINOProposalAdapter(
            model_id=grounding_model,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_proposals_per_frame=max_proposals_per_frame,
        )
        print(f"[HyperBone] Using GroundingDINO for proposals: {grounding_model}")
        print(f"[HyperBone] Text prompt: {text_prompt}")
    else:
        raise ValueError(f"Unknown proposal_source: {proposal_source}")

    print(f"[HyperBone] Video: {info['path']}")
    print(f"[HyperBone] Resolution: {info['width']}x{info['height']}")
    print(f"[HyperBone] Skeleton mapper: hyperbone-custom")
    print(f"[HyperBone] Thinning: {thinning_algorithm}")
    print(f"[HyperBone] Max side: {max_side}px")
    print()

    # Stats
    stats = {
        "video_path": str(info["path"]),
        "resolution": f"{info['width']}x{info['height']}",
        "proposal_source": proposal_source,
        "text_prompt": text_prompt if proposal_source == "groundingdino" else "",
        "frames_processed": 0,
        "objects_proposed": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "reject_reasons": [],
        "node_counts": [],
        "edge_counts": [],
        "runtimes_ms": [],
        "labels_accepted": [],
        "labels_rejected": [],
    }

    # For manual proposals, compute the max frame_idx so we can stop early
    max_proposal_frame = max((p.frame_idx for p in manual_proposals), default=-1) \
        if proposal_source == "manual" else float("inf")

    graph_jsonl_path = graph_dir / "graphs.jsonl"
    quality_jsonl_path = out / "quality.jsonl"

    with open(graph_jsonl_path, "w") as gf, open(quality_jsonl_path, "w") as qf:
        for frame_idx, ts, frame in sample_frames(
            video_path, sample_fps,
            skip_start_sec=skip_start_sec,
            skip_end_sec=skip_end_sec,
        ):
            # Early exit: stop if we've passed all manual proposal frames
            if proposal_source == "manual" and frame_idx > max_proposal_frame:
                break

            h, w = frame.shape[:2]

            # Get proposals for this frame
            if proposal_source == "manual":
                frame_proposals = [
                    p for p in manual_proposals if p.frame_idx == frame_idx
                ]
            else:
                frame_proposals = dino_adapter.generate(frame, frame_idx, text_prompt)

            if not frame_proposals:
                stats["frames_processed"] += 1
                continue

            # Clip proposals to frame
            clipped_proposals = []
            for p in frame_proposals:
                clipped = clip_proposal_to_frame(p, w, h)
                if clipped is not None:
                    clipped_proposals.append(clipped)

            stats["objects_proposed"] += len(clipped_proposals)

            # Process each proposal
            for p in clipped_proposals:
                result = map_proposal_to_skeleton(
                    frame, p,
                    max_side=max_side,
                    thinning_algorithm=thinning_algorithm,
                    min_branch_length=min_branch_length,
                    mask_method=mask_method,
                )

                meta = result["metadata"]
                graph = result["graph"]
                mask = result["mask"]

                # Quality scoring
                quality = score_graph(
                    mask=mask,
                    graph=graph,
                    frame_shape=(h, w),
                    bbox=p.bbox_xyxy,
                )

                accepted = quality["accepted"]
                reject_reasons = quality.get("reject_reasons", [])

                # Build graph record
                graph_record = {
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(ts, 3),
                    "object_id": p.object_id,
                    "object_label": p.label,
                    "object_label_source": p.label_source,
                    "object_label_confidence": round(p.label_confidence, 4),
                    "bbox_xywh": list(p.bbox_xywh),
                    "skeleton_mapper": "hyperbone-custom",
                    "mask_backend": "hyperbone-custom-mask",
                    "thinning_algorithm": thinning_algorithm,
                    "graph_builder": "hyperbone-pixel-trace-v0",
                    "node_count": meta.get("node_count", 0),
                    "edge_count": meta.get("edge_count", 0),
                    "nodes": graph["nodes"],
                    "edges": graph["edges"],
                    "accepted": accepted,
                    "reject_reasons": reject_reasons,
                    "runtime_ms": round(meta.get("skeleton_runtime_ms", 0), 2),
                }
                gf.write(json.dumps(graph_record) + "\n")

                # Quality record
                quality_record = {
                    "frame_idx": frame_idx,
                    "object_id": p.object_id,
                    "object_label": p.label,
                    "skeleton_mapper": "hyperbone-custom",
                    "accepted": accepted,
                    "reject_reasons": reject_reasons,
                    **{k: v for k, v in quality.items() if k not in ("accepted", "reject_reasons")},
                }
                qf.write(json.dumps(quality_record) + "\n")

                # Update stats
                if accepted:
                    stats["accepted_count"] += 1
                    stats["labels_accepted"].append(p.label)
                    _write_accepted_graph(accepted_dir, frame_idx, p.object_id, graph_record)
                else:
                    stats["rejected_count"] += 1
                    stats["labels_rejected"].append(p.label)
                    stats["reject_reasons"].extend(reject_reasons)
                    _write_rejected_graph(rejected_dir, frame_idx, p.object_id, graph_record)

                stats["node_counts"].append(meta.get("node_count", 0))
                stats["edge_counts"].append(meta.get("edge_count", 0))
                stats["runtimes_ms"].append(meta.get("skeleton_runtime_ms", 0))

                # Overlay
                _save_overlay(
                    overlay_dir, frame, p, graph, mask, result.get("crop_meta", {}),
                    frame_idx, accepted,
                )

            stats["frames_processed"] += 1

    # Print summary
    total = stats["accepted_count"] + stats["rejected_count"]
    acceptance_rate = stats["accepted_count"] / total * 100 if total > 0 else 0
    avg_runtime = np.mean(stats["runtimes_ms"]) if stats["runtimes_ms"] else 0

    print(f"\n[HyperBone] Done.")
    print(f"  Frames processed:   {stats['frames_processed']}")
    print(f"  Objects proposed:    {stats['objects_proposed']}")
    print(f"  Accepted:           {stats['accepted_count']}")
    print(f"  Rejected:           {stats['rejected_count']}")
    print(f"  Acceptance rate:    {acceptance_rate:.1f}%")
    print(f"  Avg runtime/object: {avg_runtime:.1f}ms")
    print(f"  Output: {out}")

    stats["acceptance_rate"] = round(acceptance_rate, 2)
    stats["avg_runtime_ms"] = round(avg_runtime, 2)
    stats["output_dir"] = str(out)

    return stats


def _write_accepted_graph(dir_path: Path, frame_idx: int, obj_id: int, record: Dict):
    """Write a single accepted graph record."""
    path = dir_path / f"frame{frame_idx:06d}_obj{obj_id:03d}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def _write_rejected_graph(dir_path: Path, frame_idx: int, obj_id: int, record: Dict):
    """Write a single rejected graph record."""
    path = dir_path / f"frame{frame_idx:06d}_obj{obj_id:03d}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def _save_overlay(
    overlay_dir: Path,
    frame: np.ndarray,
    proposal: ObjectProposal,
    graph: Dict,
    mask: Optional[np.ndarray],
    crop_meta: Dict,
    frame_idx: int,
    accepted: bool,
):
    """Save an overlay image showing the skeleton on the frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Draw bbox
    x, y, bw, bh = proposal.bbox_xywh
    color = (0, 255, 0) if accepted else (0, 0, 255)
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)

    # Draw label
    label_text = f"{proposal.label} ({proposal.label_confidence:.2f})"
    cv2.putText(vis, label_text, (x, max(y - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw skeleton graph
    node_map = {}
    for node in graph.get("nodes", []):
        nx, ny = node["xy"]
        nx, ny = int(nx), int(ny)
        if 0 <= nx < w and 0 <= ny < h:
            node_map[node["id"]] = (nx, ny)
            node_color = (255, 255, 0) if node.get("type") == "junction" else (0, 255, 255)
            cv2.circle(vis, (nx, ny), 3, node_color, -1)

    for edge in graph.get("edges", []):
        p = node_map.get(edge["parent"])
        c = node_map.get(edge["child"])
        if p and c:
            cv2.line(vis, p, c, (255, 128, 0), 1)

    # Status text
    status = "ACCEPTED" if accepted else "REJECTED"
    cv2.putText(vis, status, (x, y + bh + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    fname = f"frame{frame_idx:06d}_obj{proposal.object_id:03d}.jpg"
    cv2.imwrite(str(overlay_dir / fname), vis)
