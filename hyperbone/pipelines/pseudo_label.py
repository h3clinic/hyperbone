"""Pseudo-label pipeline — orchestrates the full video-to-skeleton factory.

Milestone 2.2: mask cleanup + graph repair for improved SAM2 acceptance.
"""

import hashlib
from pathlib import Path
from typing import Optional, Dict, List

from hyperbone.io.video import get_video_info, sample_frames
from hyperbone.io.manifest import write_manifest
from hyperbone.io.dataset_index import DatasetIndexWriter, build_index_row
from hyperbone.cv.frames import save_frame
from hyperbone.cv.masks import get_mask_generator, save_mask
from hyperbone.cv.mask_cleanup import clean_mask_for_skeleton
from hyperbone.cv.skeletonize import skeletonize_mask
from hyperbone.cv.graph import skeleton_to_graph
from hyperbone.cv.graph_repair import repair_graph
from hyperbone.cv.overlay import draw_overlay, save_overlay
from hyperbone.export.jsonl import JSONLWriter, graph_to_record
from hyperbone.export.npz import export_npz
from hyperbone.export.blender import export_blender_script
from hyperbone.quality.score import score_graph, quality_score_numeric


def run_pseudo_label(
    video_path: str,
    output_dir: str,
    sample_fps: float = 1.0,
    mask_backend: str = "threshold",
    min_branch_length: int = 10,
    save_frames: bool = True,
    save_masks: bool = True,
    save_overlays: bool = True,
    max_frames: Optional[int] = None,
    quality_thresholds: Optional[Dict] = None,
    mask_backend_kwargs: Optional[Dict] = None,
    enable_mask_cleanup: Optional[bool] = None,
    enable_graph_repair: Optional[bool] = None,
    bridge_gap_px: float = 12.0,
    cleanup_close_kernel: int = 5,
    keep_largest_component: bool = True,
    node_merge_radius: float = 4.0,
    spur_prune_length: int = 8,
    skip_start_sec: float = 0.0,
    skip_end_sec: float = 0.0,
) -> Dict:
    """Run the full pseudo-label pipeline on a single video.

    Args:
        enable_mask_cleanup: Clean masks before skeletonization.
            None = auto (enabled for sam2, disabled for threshold).
        enable_graph_repair: Bridge disconnected graph components.
            None = auto (enabled for sam2, disabled for threshold).
        bridge_gap_px: Max pixel gap to bridge between components.
        cleanup_close_kernel: Morphological closing kernel size.
        keep_largest_component: Keep only largest mask component.
        mask_backend_kwargs: Extra kwargs passed to the mask generator factory.

    Returns dict with run statistics including quality breakdown.
    """
    out = Path(output_dir)
    frames_dir = out / "frames"
    masks_dir = out / "masks"
    graphs_dir = out / "graphs"
    overlays_dir = out / "overlays"
    rejected_overlays_dir = out / "rejected" / "overlays"
    npz_dir = out / "npz"
    blender_dir = out / "blender"

    # Video info
    info = get_video_info(video_path)
    video_id = hashlib.md5(info["path"].encode()).hexdigest()[:12]

    print(f"[HyperBone] Video: {info['path']}")
    print(f"[HyperBone] Resolution: {info['width']}x{info['height']}")
    print(f"[HyperBone] Duration: {info['duration_sec']:.1f}s @ {info['fps']:.1f} fps")
    print(f"[HyperBone] Sampling at {sample_fps} fps")
    print(f"[HyperBone] Output: {out}")
    print()

    # Initialize
    mask_gen = get_mask_generator(mask_backend, **(mask_backend_kwargs or {}))

    # Auto-detect cleanup/repair: enabled by default for sam2
    use_cleanup = enable_mask_cleanup if enable_mask_cleanup is not None else (mask_backend == "sam2")
    use_repair = enable_graph_repair if enable_graph_repair is not None else (mask_backend == "sam2")

    accepted_records: List[Dict] = []
    rejected_records: List[Dict] = []
    stats = {
        "video_id": video_id,
        "frames_processed": 0,
        "objects_found": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "reject_reasons": [],
        "node_counts": [],
        "edge_counts": [],
    }

    accepted_writer = JSONLWriter(str(graphs_dir), "accepted_graphs.jsonl")
    rejected_writer = JSONLWriter(str(graphs_dir), "rejected_graphs.jsonl")
    quality_writer = JSONLWriter(str(out), "quality.jsonl")

    try:
        for frame_idx, timestamp, frame in sample_frames(
            video_path, sample_fps,
            skip_start_sec=skip_start_sec,
            skip_end_sec=skip_end_sec,
        ):
            if max_frames and stats["frames_processed"] >= max_frames:
                break

            stats["frames_processed"] += 1
            frame_shape = frame.shape[:2]

            # Save raw frame
            if save_frames:
                save_frame(frame, str(frames_dir), frame_idx)

            # Generate masks
            mask_records = mask_gen.generate(frame)

            for mask_rec in mask_records:
                object_id = mask_rec["object_id"]
                mask = mask_rec["mask"]
                mask_confidence = mask_rec["confidence"]
                mask_source = mask_rec["source"]
                mask_touches_edge = mask_rec["touches_edge"]
                mask_bbox = mask_rec["bbox"]  # [x, y, w, h]
                object_class = mask_rec.get("object_class", None)
                label_confidence = mask_rec.get("label_confidence", None)
                stats["objects_found"] += 1

                # Mask cleanup
                cleanup_info = None
                skel_mask = mask
                if use_cleanup:
                    cleanup_result = clean_mask_for_skeleton(
                        mask,
                        close_kernel=cleanup_close_kernel,
                        keep_largest=keep_largest_component,
                    )
                    skel_mask = cleanup_result["clean_mask"]
                    cleanup_info = cleanup_result

                # Skeletonize (on cleaned mask)
                skeleton = skeletonize_mask(skel_mask)

                # Extract graph
                graph = skeleton_to_graph(skeleton, min_branch_length=min_branch_length)

                # Graph repair
                repair_info = None
                if use_repair and graph["nodes"]:
                    repair_result = repair_graph(
                        graph,
                        mask_shape=frame_shape,
                        bridge_gap_px=bridge_gap_px,
                        merge_distance=node_merge_radius,
                        min_branch_length=min_branch_length,
                        spur_prune_length=spur_prune_length,
                    )
                    graph = repair_result["graph"]
                    repair_info = {k: v for k, v in repair_result.items() if k != "graph"}

                if not graph["nodes"]:
                    continue

                # Compute bbox from mask (x_min, y_min, x_max, y_max)
                ys, xs = mask.nonzero() if mask.ndim == 2 else ([], [])
                if hasattr(ys, '__len__') and len(ys) > 0:
                    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                else:
                    bbox = None

                # Quality scoring
                quality = score_graph(
                    mask=mask,
                    graph=graph,
                    frame_shape=frame_shape,
                    bbox=bbox,
                    thresholds=quality_thresholds,
                )
                q_score = quality_score_numeric(quality)
                quality["quality_score"] = q_score

                # Mask diagnostics
                quality["mask_backend"] = mask_source
                quality["mask_confidence"] = mask_confidence
                quality["touches_edge"] = mask_touches_edge
                quality["mask_bbox_xywh"] = mask_bbox

                # Object label (from grounded detection)
                if object_class is not None:
                    quality["object_class"] = object_class
                    quality["label_confidence"] = label_confidence

                # Repair diagnostics
                quality["mask_cleanup_applied"] = use_cleanup
                quality["graph_repair_applied"] = use_repair
                if repair_info:
                    quality["components_before_repair"] = repair_info["components_before"]
                    quality["components_after_repair"] = repair_info["components_after"]
                    quality["bridges_added"] = repair_info["bridges_added"]
                    quality["nodes_merged"] = repair_info["nodes_merged"]
                    quality["spurs_pruned"] = repair_info["spurs_pruned"]
                if cleanup_info:
                    quality["cleanup_components_removed"] = cleanup_info["components_removed"]
                    quality["cleanup_holes_filled"] = cleanup_info["holes_filled"]

                # Record quality
                quality_entry = {
                    "frame_idx": frame_idx,
                    "object_id": object_id,
                    **quality,
                }
                quality_writer.write(quality_entry)

                # Track stats
                stats["node_counts"].append(quality["skeleton_node_count"])
                stats["edge_counts"].append(quality["skeleton_edge_count"])

                # Build record
                record = graph_to_record(
                    video_id=video_id,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp,
                    object_id=object_id,
                    graph=graph,
                    bbox=bbox,
                )
                # Enrich with quality and mask diagnostics
                record["quality"] = quality
                record["accepted"] = quality["accepted"]
                record["reject_reasons"] = quality["reject_reasons"]
                record["mask_backend"] = mask_source
                record["mask_confidence"] = mask_confidence
                if object_class is not None:
                    record["object_class"] = object_class
                    record["label_confidence"] = label_confidence

                if quality["accepted"]:
                    accepted_writer.write(record)
                    accepted_records.append(record)
                    stats["accepted_count"] += 1

                    if save_masks:
                        save_mask(mask, str(masks_dir), frame_idx, object_id)

                    if save_overlays:
                        overlay_img = draw_overlay(frame, mask, graph, object_id, quality)
                        save_overlay(overlay_img, str(overlays_dir), frame_idx, object_id)
                else:
                    rejected_writer.write(record)
                    rejected_records.append(record)
                    stats["rejected_count"] += 1
                    stats["reject_reasons"].extend(quality["reject_reasons"])

                    if save_overlays:
                        overlay_img = draw_overlay(frame, mask, graph, object_id, quality)
                        save_overlay(overlay_img, str(rejected_overlays_dir), frame_idx, object_id)

            # Progress
            if stats["frames_processed"] % 10 == 0:
                print(f"  [frame {frame_idx}] {stats['frames_processed']} frames, "
                      f"{stats['accepted_count']} accepted, {stats['rejected_count']} rejected")

    finally:
        accepted_writer.close()
        rejected_writer.close()
        quality_writer.close()

    # NPZ export (accepted only)
    if accepted_records:
        npz_path = export_npz(accepted_records, str(npz_dir))
        print(f"[HyperBone] NPZ export: {npz_path}")

    # Blender export (accepted only)
    if accepted_records:
        blender_path = export_blender_script(accepted_records, str(blender_dir))
        print(f"[HyperBone] Blender script: {blender_path}")

    # Manifest
    total = stats["accepted_count"] + stats["rejected_count"]
    manifest_info = {
        "video_id": video_id,
        "video_info": info,
        "params": {
            "sample_fps": sample_fps,
            "mask_backend": mask_backend,
            "min_branch_length": min_branch_length,
            "mask_cleanup_enabled": use_cleanup,
            "graph_repair_enabled": use_repair,
            "bridge_gap_px": bridge_gap_px,
            "node_merge_radius": node_merge_radius,
            "spur_prune_length": spur_prune_length,
        },
        "stats": {
            "frames_processed": stats["frames_processed"],
            "objects_found": stats["objects_found"],
            "accepted_count": stats["accepted_count"],
            "rejected_count": stats["rejected_count"],
            "acceptance_rate": round(stats["accepted_count"] / total * 100, 1) if total > 0 else 0,
        },
    }
    write_manifest(str(out), manifest_info)

    print()
    print(f"[HyperBone] Done.")
    print(f"  Frames processed: {stats['frames_processed']}")
    print(f"  Objects found:    {stats['objects_found']}")
    print(f"  Accepted:         {stats['accepted_count']}")
    print(f"  Rejected:         {stats['rejected_count']}")
    if total > 0:
        print(f"  Acceptance rate:  {stats['accepted_count']/total*100:.1f}%")

    return stats
