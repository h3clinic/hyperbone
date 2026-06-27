"""HyperBone Custom Skeleton Mapper — fully owned pipeline.

Architecture:
    DINO bbox/label (optional) → crop → custom mask → custom thinning → graph extraction

No SAM2. No Skan. No skeleton-tracing. No cv2.ximgproc.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple

from hyperbone.objects.proposals import ObjectProposal
from hyperbone.cv.crops import crop_to_bbox, resize_crop_max_side, map_crop_point_to_frame
from hyperbone.cv.custom_mask import mask_from_bbox_crop
from hyperbone.cv.custom_thinning import skeletonize_custom
from hyperbone.cv.graph import skeleton_to_graph
from hyperbone.cv.graph_repair import repair_graph


def map_proposal_to_skeleton(
    frame: np.ndarray,
    proposal: ObjectProposal,
    max_side: int = 384,
    thinning_algorithm: str = "zhang-suen",
    max_thinning_iterations: int = 100,
    min_branch_length: int = 10,
    mask_method: str = "combined",
    enable_repair: bool = True,
    bridge_gap_px: float = 8.0,
) -> Dict:
    """Full custom skeleton mapping for one object proposal.

    Args:
        frame: BGR image (full frame).
        proposal: ObjectProposal with bbox.
        max_side: Resize crop to this max side for speed.
        thinning_algorithm: "zhang-suen" or "guo-hall".
        max_thinning_iterations: Cap on thinning iterations.
        min_branch_length: Prune branches shorter than this.
        mask_method: "combined", "edges", "threshold", "grabcut_lite".
        enable_repair: Bridge disconnected components.
        bridge_gap_px: Max gap for bridging.

    Returns:
        Dict with graph, mask, skeleton, timing, and metadata.
    """
    timings = {}
    t0 = time.perf_counter()

    # Step 1: Crop to bbox
    t = time.perf_counter()
    crop_result = crop_to_bbox(frame, proposal.bbox_xywh, pad_px=8)
    crop = crop_result["crop"]
    crop_meta = {
        "offset_xy": crop_result["offset_xy"],
        "scale": 1.0,
    }
    timings["crop_ms"] = (time.perf_counter() - t) * 1000

    # Step 2: Resize for speed
    t = time.perf_counter()
    resize_result = resize_crop_max_side(crop, max_side)
    resized = resize_result["resized"]
    crop_meta["scale"] = resize_result["scale"]
    timings["resize_ms"] = (time.perf_counter() - t) * 1000

    # Step 3: Custom mask from crop
    t = time.perf_counter()
    mask_result = mask_from_bbox_crop(resized, method=mask_method)
    mask = mask_result["mask"]
    timings["mask_ms"] = (time.perf_counter() - t) * 1000

    mask_area = int(np.count_nonzero(mask))
    if mask_area == 0:
        return _empty_result(proposal, timings, t0)

    # Step 4: Custom thinning (HyperBone-owned Zhang-Suen)
    t = time.perf_counter()
    skeleton = skeletonize_custom(mask, algorithm=thinning_algorithm,
                                  max_iterations=max_thinning_iterations)
    timings["thinning_ms"] = (time.perf_counter() - t) * 1000

    skel_pixels = int(np.count_nonzero(skeleton))
    if skel_pixels == 0:
        return _empty_result(proposal, timings, t0)

    # Step 5: Graph extraction (HyperBone-owned pixel trace)
    t = time.perf_counter()
    graph = skeleton_to_graph(skeleton, min_branch_length=min_branch_length)
    timings["graph_ms"] = (time.perf_counter() - t) * 1000

    if not graph["nodes"]:
        return _empty_result(proposal, timings, t0)

    # Step 6: Graph repair (bridge disconnected components)
    if enable_repair:
        t = time.perf_counter()
        repair_result = repair_graph(
            graph,
            mask_shape=resized.shape[:2],
            bridge_gap_px=bridge_gap_px,
            min_branch_length=min_branch_length,
        )
        graph = repair_result["graph"]
        timings["repair_ms"] = (time.perf_counter() - t) * 1000
    else:
        repair_result = None

    # Step 7: Map coordinates back to frame space
    for node in graph["nodes"]:
        cx, cy = node["xy"]
        fx, fy = map_crop_point_to_frame((cx, cy), crop_meta)
        node["xy"] = [fx, fy]
        node["crop_xy"] = [cx, cy]  # preserve crop coords for debugging

    timings["total_ms"] = (time.perf_counter() - t0) * 1000

    return {
        "graph": graph,
        "mask": mask,
        "skeleton": skeleton,
        "proposal": proposal.to_dict(),
        "crop_meta": crop_meta,
        "timings": timings,
        "metadata": {
            "skeleton_mapper": "hyperbone-custom",
            "thinning_algorithm": thinning_algorithm,
            "graph_builder": "hyperbone-pixel-trace-v0",
            "mask_method": mask_method,
            "crop_scale": crop_meta["scale"],
            "mask_area_px": mask_area,
            "skeleton_pixels": skel_pixels,
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "object_label": proposal.label,
            "object_label_source": proposal.label_source,
            "object_label_confidence": proposal.label_confidence,
            "bbox_xywh": list(proposal.bbox_xywh),
            "skeleton_runtime_ms": timings["total_ms"],
        },
        "repair_info": {k: v for k, v in (repair_result or {}).items() if k != "graph"} if repair_result else None,
    }


def _empty_result(proposal: ObjectProposal, timings: Dict, t0: float) -> Dict:
    """Return empty result when mask/skeleton is empty."""
    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    return {
        "graph": {"nodes": [], "edges": []},
        "mask": None,
        "skeleton": None,
        "proposal": proposal.to_dict(),
        "crop_meta": {},
        "timings": timings,
        "metadata": {
            "skeleton_mapper": "hyperbone-custom",
            "thinning_algorithm": "zhang-suen",
            "graph_builder": "hyperbone-pixel-trace-v0",
            "node_count": 0,
            "edge_count": 0,
            "object_label": proposal.label,
            "skeleton_runtime_ms": timings["total_ms"],
        },
        "repair_info": None,
    }


def batch_map_proposals(
    frame: np.ndarray,
    proposals: List[ObjectProposal],
    **kwargs,
) -> List[Dict]:
    """Map multiple proposals on one frame."""
    results = []
    for p in proposals:
        result = map_proposal_to_skeleton(frame, p, **kwargs)
        results.append(result)
    return results
