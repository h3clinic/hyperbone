"""Manual proposal loader — load ObjectProposal from JSONL files.

Enables testing the custom mapper without requiring DINO/GroundingDINO installed.
"""

import json
from pathlib import Path
from typing import List, Optional

from hyperbone.objects.proposals import ObjectProposal


def load_proposals_jsonl(path: str) -> List[ObjectProposal]:
    """Load proposals from a JSONL file.

    Each line must be a JSON object with:
        frame_idx: int
        object_id: int (optional, auto-assigned if missing)
        label: str
        label_confidence: float (optional, default 1.0)
        bbox_xywh: [x, y, w, h]
        prompt: str (optional)

    Args:
        path: Path to the JSONL file.

    Returns:
        List of ObjectProposal.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any record has invalid format.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Proposals file not found: {path}")

    proposals = []
    auto_id = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} in {path}: {e}"
                )

            # Validate required fields
            if "frame_idx" not in rec:
                raise ValueError(
                    f"Missing 'frame_idx' on line {line_num} in {path}"
                )
            if "bbox_xywh" not in rec:
                raise ValueError(
                    f"Missing 'bbox_xywh' on line {line_num} in {path}"
                )

            bbox = rec["bbox_xywh"]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(
                    f"Invalid 'bbox_xywh' on line {line_num} in {path}: must be [x, y, w, h]"
                )

            # Validate bbox values
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                raise ValueError(
                    f"Invalid bbox dimensions on line {line_num} in {path}: "
                    f"w={w}, h={h} (must be positive)"
                )

            object_id = rec.get("object_id", auto_id)
            auto_id += 1

            proposals.append(ObjectProposal(
                frame_idx=int(rec["frame_idx"]),
                object_id=int(object_id),
                label=rec.get("label", "unknown"),
                label_source="manual",
                label_confidence=float(rec.get("label_confidence", 1.0)),
                bbox_xywh=(int(x), int(y), int(w), int(h)),
                prompt=rec.get("prompt", ""),
                metadata=rec.get("metadata", {}),
            ))

    return proposals


def clip_proposal_to_frame(
    proposal: ObjectProposal,
    frame_width: int,
    frame_height: int,
) -> Optional[ObjectProposal]:
    """Clip a proposal's bbox to frame boundaries.

    Returns None if the clipped bbox is too small (< 4px either side).
    """
    x, y, w, h = proposal.bbox_xywh

    # Clip to frame
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + w)
    y2 = min(frame_height, y + h)

    cw = x2 - x1
    ch = y2 - y1

    if cw < 4 or ch < 4:
        return None

    return ObjectProposal(
        frame_idx=proposal.frame_idx,
        object_id=proposal.object_id,
        label=proposal.label,
        label_source=proposal.label_source,
        label_confidence=proposal.label_confidence,
        bbox_xywh=(x1, y1, cw, ch),
        prompt=proposal.prompt,
        metadata=proposal.metadata,
    )
