"""Object proposals — structured detection results for skeleton mapping.

ObjectProposal carries a detected object's label, bbox, and metadata
through the HyperBone pipeline. Decoupled from any specific detector.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


@dataclass
class ObjectProposal:
    """A detected object proposal for skeleton extraction."""
    frame_idx: int
    object_id: int
    label: str = "unknown"
    label_source: str = "unknown"  # "dino" | "groundingdino" | "manual" | "unknown"
    label_confidence: float = 0.0
    bbox_xywh: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    prompt: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def bbox_xyxy(self) -> Tuple[int, int, int, int]:
        """Convert xywh to x1,y1,x2,y2."""
        x, y, w, h = self.bbox_xywh
        return (x, y, x + w, y + h)

    @classmethod
    def from_mask_record(cls, mask_rec: Dict, frame_idx: int) -> "ObjectProposal":
        """Create proposal from a mask backend record."""
        bbox = mask_rec.get("bbox", [0, 0, 0, 0])
        return cls(
            frame_idx=frame_idx,
            object_id=mask_rec.get("object_id", 0),
            label=mask_rec.get("object_class", "unknown"),
            label_source=mask_rec.get("source", "unknown"),
            label_confidence=mask_rec.get("label_confidence", mask_rec.get("confidence", 0.0)),
            bbox_xywh=tuple(bbox),
            prompt=mask_rec.get("metadata", {}).get("phrase", ""),
            metadata=mask_rec.get("metadata", {}),
        )

    @classmethod
    def manual(cls, frame_idx: int, object_id: int, bbox_xywh: Tuple[int, int, int, int],
               label: str = "unknown") -> "ObjectProposal":
        """Create a manual proposal (for testing without DINO)."""
        return cls(
            frame_idx=frame_idx,
            object_id=object_id,
            label=label,
            label_source="manual",
            label_confidence=1.0,
            bbox_xywh=bbox_xywh,
        )

    def to_dict(self) -> Dict:
        return {
            "frame_idx": self.frame_idx,
            "object_id": self.object_id,
            "label": self.label,
            "label_source": self.label_source,
            "label_confidence": self.label_confidence,
            "bbox_xywh": list(self.bbox_xywh),
            "prompt": self.prompt,
        }
