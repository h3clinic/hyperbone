"""SAM2 mask generation adapter — lazy import, no hard dependency.

Requires:
    pip install sam2  (or clone from https://github.com/facebookresearch/sam2)
    Download checkpoint: e.g. sam2.1_hiera_tiny.pt

If SAM2 is not installed, importing this module succeeds but instantiating
SAM2MaskBackend raises RuntimeError with install instructions.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from hyperbone.cv.masks import MaskBackend, _mask_touches_edge


_SAM2_AVAILABLE = None
_SAM2_IMPORT_ERROR = None


def _check_sam2():
    """Lazy check for SAM2 availability."""
    global _SAM2_AVAILABLE, _SAM2_IMPORT_ERROR
    if _SAM2_AVAILABLE is not None:
        return _SAM2_AVAILABLE

    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        _SAM2_AVAILABLE = True
    except ImportError as e:
        _SAM2_AVAILABLE = False
        _SAM2_IMPORT_ERROR = str(e)

    return _SAM2_AVAILABLE


def is_sam2_available() -> bool:
    """Check if SAM2 is importable without raising."""
    return _check_sam2()


class SAM2MaskBackend(MaskBackend):
    """SAM2-based automatic mask generation.

    Uses SAM2AutomaticMaskGenerator for per-frame instance segmentation.
    Requires CUDA for practical use; CPU works but is extremely slow.
    """

    def __init__(
        self,
        checkpoint: str = "checkpoints/sam2.1_hiera_tiny.pt",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        device: str = "cuda",
        max_masks_per_frame: int = 10,
        min_area_ratio: float = 0.002,
        max_area_ratio: float = 0.80,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
        **kwargs,
    ):
        if not _check_sam2():
            raise RuntimeError(
                f"SAM2 is not installed. Cannot use backend='sam2'.\n"
                f"Import error: {_SAM2_IMPORT_ERROR}\n\n"
                f"To install SAM2:\n"
                f"  pip install sam2\n"
                f"  # or clone: git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .\n\n"
                f"Then download a checkpoint:\n"
                f"  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -P checkpoints/\n\n"
                f"Usage:\n"
                f"  --mask-backend sam2 --sam2-checkpoint checkpoints/sam2.1_hiera_tiny.pt "
                f"--sam2-model-cfg configs/sam2.1/sam2.1_hiera_t.yaml --device cuda"
            )

        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        self.device = device
        self.max_masks_per_frame = max_masks_per_frame
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

        # Build SAM2 model
        sam2_model = build_sam2(
            model_cfg,
            checkpoint,
            device=device,
        )

        # Create automatic mask generator
        self._generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )

    def generate(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Generate masks using SAM2 automatic mask generator.

        Args:
            frame: BGR uint8 image (H, W, 3).

        Returns:
            List of mask records sorted by area (largest first), filtered and capped.
        """
        import cv2

        h, w = frame.shape[:2]
        frame_area = h * w

        # SAM2 expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run SAM2 automatic mask generation
        sam2_masks = self._generator.generate(rgb)

        # Filter and convert to our record format
        records = []
        for ann in sam2_masks:
            area = int(ann["area"])
            area_ratio = area / frame_area

            if area_ratio < self.min_area_ratio:
                continue
            if area_ratio > self.max_area_ratio:
                continue

            # Convert bool mask to uint8
            mask = ann["segmentation"].astype(np.uint8) * 255

            bbox = list(ann["bbox"])  # [x, y, w, h] in XYWH format
            confidence = float(ann.get("predicted_iou", ann.get("stability_score", 0.9)))
            touches_edge = _mask_touches_edge(mask, h, w)

            records.append({
                "mask": mask,
                "bbox": bbox,
                "area": area,
                "confidence": confidence,
                "source": "sam2",
                "object_id": -1,  # assigned below
                "touches_edge": touches_edge,
                "metadata": {
                    "stability_score": float(ann.get("stability_score", 0.0)),
                    "predicted_iou": float(ann.get("predicted_iou", 0.0)),
                    "crop_box": ann.get("crop_box", []),
                },
            })

        # Sort by area descending (largest objects first)
        records.sort(key=lambda r: r["area"], reverse=True)

        # Cap to max masks
        records = records[:self.max_masks_per_frame]

        # Assign sequential object IDs
        for i, rec in enumerate(records):
            rec["object_id"] = i

        return records
