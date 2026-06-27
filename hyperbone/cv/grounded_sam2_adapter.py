"""Grounded SAM2 adapter — open-vocabulary detection + SAM2 segmentation.

Uses Grounding DINO (HuggingFace) for text-prompted object detection,
then SAM2 for precise mask generation from the detected boxes.

Each mask gets an object_class label from the grounding model.
"""

import numpy as np
import torch
from typing import Dict, List, Optional


class GroundedSAM2Backend:
    """Mask backend that uses Grounding DINO + SAM2 for labeled segmentation."""

    def __init__(
        self,
        checkpoint: str = "checkpoints/sam2.1_hiera_tiny.pt",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        device: str = "cuda",
        grounding_model: str = "IDEA-Research/grounding-dino-tiny",
        text_prompt: str = "person. hand. arm. leg. head. torso. body. face. foot. knee. elbow. shoulder.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        max_masks_per_frame: int = 10,
        min_area_ratio: float = 0.002,
        max_area_ratio: float = 0.80,
    ):
        self.device = device
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_masks_per_frame = max_masks_per_frame
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

        # Lazy load models
        self._gdino_model = None
        self._gdino_processor = None
        self._sam2_predictor = None
        self._grounding_model_id = grounding_model
        self._sam2_checkpoint = checkpoint
        self._sam2_model_cfg = model_cfg

    def _load_grounding_dino(self):
        """Load Grounding DINO from HuggingFace."""
        if self._gdino_model is not None:
            return

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self._gdino_processor = AutoProcessor.from_pretrained(self._grounding_model_id)
        self._gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._grounding_model_id
        ).to(self.device)
        self._gdino_model.eval()

    def _load_sam2(self):
        """Load SAM2 image predictor for box-prompted segmentation."""
        if self._sam2_predictor is not None:
            return

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(
            self._sam2_model_cfg,
            self._sam2_checkpoint,
            device=self.device,
        )
        self._sam2_predictor = SAM2ImagePredictor(sam2_model)

    def generate(self, frame: np.ndarray) -> List[Dict]:
        """Generate labeled masks from a BGR frame.

        Returns list of dicts with keys:
            mask, bbox, area, confidence, source, object_id,
            touches_edge, metadata, object_class, label_confidence
        """
        self._load_grounding_dino()
        self._load_sam2()

        h, w = frame.shape[:2]
        frame_area = h * w

        # Convert BGR to RGB for models
        rgb_frame = frame[:, :, ::-1].copy()

        # Step 1: Grounding DINO detection
        from PIL import Image
        pil_image = Image.fromarray(rgb_frame)

        inputs = self._gdino_processor(
            images=pil_image,
            text=self.text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._gdino_model(**inputs)

        results = self._gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"].cpu().numpy()  # [N, 4] xyxy
        labels = results["labels"]  # list of strings
        scores = results["scores"].cpu().numpy()  # [N]

        if len(boxes) == 0:
            return []

        # Sort by confidence, limit count
        order = np.argsort(-scores)[:self.max_masks_per_frame]
        boxes = boxes[order]
        labels = [labels[i] for i in order]
        scores = scores[order]

        # Step 2: SAM2 mask prediction from boxes
        self._sam2_predictor.set_image(rgb_frame)

        records = []
        for obj_id, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / frame_area

            # Filter by area
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            # SAM2 predict from box
            input_box = np.array([[x1, y1, x2, y2]])
            masks, iou_scores, _ = self._sam2_predictor.predict(
                box=input_box,
                multimask_output=False,
            )

            mask = masks[0].astype(np.uint8) * 255  # (H, W), uint8
            mask_area = int(mask.sum() // 255)
            mask_area_ratio = mask_area / frame_area

            if mask_area_ratio < self.min_area_ratio:
                continue

            # Bbox in xywh format
            bx = int(x1)
            by = int(y1)
            bw = int(x2 - x1)
            bh = int(y2 - y1)

            # Edge detection
            touches_edge = _mask_touches_edge(mask, h, w)

            records.append({
                "mask": mask,
                "bbox": [bx, by, bw, bh],
                "area": mask_area,
                "confidence": float(iou_scores[0]),
                "source": "grounded_sam2",
                "object_id": obj_id,
                "touches_edge": touches_edge,
                "metadata": {
                    "iou_score": float(iou_scores[0]),
                    "grounding_score": float(score),
                },
                "object_class": label.strip().rstrip("."),
                "label_confidence": float(score),
            })

        return records


def _mask_touches_edge(mask: np.ndarray, h: int, w: int, margin: int = 2) -> bool:
    """Check if mask touches frame edges."""
    if mask[:margin, :].any():
        return True
    if mask[-margin:, :].any():
        return True
    if mask[:, :margin].any():
        return True
    if mask[:, -margin:].any():
        return True
    return False
