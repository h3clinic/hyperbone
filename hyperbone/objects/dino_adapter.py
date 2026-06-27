"""DINO/GroundingDINO adapter — object proposals only.

Returns ObjectProposal objects with label, bbox, and confidence.
Does NOT return masks. Does NOT call SAM2. Does NOT skeletonize.

HyperBone uses this only for open-vocabulary candidate identification.
"""

import numpy as np
from typing import List, Optional

from hyperbone.objects.proposals import ObjectProposal


class DINOProposalAdapter:
    """Generate ObjectProposal objects from GroundingDINO detections.

    Lazy-loads the model. Fails with clear error only when generate() is called
    if dependencies are missing.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        max_proposals_per_frame: int = 20,
    ):
        self.model_id = model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_proposals_per_frame = max_proposals_per_frame

        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy-load GroundingDINO from HuggingFace."""
        if self._model is not None:
            return

        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError as e:
            raise RuntimeError(
                "GroundingDINO requires `transformers` package. "
                "Install with: pip install transformers"
            ) from e

        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "GroundingDINO requires PyTorch. "
                "Install with: pip install torch"
            ) from e

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        self._model.eval()

    def generate(
        self,
        frame: np.ndarray,
        frame_idx: int,
        text_prompt: str,
    ) -> List[ObjectProposal]:
        """Detect objects and return proposals (label + bbox only).

        Args:
            frame: BGR image (H, W, 3).
            frame_idx: Index of the frame in the video.
            text_prompt: Grounding DINO text prompt (e.g. "person. hand. arm.").

        Returns:
            List of ObjectProposal (no masks, no skeletons).
        """
        import torch
        from PIL import Image

        self._load_model()

        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1].copy()
        pil_image = Image.fromarray(rgb)

        inputs = self._processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"].cpu().numpy()  # (N, 4) x1,y1,x2,y2
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        proposals = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if i >= self.max_proposals_per_frame:
                break

            x1, y1, x2, y2 = box
            bx = max(0, int(x1))
            by = max(0, int(y1))
            bw = min(w, int(x2)) - bx
            bh = min(h, int(y2)) - by

            if bw < 4 or bh < 4:
                continue

            proposals.append(ObjectProposal(
                frame_idx=frame_idx,
                object_id=i,
                label=label.strip().lower(),
                label_source="groundingdino",
                label_confidence=float(score),
                bbox_xywh=(bx, by, bw, bh),
                prompt=text_prompt,
                metadata={"model_id": self.model_id},
            ))

        return proposals


    # Backward-compatible alias for older call sites.
    DINOAdapter = DINOProposalAdapter
