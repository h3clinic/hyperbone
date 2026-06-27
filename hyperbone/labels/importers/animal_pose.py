"""
Import labels from real animal pose datasets.

Supports:
- AP-10K style COCO keypoints (JSON)
- Animal3D style 3D keypoints
- Manual HyperBone JSONL labels

Maps imported labels to the HyperBone node schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from hyperbone.labels.schema import (
    GraphLabel,
    HyperNodeLabel,
    HyperEdgeLabel,
    NodeType,
    EdgeType,
    LabelSource,
)


# AP-10K keypoint names (17 keypoints per animal)
AP10K_KEYPOINTS = [
    "left_eye", "right_eye", "nose",
    "neck", "root_of_tail",
    "left_shoulder", "left_elbow", "left_front_paw",
    "right_shoulder", "right_elbow", "right_front_paw",
    "left_hip", "left_knee", "left_back_paw",
    "right_hip", "right_knee", "right_back_paw",
]

# Skeleton edges for AP-10K
AP10K_SKELETON = [
    (0, 2), (1, 2), (2, 3),  # eyes → nose → neck
    (3, 5), (5, 6), (6, 7),  # neck → left front leg
    (3, 8), (8, 9), (9, 10),  # neck → right front leg
    (3, 4),                    # neck → tail root
    (4, 11), (11, 12), (12, 13),  # tail → left back leg
    (4, 14), (14, 15), (15, 16),  # tail → right back leg
]

# Map AP-10K names to HyperBone semantic names
AP10K_TO_HYPERBONE = {
    "left_eye": "left_eye",
    "right_eye": "right_eye",
    "nose": "nose",
    "neck": "neck",
    "root_of_tail": "tail_base",
    "left_shoulder": "front_left_shoulder",
    "left_elbow": "front_left_elbow",
    "left_front_paw": "front_left_hoof",
    "right_shoulder": "front_right_shoulder",
    "right_elbow": "front_right_elbow",
    "right_front_paw": "front_right_hoof",
    "left_hip": "rear_left_hip",
    "left_knee": "rear_left_knee",
    "left_back_paw": "rear_left_hoof",
    "right_hip": "rear_right_hip",
    "right_knee": "rear_right_knee",
    "right_back_paw": "rear_right_hoof",
}


def import_ap10k_coco(
    annotation_path: Path,
    image_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> list[GraphLabel]:
    """
    Import AP-10K style COCO keypoint annotations.

    Expected format: COCO keypoints JSON with 'annotations' and 'images'.
    Each annotation has 'keypoints' as [x1, y1, v1, x2, y2, v2, ...].
    """
    with open(annotation_path) as f:
        data = json.load(f)

    # Build image lookup
    images = {img["id"]: img for img in data.get("images", [])}

    labels: list[GraphLabel] = []

    annotations = data.get("annotations", [])
    if max_samples:
        annotations = annotations[:max_samples]

    for ann in annotations:
        img_id = ann["image_id"]
        img_info = images.get(img_id, {})
        img_filename = img_info.get("file_name", f"{img_id}.jpg")

        keypoints = ann.get("keypoints", [])
        if len(keypoints) < 51:  # 17 * 3
            continue

        nodes = []
        edges = []

        for ki in range(17):
            x = keypoints[ki * 3]
            y = keypoints[ki * 3 + 1]
            v = keypoints[ki * 3 + 2]  # 0=not labeled, 1=labeled not visible, 2=labeled visible

            if v == 0:
                continue

            kp_name = AP10K_KEYPOINTS[ki]
            semantic = AP10K_TO_HYPERBONE.get(kp_name, kp_name)

            # Determine node type from position in skeleton
            ntype = NodeType.SEMANTIC_JOINT
            if "eye" in kp_name or "nose" in kp_name or "paw" in kp_name:
                ntype = NodeType.ENDPOINT

            conf = 0.8 if v == 2 else 0.5  # visible vs occluded

            nodes.append(HyperNodeLabel(
                id=ki,
                node_type=ntype,
                xy=[x, y],
                semantic=semantic,
                confidence=conf,
                label_sources={LabelSource.ANIMAL_DATASET.value: conf},
                accepted=True,
            ))

        # Add skeleton edges
        edge_id = 0
        node_ids_present = {n.id for n in nodes}
        for src, tgt in AP10K_SKELETON:
            if src in node_ids_present and tgt in node_ids_present:
                edges.append(HyperEdgeLabel(
                    id=edge_id,
                    source_node_id=src,
                    target_node_id=tgt,
                    edge_type=EdgeType.BONE,
                    confidence=0.8,
                    label_sources={LabelSource.ANIMAL_DATASET.value: 0.8},
                ))
                edge_id += 1

        image_path = str(Path(image_dir) / img_filename) if image_dir else img_filename

        labels.append(GraphLabel(
            sample_id=f"ap10k_{ann.get('id', img_id)}",
            image_path=image_path,
            nodes=nodes,
            edges=edges,
            metadata={
                "source": "ap10k",
                "category_id": ann.get("category_id"),
                "bbox": ann.get("bbox"),
                "image_id": img_id,
            },
        ))

    return labels


def import_animal3d(
    annotation_path: Path,
    image_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> list[GraphLabel]:
    """
    Import Animal3D annotations with 3D keypoints.

    Expected: JSON with per-image annotations containing 'keypoints_3d'.
    """
    with open(annotation_path) as f:
        data = json.load(f)

    labels: list[GraphLabel] = []

    annotations = data.get("annotations", data.get("data", []))
    if max_samples:
        annotations = annotations[:max_samples]

    for ann in annotations:
        img_path = ann.get("image_path", ann.get("file_name", ""))
        keypoints_3d = ann.get("keypoints_3d", [])
        keypoints_2d = ann.get("keypoints_2d", ann.get("keypoints", []))

        nodes = []
        for ki, kp3d in enumerate(keypoints_3d):
            if len(kp3d) < 3:
                continue

            xyz = kp3d[:3]
            # Skip if all zeros (not annotated)
            if abs(xyz[0]) < 1e-8 and abs(xyz[1]) < 1e-8 and abs(xyz[2]) < 1e-8:
                continue

            xy = None
            if ki < len(keypoints_2d):
                kp2d = keypoints_2d[ki]
                if len(kp2d) >= 2:
                    xy = [kp2d[0], kp2d[1]]

            nodes.append(HyperNodeLabel(
                id=ki,
                node_type=NodeType.SEMANTIC_JOINT,
                xyz=xyz,
                xy=xy,
                confidence=0.8,
                label_sources={LabelSource.ANIMAL_DATASET.value: 0.8},
                accepted=True,
            ))

        image_path = str(Path(image_dir) / img_path) if image_dir else img_path

        labels.append(GraphLabel(
            sample_id=f"animal3d_{ann.get('id', len(labels))}",
            image_path=image_path,
            nodes=nodes,
            edges=[],
            metadata={
                "source": "animal3d",
                "species": ann.get("species", "unknown"),
            },
        ))

    return labels


def import_hyperbone_jsonl(
    jsonl_path: Path,
    max_samples: Optional[int] = None,
) -> list[GraphLabel]:
    """
    Import existing HyperBone JSONL labels (manual or previously exported).
    """
    from hyperbone.labels.schema import load_graph_labels
    labels = load_graph_labels(jsonl_path)
    if max_samples:
        labels = labels[:max_samples]
    return labels
