"""
Canonical quadruped joint schema for 2D pose estimation.

19 anatomical joints with parent-child hierarchy.
Species-agnostic: works for fox, wolf, horse, pig, deer, etc.
Endpoint naming uses 'hoof' canonically; aliases available for paw/foot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class JointDef:
    """Definition of a canonical joint."""
    id: int
    name: str
    parent_id: Optional[int]
    side: str  # "left", "right", "midline"


# Canonical 19-joint quadruped schema
QUADRUPED_JOINTS_2D = [
    "root",                    # 0 - pelvis/hip center
    "spine_1",                 # 1 - mid-back
    "spine_2",                 # 2 - upper back
    "neck",                    # 3 - base of neck
    "head",                    # 4 - top of head
    "tail_base",               # 5 - base of tail
    "tail_tip",                # 6 - end of tail
    "front_left_shoulder",     # 7
    "front_left_elbow",        # 8
    "front_left_hoof",         # 9
    "front_right_shoulder",    # 10
    "front_right_elbow",       # 11
    "front_right_hoof",        # 12
    "rear_left_hip",           # 13
    "rear_left_knee",          # 14
    "rear_left_hoof",          # 15
    "rear_right_hip",          # 16
    "rear_right_knee",         # 17
    "rear_right_hoof",         # 18
]

NUM_JOINTS_2D = len(QUADRUPED_JOINTS_2D)  # 19

# Joint index lookup
JOINT_ID = {name: i for i, name in enumerate(QUADRUPED_JOINTS_2D)}

# Parent-child hierarchy (parent_id for each joint)
JOINT_PARENTS = [
    None,  # root (no parent)
    0,     # spine_1 -> root
    1,     # spine_2 -> spine_1
    2,     # neck -> spine_2
    3,     # head -> neck
    0,     # tail_base -> root
    5,     # tail_tip -> tail_base
    2,     # front_left_shoulder -> spine_2
    7,     # front_left_elbow -> front_left_shoulder
    8,     # front_left_hoof -> front_left_elbow
    2,     # front_right_shoulder -> spine_2
    10,    # front_right_elbow -> front_right_shoulder
    11,    # front_right_hoof -> front_right_elbow
    0,     # rear_left_hip -> root
    13,    # rear_left_knee -> rear_left_hip
    14,    # rear_left_hoof -> rear_left_knee
    0,     # rear_right_hip -> root
    16,    # rear_right_knee -> rear_right_hip
    17,    # rear_right_hoof -> rear_right_knee
]

# Bone connections (parent_idx, child_idx)
QUADRUPED_BONES_2D = [
    (0, 1),    # root -> spine_1
    (1, 2),    # spine_1 -> spine_2
    (2, 3),    # spine_2 -> neck
    (3, 4),    # neck -> head
    (0, 5),    # root -> tail_base
    (5, 6),    # tail_base -> tail_tip
    (2, 7),    # spine_2 -> front_left_shoulder
    (7, 8),    # front_left_shoulder -> front_left_elbow
    (8, 9),    # front_left_elbow -> front_left_hoof
    (2, 10),   # spine_2 -> front_right_shoulder
    (10, 11),  # front_right_shoulder -> front_right_elbow
    (11, 12),  # front_right_elbow -> front_right_hoof
    (0, 13),   # root -> rear_left_hip
    (13, 14),  # rear_left_hip -> rear_left_knee
    (14, 15),  # rear_left_knee -> rear_left_hoof
    (0, 16),   # root -> rear_right_hip
    (16, 17),  # rear_right_hip -> rear_right_knee
    (17, 18),  # rear_right_knee -> rear_right_hoof
]

# Side classification
JOINT_SIDES = {
    "root": "midline",
    "spine_1": "midline",
    "spine_2": "midline",
    "neck": "midline",
    "head": "midline",
    "tail_base": "midline",
    "tail_tip": "midline",
    "front_left_shoulder": "left",
    "front_left_elbow": "left",
    "front_left_hoof": "left",
    "front_right_shoulder": "right",
    "front_right_elbow": "right",
    "front_right_hoof": "right",
    "rear_left_hip": "left",
    "rear_left_knee": "left",
    "rear_left_hoof": "left",
    "rear_right_hip": "right",
    "rear_right_knee": "right",
    "rear_right_hoof": "right",
}

# Species endpoint aliases (map to canonical 'hoof')
ENDPOINT_ALIASES = {
    "paw": "hoof",
    "foot": "hoof",
    "claw": "hoof",
}

# Mapping from old pose3d schema names to pose2d schema names
POSE3D_TO_POSE2D = {
    "root": "root",
    "spine_1": "spine_1",
    "spine_2": "spine_2",
    "neck": "neck",
    "head": "head",
    "tail_1": "tail_base",
    "tail_2": "tail_tip",
    "front_left_shoulder": "front_left_shoulder",
    "front_left_elbow": "front_left_elbow",
    "front_left_paw": "front_left_hoof",
    "front_right_shoulder": "front_right_shoulder",
    "front_right_elbow": "front_right_elbow",
    "front_right_paw": "front_right_hoof",
    "rear_left_hip": "rear_left_hip",
    "rear_left_knee": "rear_left_knee",
    "rear_left_paw": "rear_left_hoof",
    "rear_right_hip": "rear_right_hip",
    "rear_right_knee": "rear_right_knee",
    "rear_right_paw": "rear_right_hoof",
}

# Reverse mapping
POSE2D_TO_POSE3D = {v: k for k, v in POSE3D_TO_POSE2D.items()}


def get_joint_defs() -> List[JointDef]:
    """Get full joint definitions."""
    return [
        JointDef(
            id=i,
            name=name,
            parent_id=JOINT_PARENTS[i],
            side=JOINT_SIDES[name],
        )
        for i, name in enumerate(QUADRUPED_JOINTS_2D)
    ]


def validate_pose(
    joints_xy: Dict[str, Tuple[float, float]],
    confidences: Dict[str, float],
    conf_threshold: float = 0.3,
    min_joints: int = 5,
) -> Tuple[bool, str]:
    """
    Validate a predicted pose.

    Returns (is_valid, reason).
    Rejects if fewer than min_joints are above confidence threshold.
    """
    confident = [
        name for name, conf in confidences.items()
        if conf >= conf_threshold
    ]
    if len(confident) < min_joints:
        return False, f"only {len(confident)} confident joints (need {min_joints})"
    return True, "ok"


def joints_inside_bbox_ratio(
    joints_xy: Dict[str, Tuple[float, float]],
    bbox_xywh: Tuple[float, float, float, float],
    confidences: Optional[Dict[str, float]] = None,
    conf_threshold: float = 0.3,
) -> float:
    """
    Compute fraction of confident joints that fall inside the bbox.
    A healthy pose should have most joints inside the detection box.
    """
    x, y, w, h = bbox_xywh
    inside = 0
    total = 0
    for name, (jx, jy) in joints_xy.items():
        if confidences and confidences.get(name, 0) < conf_threshold:
            continue
        total += 1
        if x <= jx <= x + w and y <= jy <= y + h:
            inside += 1
    return inside / max(total, 1)
