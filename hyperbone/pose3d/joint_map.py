"""
Canonical quadruped joint map.

Maps asset-specific armature bone names to a fixed 19-joint quadruped schema.
If a source rig lacks a joint, it is marked invisible with xyz=(0,0,0).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Canonical quadruped joint order (19 joints)
QUADRUPED_JOINTS = [
    "root",
    "spine_1",
    "spine_2",
    "neck",
    "head",
    "tail_1",
    "tail_2",
    "front_left_shoulder",
    "front_left_elbow",
    "front_left_paw",
    "front_right_shoulder",
    "front_right_elbow",
    "front_right_paw",
    "rear_left_hip",
    "rear_left_knee",
    "rear_left_paw",
    "rear_right_hip",
    "rear_right_knee",
    "rear_right_paw",
]

NUM_JOINTS = len(QUADRUPED_JOINTS)

# Canonical bone connections (parent_idx, child_idx)
QUADRUPED_BONES = [
    (0, 1),   # root -> spine_1
    (1, 2),   # spine_1 -> spine_2
    (2, 3),   # spine_2 -> neck
    (3, 4),   # neck -> head
    (0, 5),   # root -> tail_1
    (5, 6),   # tail_1 -> tail_2
    (2, 7),   # spine_2 -> front_left_shoulder
    (7, 8),   # front_left_shoulder -> front_left_elbow
    (8, 9),   # front_left_elbow -> front_left_paw
    (2, 10),  # spine_2 -> front_right_shoulder
    (10, 11), # front_right_shoulder -> front_right_elbow
    (11, 12), # front_right_elbow -> front_right_paw
    (0, 13),  # root -> rear_left_hip
    (13, 14), # rear_left_hip -> rear_left_knee
    (14, 15), # rear_left_knee -> rear_left_paw
    (0, 16),  # root -> rear_right_hip
    (16, 17), # rear_right_hip -> rear_right_knee
    (17, 18), # rear_right_knee -> rear_right_paw
]


# ─── Asset-specific name mappings ────────────────────────────────────────────

# Fox.glb (KhronosGroup sample) bone names → canonical
FOX_JOINT_MAP: Dict[str, str] = {
    "b_Root_00": "root",
    "b_Spine01_03": "spine_1",
    "b_Spine02_04": "spine_2",
    "b_Neck_04": "neck",
    "b_Head_05": "head",
    "b_Tail01_012": "tail_1",
    "b_Tail02_013": "tail_2",
    "b_LeftUpperArm_06": "front_left_shoulder",
    "b_LeftForeArm_07": "front_left_elbow",
    "b_LeftHand_08": "front_left_paw",
    "b_RightUpperArm_06": "front_right_shoulder",
    "b_RightForeArm_07": "front_right_elbow",
    "b_RightHand_08": "front_right_paw",
    "b_LeftLeg_09": "rear_left_hip",
    "b_LeftLeg01_010": "rear_left_knee",
    "b_LeftFoot_011": "rear_left_paw",
    "b_RightLeg_09": "rear_right_hip",
    "b_RightLeg01_010": "rear_right_knee",
    "b_RightFoot_011": "rear_right_paw",
}

# Generic mapping patterns (tried in order)
GENERIC_PATTERNS: Dict[str, List[str]] = {
    "root": ["root", "pelvis", "hips"],
    "spine_1": ["spine", "spine1", "spine01"],
    "spine_2": ["spine2", "spine02", "chest"],
    "neck": ["neck"],
    "head": ["head"],
    "tail_1": ["tail", "tail1", "tail01"],
    "tail_2": ["tail2", "tail02", "tail_end"],
    "front_left_shoulder": ["leftshoulder", "leftupperarm", "l_shoulder", "front_l_upper"],
    "front_left_elbow": ["leftforearm", "leftelbow", "l_elbow", "front_l_lower"],
    "front_left_paw": ["lefthand", "leftpaw", "l_hand", "l_paw", "front_l_foot"],
    "front_right_shoulder": ["rightshoulder", "rightupperarm", "r_shoulder", "front_r_upper"],
    "front_right_elbow": ["rightforearm", "rightelbow", "r_elbow", "front_r_lower"],
    "front_right_paw": ["righthand", "rightpaw", "r_hand", "r_paw", "front_r_foot"],
    "rear_left_hip": ["leftleg", "lefthip", "leftthigh", "l_hip", "rear_l_upper"],
    "rear_left_knee": ["leftleg01", "leftknee", "leftshin", "l_knee", "rear_l_lower"],
    "rear_left_paw": ["leftfoot", "leftankle", "l_foot", "rear_l_foot"],
    "rear_right_hip": ["rightleg", "righthip", "rightthigh", "r_hip", "rear_r_upper"],
    "rear_right_knee": ["rightleg01", "rightknee", "rightshin", "r_knee", "rear_r_lower"],
    "rear_right_paw": ["rightfoot", "rightankle", "r_foot", "rear_r_foot"],
}


def get_asset_map(asset_id: str) -> Dict[str, str]:
    """Return the name→canonical joint map for a known asset."""
    asset_lower = asset_id.lower()
    if "fox" in asset_lower:
        return FOX_JOINT_MAP
    return {}


def auto_map_joints(source_names: List[str], asset_id: str = "") -> Dict[str, str]:
    """
    Automatically map source joint names to canonical quadruped joints.

    Returns: {source_name: canonical_name}
    """
    # Try asset-specific map first
    known = get_asset_map(asset_id)
    if known:
        return known

    # Generic pattern matching
    mapping = {}
    used_canonical = set()

    for canonical_joint, patterns in GENERIC_PATTERNS.items():
        if canonical_joint in used_canonical:
            continue
        for src_name in source_names:
            src_lower = src_name.lower().replace("_", "").replace("-", "").replace(" ", "")
            for pattern in patterns:
                pat_clean = pattern.replace("_", "").replace("-", "")
                if pat_clean in src_lower and src_name not in mapping:
                    mapping[src_name] = canonical_joint
                    used_canonical.add(canonical_joint)
                    break
            if canonical_joint in used_canonical:
                break

    return mapping


def remap_joints_to_canonical(
    source_joints: List[Dict],
    asset_id: str = "",
) -> Tuple[List[Tuple[float, float, float]], List[bool]]:
    """
    Remap source joint list to canonical 19-joint format.

    Returns:
        xyz_canonical: list of 19 (x,y,z) tuples
        visible: list of 19 booleans
    """
    source_names = [j["name"] for j in source_joints]
    mapping = auto_map_joints(source_names, asset_id)

    # Build source name → joint data lookup
    src_by_name = {j["name"]: j for j in source_joints}

    xyz = [(0.0, 0.0, 0.0)] * NUM_JOINTS
    visible = [False] * NUM_JOINTS

    # Reverse mapping: canonical_name → source_name
    canonical_to_src = {}
    for src_name, can_name in mapping.items():
        canonical_to_src[can_name] = src_name

    for i, can_name in enumerate(QUADRUPED_JOINTS):
        if can_name in canonical_to_src:
            src_name = canonical_to_src[can_name]
            if src_name in src_by_name:
                j = src_by_name[src_name]
                # Use world_xyz if available, else image_xy with z=0
                world = j.get("world_xyz")
                if world:
                    xyz[i] = tuple(world)
                visible[i] = j.get("visible", True)

    return xyz, visible
