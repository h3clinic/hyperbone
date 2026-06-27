"""
Procedural motion synthesis for assets without animations.

Generates controlled rig motions to teach HyperBone how joints move.
These are NOT real animation clips — they are synthetic labeled data
with motion_source="synthetic_rig_motion".

Motion types:
- idle_sway: gentle oscillation around rest pose
- walk_like: alternating limb movement pattern
- run_like: faster alternating limbs with spine flex
- tail_swing: tail bones oscillate
- hinge_sweep: single-axis rotation for mechanical joints
- branch_bend: organic bending (plants, tentacles)
- generic_articulation: random controlled joint rotations

All motions are parameterized by:
- amplitude (max rotation angle in radians)
- frequency (cycles per second)
- phase offset per joint (to create wave patterns)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np


@dataclass
class MotionParams:
    """Parameters for procedural motion generation."""
    motion_type: str = "idle_sway"
    duration_sec: float = 3.0
    fps: int = 24
    amplitude_deg: float = 15.0
    frequency_hz: float = 0.5
    phase_spread: float = 0.3  # Phase offset between parent-child joints
    seed: int = 42

    @property
    def frame_count(self) -> int:
        return int(self.duration_sec * self.fps)

    @property
    def amplitude_rad(self) -> float:
        return math.radians(self.amplitude_deg)


MOTION_PRESETS: Dict[str, dict] = {
    "idle_sway": {"amplitude_deg": 8.0, "frequency_hz": 0.4, "phase_spread": 0.2},
    "walk_like": {"amplitude_deg": 25.0, "frequency_hz": 1.0, "phase_spread": 0.5},
    "run_like": {"amplitude_deg": 35.0, "frequency_hz": 1.8, "phase_spread": 0.4},
    "tail_swing": {"amplitude_deg": 30.0, "frequency_hz": 0.8, "phase_spread": 0.6},
    "hinge_sweep": {"amplitude_deg": 45.0, "frequency_hz": 0.3, "phase_spread": 0.0},
    "branch_bend": {"amplitude_deg": 20.0, "frequency_hz": 0.25, "phase_spread": 0.8},
    "generic_articulation": {"amplitude_deg": 20.0, "frequency_hz": 0.6, "phase_spread": 0.4},
}


def generate_motion_curves(
    joint_count: int,
    parent_ids: List[Optional[int]],
    joint_names: List[str],
    params: MotionParams,
) -> np.ndarray:
    """
    Generate per-joint rotation curves for a procedural motion.
    
    Args:
        joint_count: number of joints
        parent_ids: parent index for each joint (None for root)
        joint_names: names for heuristic role assignment
        params: motion parameters
        
    Returns:
        rotations: [T, J, 4] quaternion (wxyz) per frame per joint
            These are LOCAL rotations relative to rest pose.
    """
    rng = np.random.default_rng(params.seed)
    T = params.frame_count
    J = joint_count
    amp = params.amplitude_rad
    freq = params.frequency_hz
    phase_spread = params.phase_spread

    # Compute depth of each joint in the hierarchy for phase offset
    depths = _compute_depths(parent_ids)
    max_depth = max(depths) if depths else 1

    # Assign rotation axes and amplitudes per joint based on motion type
    axes = np.zeros((J, 3), dtype=np.float32)
    amps = np.zeros(J, dtype=np.float32)
    phases = np.zeros(J, dtype=np.float32)

    for ji in range(J):
        name = joint_names[ji].lower() if ji < len(joint_names) else ""
        depth = depths[ji]

        if params.motion_type == "idle_sway":
            # All joints sway gently, mostly around Y (up) and Z (side)
            axes[ji] = rng.normal(0, 1, 3)
            axes[ji] /= np.linalg.norm(axes[ji]) + 1e-8
            amps[ji] = amp * (0.5 + 0.5 * depth / max(max_depth, 1))
            phases[ji] = depth * phase_spread

        elif params.motion_type in ("walk_like", "run_like"):
            # Limb joints rotate around X (forward flex), alternating sides
            is_left = "left" in name or "l_" in name
            is_right = "right" in name or "r_" in name
            is_limb = any(k in name for k in ["leg", "arm", "knee", "elbow", "hip", "shoulder", "thigh", "shin", "forearm"])

            if is_limb:
                axes[ji] = [1.0, 0.0, 0.0]  # Flex axis
                amps[ji] = amp
                # Opposite phase for left vs right
                phases[ji] = 0.0 if is_left else math.pi
                if "knee" in name or "elbow" in name or "shin" in name or "forearm" in name:
                    phases[ji] += math.pi * 0.3  # Slight lag for child joints
            elif "spine" in name or "back" in name:
                axes[ji] = [0.0, 1.0, 0.0]
                amps[ji] = amp * 0.3
                phases[ji] = 0.0
            else:
                axes[ji] = rng.normal(0, 1, 3)
                axes[ji] /= np.linalg.norm(axes[ji]) + 1e-8
                amps[ji] = amp * 0.1
                phases[ji] = depth * phase_spread

        elif params.motion_type == "tail_swing":
            if "tail" in name:
                axes[ji] = [0.0, 0.0, 1.0]  # Side-to-side
                amps[ji] = amp
                phases[ji] = depth * phase_spread
            else:
                amps[ji] = amp * 0.05
                axes[ji] = rng.normal(0, 1, 3)
                axes[ji] /= np.linalg.norm(axes[ji]) + 1e-8

        elif params.motion_type == "hinge_sweep":
            # All joints rotate around a single axis (like a mechanical arm)
            axes[ji] = [1.0, 0.0, 0.0]
            amps[ji] = amp
            phases[ji] = depth * phase_spread * 0.5

        elif params.motion_type == "branch_bend":
            # Wave propagation from root to tips
            axes[ji] = [0.0, 0.0, 1.0]
            amps[ji] = amp * min(1.0, depth / max(max_depth * 0.5, 1))
            phases[ji] = depth * phase_spread

        else:  # generic_articulation
            axes[ji] = rng.normal(0, 1, 3)
            axes[ji] /= np.linalg.norm(axes[ji]) + 1e-8
            amps[ji] = amp * rng.uniform(0.3, 1.0)
            phases[ji] = rng.uniform(0, 2 * math.pi)

    # Generate time-varying quaternions
    rotations = np.zeros((T, J, 4), dtype=np.float32)
    rotations[:, :, 0] = 1.0  # Identity quaternion w=1

    for t in range(T):
        time_sec = t / params.fps
        for ji in range(J):
            if amps[ji] < 1e-6:
                continue

            angle = amps[ji] * math.sin(2 * math.pi * freq * time_sec + phases[ji])
            axis = axes[ji]

            # Axis-angle to quaternion
            half = angle / 2.0
            w = math.cos(half)
            s = math.sin(half)
            rotations[t, ji] = [w, axis[0] * s, axis[1] * s, axis[2] * s]

    return rotations


def _compute_depths(parent_ids: List[Optional[int]]) -> List[int]:
    """Compute depth of each joint in the hierarchy tree."""
    J = len(parent_ids)
    depths = [0] * J
    for ji in range(J):
        d = 0
        current = ji
        visited = set()
        while parent_ids[current] is not None and current not in visited:
            visited.add(current)
            current = parent_ids[current]
            d += 1
        depths[ji] = d
    return depths


def get_motion_preset(motion_type: str, **overrides) -> MotionParams:
    """Get a MotionParams with preset defaults, allowing overrides."""
    preset = MOTION_PRESETS.get(motion_type, MOTION_PRESETS["generic_articulation"])
    params = MotionParams(motion_type=motion_type, **preset)
    for k, v in overrides.items():
        if hasattr(params, k):
            setattr(params, k, v)
    return params


def apply_motion_to_blender_armature(
    armature_obj,
    rotations: np.ndarray,
    fps: int = 24,
    action_name: str = "synth_motion",
):
    """
    Apply generated motion curves as keyframes on a Blender armature.
    Must be called inside Blender.
    
    Args:
        armature_obj: Blender armature object
        rotations: [T, J, 4] quaternion rotations (wxyz format)
        fps: frames per second
        action_name: name for the created Blender Action
    """
    import bpy

    T, J, _ = rotations.shape
    scene = bpy.context.scene
    scene.render.fps = fps

    # Create or get action
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()

    action = bpy.data.actions.new(name=action_name)
    armature_obj.animation_data.action = action

    pose_bones = armature_obj.pose.bones

    for ji, pb in enumerate(pose_bones):
        if ji >= J:
            break

        pb.rotation_mode = 'QUATERNION'

        for t in range(T):
            frame = t + 1  # Blender frames are 1-indexed
            w, x, y, z = rotations[t, ji]
            pb.rotation_quaternion = (w, x, y, z)
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    scene.frame_start = 1
    scene.frame_end = T
