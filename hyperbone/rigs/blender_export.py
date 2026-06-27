"""
Blender export utilities for the Anymate pipeline.

This module contains functions to be called INSIDE Blender's Python environment.
They extract evaluated armature poses, camera parameters, and render passes.

CRITICAL RULE: Always call scene.frame_set(frame_idx) and get the evaluated
depsgraph BEFORE exporting joints. Exporting rest-pose joints is the #1 mistake.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# These imports only work inside Blender
try:
    import bpy
    import bmesh
    import mathutils
    from mathutils import Vector, Matrix, Quaternion
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def get_evaluated_armature(armature_obj):
    """
    Get the evaluated armature after animation/constraints are applied.
    
    MUST be called after scene.frame_set(frame_idx).
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return armature_obj.evaluated_get(depsgraph)


def extract_joints_at_frame(
    armature_obj,
    frame_idx: int,
    camera_obj=None,
    resolution: Tuple[int, int] = (512, 512),
) -> Tuple[List[dict], List[dict]]:
    """
    Extract all joint positions and bone data at the given frame.
    
    Returns (joints_list, bones_list) with world/camera/image coordinates.
    
    CRITICAL: This sets the frame and evaluates the depsgraph. The returned
    positions are the ANIMATED positions, not rest pose.
    """
    scene = bpy.context.scene
    scene.frame_set(frame_idx)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_eval = armature_obj.evaluated_get(depsgraph)
    pose_bones = arm_eval.pose.bones

    # Camera matrices
    cam_matrix_world = None
    cam_matrix_proj = None
    K = None
    extrinsic = None

    if camera_obj:
        cam_eval = camera_obj.evaluated_get(depsgraph)
        cam_matrix_world = cam_eval.matrix_world
        cam_matrix_view = cam_matrix_world.inverted()

        # Build intrinsic K from camera data
        cam_data = cam_eval.data
        render = scene.render
        res_x, res_y = resolution
        sensor_width = cam_data.sensor_width
        focal_length = cam_data.lens
        pixel_aspect = render.pixel_aspect_x / render.pixel_aspect_y

        fx = focal_length * res_x / sensor_width
        fy = fx / pixel_aspect
        cx = res_x / 2.0
        cy = res_y / 2.0

        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        extrinsic = [list(row) for row in cam_matrix_view]

    # Build joint name -> id mapping
    bone_names = [pb.name for pb in pose_bones]
    name_to_id = {name: idx for idx, name in enumerate(bone_names)}

    joints = []
    bones = []

    for idx, pb in enumerate(pose_bones):
        # World position (animated)
        head_world = arm_eval.matrix_world @ pb.head
        # Alternatively use pb.matrix which is bone-space:
        # bone_world_matrix = arm_eval.matrix_world @ pb.matrix

        # Camera-space position
        camera_xyz = (0.0, 0.0, 0.0)
        image_xy = (0.0, 0.0)
        visible = True

        if cam_matrix_world is not None:
            cam_view = cam_matrix_world.inverted()
            head_cam = cam_view @ head_world
            camera_xyz = (head_cam.x, head_cam.y, head_cam.z)

            # Project to image
            if head_cam.z < 0:  # In front of camera (Blender uses -Z forward)
                px = K[0][0] * (-head_cam.x / head_cam.z) + K[0][2]
                py = K[1][1] * (-head_cam.y / head_cam.z) + K[1][2]
                image_xy = (px, py)
                # Visibility: check if within image bounds
                visible = (0 <= px < resolution[0] and 0 <= py < resolution[1])
            else:
                visible = False

        # Rotations
        world_rot = (arm_eval.matrix_world @ pb.matrix).to_quaternion()
        local_rot = pb.rotation_quaternion if pb.rotation_mode == 'QUATERNION' else pb.matrix_basis.to_quaternion()

        # Rest pose position
        rest_head = arm_eval.matrix_world @ pb.bone.head_local

        parent_id = name_to_id.get(pb.parent.name) if pb.parent else None

        joint = {
            "id": idx,
            "name": pb.name,
            "parent_id": parent_id,
            "type": _guess_joint_type(pb.name),
            "rest_world_xyz": list(rest_head),
            "world_xyz": list(head_world),
            "camera_xyz": list(camera_xyz),
            "image_xy": list(image_xy),
            "visible": visible,
            "confidence": 1.0,
            "world_rotation_quat": [world_rot.w, world_rot.x, world_rot.y, world_rot.z],
            "local_rotation_quat": [local_rot.w, local_rot.x, local_rot.y, local_rot.z],
        }
        joints.append(joint)

        # Bone from parent to this joint
        if parent_id is not None:
            parent_pb = pose_bones[parent_id]
            parent_head_world = arm_eval.matrix_world @ parent_pb.head
            bone_length = (head_world - parent_head_world).length

            bone_world_rot = (arm_eval.matrix_world @ pb.matrix).to_quaternion()
            bone_local_rot = pb.matrix_basis.to_quaternion()

            bone = {
                "parent_id": parent_id,
                "child_id": idx,
                "length": bone_length,
                "world_rotation_quat": [bone_world_rot.w, bone_world_rot.x, bone_world_rot.y, bone_world_rot.z],
                "local_rotation_quat": [bone_local_rot.w, bone_local_rot.x, bone_local_rot.y, bone_local_rot.z],
            }
            bones.append(bone)

    return joints, bones


def extract_rest_skeleton(armature_obj) -> Tuple[List[dict], List[dict]]:
    """Extract the rest/bind pose skeleton (no animation evaluation needed)."""
    pose_bones = armature_obj.pose.bones
    bone_names = [pb.name for pb in pose_bones]
    name_to_id = {name: idx for idx, name in enumerate(bone_names)}

    joints = []
    bones = []

    for idx, pb in enumerate(pose_bones):
        head_world = armature_obj.matrix_world @ pb.bone.head_local
        parent_id = name_to_id.get(pb.parent.name) if pb.parent else None

        joint = {
            "id": idx,
            "name": pb.name,
            "parent_id": parent_id,
            "type": _guess_joint_type(pb.name),
            "rest_world_xyz": list(head_world),
            "world_xyz": list(head_world),
        }
        joints.append(joint)

        if parent_id is not None:
            parent_pb = pose_bones[parent_id]
            parent_head = armature_obj.matrix_world @ parent_pb.bone.head_local
            bone_length = (head_world - parent_head).length
            bone = {
                "parent_id": parent_id,
                "child_id": idx,
                "length": bone_length,
            }
            bones.append(bone)

    return joints, bones


def get_camera_params(camera_obj, resolution: Tuple[int, int] = (512, 512)) -> dict:
    """Extract camera intrinsics and extrinsics."""
    cam_data = camera_obj.data
    scene = bpy.context.scene
    res_x, res_y = resolution
    sensor_width = cam_data.sensor_width
    focal_length = cam_data.lens
    pixel_aspect = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    fx = focal_length * res_x / sensor_width
    fy = fx / pixel_aspect
    cx = res_x / 2.0
    cy = res_y / 2.0

    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    extrinsic = [list(row) for row in camera_obj.matrix_world.inverted()]

    return {
        "K": K,
        "extrinsic": extrinsic,
        "resolution": [res_x, res_y],
    }


def setup_render_passes(scene, output_dir: str, resolution: Tuple[int, int] = (512, 512)):
    """Configure scene for RGB + mask + depth rendering."""
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'

    # Enable compositing for depth/mask
    scene.use_nodes = True
    tree = scene.node_tree
    tree.links.clear()
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Render layers
    rl = tree.nodes.new('CompositorNodeRLayers')
    rl.location = (0, 0)

    # RGB output
    rgb_out = tree.nodes.new('CompositorNodeOutputFile')
    rgb_out.location = (400, 200)
    rgb_out.base_path = str(Path(output_dir) / "rgb")
    rgb_out.format.file_format = 'PNG'
    tree.links.new(rl.outputs['Image'], rgb_out.inputs[0])

    # Depth output
    scene.view_layers[0].use_pass_z = True
    depth_out = tree.nodes.new('CompositorNodeOutputFile')
    depth_out.location = (400, 0)
    depth_out.base_path = str(Path(output_dir) / "depth")
    depth_out.format.file_format = 'OPEN_EXR'
    depth_out.format.color_depth = '32'
    tree.links.new(rl.outputs['Depth'], depth_out.inputs[0])

    # Object index pass for mask
    scene.view_layers[0].use_pass_object_index = True
    mask_out = tree.nodes.new('CompositorNodeOutputFile')
    mask_out.location = (400, -200)
    mask_out.base_path = str(Path(output_dir) / "mask")
    mask_out.format.file_format = 'PNG'
    mask_out.format.color_mode = 'BW'

    # ID mask node
    id_mask = tree.nodes.new('CompositorNodeIDMask')
    id_mask.location = (200, -200)
    id_mask.index = 1  # Object pass index for the asset
    tree.links.new(rl.outputs['IndexOB'], id_mask.inputs[0])
    tree.links.new(id_mask.outputs[0], mask_out.inputs[0])


def render_frame(scene, frame_idx: int, output_dir: str):
    """Render a single frame (RGB, depth, mask are handled by compositor nodes)."""
    scene.frame_set(frame_idx)
    scene.render.filepath = str(Path(output_dir) / "rgb" / f"frame_{frame_idx:06d}.png")
    bpy.ops.render.render(write_still=True)


def _guess_joint_type(bone_name: str) -> str:
    """Heuristic to classify joint type from bone name."""
    name = bone_name.lower()
    if "root" in name or "hips" in name:
        return "root"
    elif "spine" in name or "back" in name:
        return "spine"
    elif "neck" in name:
        return "neck"
    elif "head" in name:
        return "head"
    elif "shoulder" in name or "clavicle" in name:
        return "shoulder"
    elif "elbow" in name or "forearm" in name:
        return "elbow"
    elif "wrist" in name or "hand" in name:
        return "wrist"
    elif "hip" in name or "thigh" in name or "upper_leg" in name:
        return "hip"
    elif "knee" in name or "shin" in name or "lower_leg" in name:
        return "knee"
    elif "ankle" in name or "foot" in name:
        return "foot"
    elif "tail" in name:
        return "tail"
    elif "wing" in name:
        return "wing"
    elif "fin" in name:
        return "fin"
    else:
        return "unknown"
