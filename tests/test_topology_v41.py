"""
Regression tests for HyperBone v4.1 skinning-aware topology.

Track A: Rigged/skinned asset topology. Tests verify:
  - Config loads and matches locked values
  - Bone-to-joint skinning mapping produces valid output
  - v4.1 optimizer produces no cycles
  - selected/GT ratio near 1.0
  - F1 does not drop below smoke threshold on test cache
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.rigs.skinning_topology_features import (
    build_bone_to_joint_map,
    build_joint_influence_matrix,
    compute_skinning_edge_features,
    compute_skinning_score_matrices,
)
from hyperbone.rigs.topology_optimizers import (
    hybrid_skinning_cost_optimize,
    OPTIMIZER_REGISTRY,
)
from hyperbone.rigs.undirected_topology import edge_prf, graph_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "topology" / "hyperbone_v4_1_default.json"
TEST_CACHE_PATH = PROJECT_ROOT / "outputs" / "models" / "hyperbone_v4_1_skinning_topology" / "cache_test_v41.pt"

LOCKED_CONFIG = {
    "candidate_k": 16,
    "neural_weight": 1.0,
    "skin_cosine_weight": 4.0,
    "shared_weight": 4.0,
    "distance_weight": 1.0,
}

SMOKE_F1_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_file_exists(self):
        assert CONFIG_PATH.exists(), f"v4.1 config not found at {CONFIG_PATH}"

    def test_config_matches_locked_values(self):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        for key, expected in LOCKED_CONFIG.items():
            assert cfg[key] == expected, f"Config {key}: expected {expected}, got {cfg[key]}"

    def test_config_track_label(self):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        assert cfg["track"] == "rigged_skinned_asset_topology"

    def test_config_has_caveat(self):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        assert "caveat" in cfg
        assert "unrigged" in cfg["caveat"].lower() or "not available" in cfg["caveat"].lower()

    def test_optimizer_in_registry(self):
        assert "hybrid_skinning_cost_mst" in OPTIMIZER_REGISTRY


# ---------------------------------------------------------------------------
# Skinning feature tests (synthetic fixtures)
# ---------------------------------------------------------------------------

class TestSkinningFeatures:
    @pytest.fixture
    def simple_rig(self):
        """3-joint chain: 0 -- 1 -- 2, 2 bones, 10 mesh vertices."""
        joints = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        bones = np.array([
            [0, 0, 0, 1, 0, 0],  # bone 0: joint 0 -> joint 1
            [1, 0, 0, 2, 0, 0],  # bone 1: joint 1 -> joint 2
        ], dtype=np.float32)
        n_joints, n_bones = 3, 2

        # 10 verts: 5 near joint 0-1, 5 near joint 1-2
        mesh_pc = np.array([
            [0.1, 0, 0], [0.3, 0, 0], [0.5, 0, 0], [0.7, 0, 0], [0.9, 0, 0],
            [1.1, 0, 0], [1.3, 0, 0], [1.5, 0, 0], [1.7, 0, 0], [1.9, 0, 0],
        ], dtype=np.float32)

        # Skinning: first 5 verts -> bone 0, last 5 -> bone 1
        mesh_skins_index = np.array([
            [0, -1], [0, -1], [0, -1], [0, -1], [0, -1],
            [1, -1], [1, -1], [1, -1], [1, -1], [1, -1],
        ], dtype=np.int64)
        mesh_skins_weight = np.array([
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        ], dtype=np.float32)

        return {
            "joints": joints, "bones": bones,
            "n_joints": n_joints, "n_bones": n_bones,
            "mesh_pc": mesh_pc,
            "mesh_skins_index": mesh_skins_index,
            "mesh_skins_weight": mesh_skins_weight,
        }

    def test_bone_to_joint_map(self, simple_rig):
        bone_start, bone_end = build_bone_to_joint_map(
            simple_rig["bones"], simple_rig["joints"],
            simple_rig["n_bones"], simple_rig["n_joints"],
        )
        assert len(bone_start) == 2
        assert len(bone_end) == 2
        # Bone 0: joint 0 -> joint 1
        assert bone_start[0] == 0
        assert bone_end[0] == 1
        # Bone 1: joint 1 -> joint 2
        assert bone_start[1] == 1
        assert bone_end[1] == 2

    def test_joint_influence_matrix(self, simple_rig):
        bone_start, bone_end = build_bone_to_joint_map(
            simple_rig["bones"], simple_rig["joints"],
            simple_rig["n_bones"], simple_rig["n_joints"],
        )
        influence, joint_verts = build_joint_influence_matrix(
            simple_rig["mesh_skins_index"], simple_rig["mesh_skins_weight"],
            simple_rig["n_joints"], simple_rig["n_bones"],
            bone_start, bone_end,
        )
        # Joint 0 gets influence from bone 0 (start)
        assert len(joint_verts[0]) > 0
        # Joint 1 gets influence from bone 0 (end) AND bone 1 (start)
        assert len(joint_verts[1]) > 0
        # Joint 2 (leaf) gets influence from bone 1 (end) — the critical fix
        assert len(joint_verts[2]) > 0, "Leaf joint must receive influence from parent bone endpoint"

    def test_leaf_joint_gets_influence(self, simple_rig):
        """The v1 bug: leaf joints had zero influence. Must not regress."""
        bone_start, bone_end = build_bone_to_joint_map(
            simple_rig["bones"], simple_rig["joints"],
            simple_rig["n_bones"], simple_rig["n_joints"],
        )
        influence, joint_verts = build_joint_influence_matrix(
            simple_rig["mesh_skins_index"], simple_rig["mesh_skins_weight"],
            simple_rig["n_joints"], simple_rig["n_bones"],
            bone_start, bone_end,
        )
        leaf_influence = influence[2].toarray().flatten()
        assert leaf_influence.sum() > 0, "Leaf joint 2 must have nonzero skinning influence"

    def test_connected_joints_share_influence(self, simple_rig):
        """Adjacent joints on the same bone should share vertex influence."""
        bone_start, bone_end = build_bone_to_joint_map(
            simple_rig["bones"], simple_rig["joints"],
            simple_rig["n_bones"], simple_rig["n_joints"],
        )
        influence, joint_verts = build_joint_influence_matrix(
            simple_rig["mesh_skins_index"], simple_rig["mesh_skins_weight"],
            simple_rig["n_joints"], simple_rig["n_bones"],
            bone_start, bone_end,
        )
        # Joints 0 and 1 share bone 0's vertices
        shared_01 = joint_verts[0] & joint_verts[1]
        assert len(shared_01) > 0, "Connected joints should share vertex influence"

    def test_skinning_cosine_positive_for_neighbors(self, simple_rig):
        bone_start, bone_end = build_bone_to_joint_map(
            simple_rig["bones"], simple_rig["joints"],
            simple_rig["n_bones"], simple_rig["n_joints"],
        )
        influence, joint_verts = build_joint_influence_matrix(
            simple_rig["mesh_skins_index"], simple_rig["mesh_skins_weight"],
            simple_rig["n_joints"], simple_rig["n_bones"],
            bone_start, bone_end,
        )
        feats = compute_skinning_edge_features(
            simple_rig["joints"], influence, joint_verts,
            simple_rig["mesh_pc"], [(0, 1), (1, 2), (0, 2)],
        )
        # Adjacent pairs should have positive cosine
        assert feats["skinning_cosine"][0] > 0, "Adjacent joint pair (0,1) should have positive cosine"
        assert feats["skinning_cosine"][1] > 0, "Adjacent joint pair (1,2) should have positive cosine"


# ---------------------------------------------------------------------------
# Optimizer tests (synthetic)
# ---------------------------------------------------------------------------

class TestOptimizer:
    def test_produces_no_cycles_on_chain(self):
        """5-node chain: optimizer should produce a tree (no cycles)."""
        N = 5
        joint_pos = torch.zeros(N, 3)
        for i in range(N):
            joint_pos[i, 0] = float(i)
        active_mask = torch.ones(N, dtype=torch.bool)
        neural_scores = torch.randn(N, N) * 0.1
        skin_cos = torch.zeros(N, N)
        msw = torch.zeros(N, N)
        # Set high skinning scores for adjacent pairs
        for i in range(N - 1):
            skin_cos[i, i+1] = 1.0
            skin_cos[i+1, i] = 1.0
            msw[i, i+1] = 0.5
            msw[i+1, i] = 0.5

        pred = hybrid_skinning_cost_optimize(
            joint_pos=joint_pos, active_mask=active_mask,
            neural_scores=neural_scores,
            skinning_cosine=skin_cos, max_shared_weight=msw,
            mode="density_normalized_mst", k=4,
            distance_weight=1.0, neural_weight=1.0,
            skin_cosine_weight=4.0, shared_weight=4.0,
        )
        stats = graph_stats(pred, active_mask)
        assert stats["cycle_count"] == 0, f"MST should have no cycles, got {stats['cycle_count']}"

    def test_selected_edges_equals_n_minus_1(self):
        """MST on N nodes should select exactly N-1 edges."""
        N = 8
        joint_pos = torch.randn(N, 3)
        active_mask = torch.ones(N, dtype=torch.bool)

        pred = hybrid_skinning_cost_optimize(
            joint_pos=joint_pos, active_mask=active_mask,
            neural_scores=torch.randn(N, N) * 0.1,
            mode="density_normalized_mst", k=7,
        )
        n_edges = _edge_count(pred)
        assert n_edges == N - 1, f"Expected {N-1} edges, got {n_edges}"

    def test_handles_none_skinning(self):
        """Optimizer should work when skinning matrices are None."""
        N = 5
        joint_pos = torch.randn(N, 3)
        active_mask = torch.ones(N, dtype=torch.bool)

        pred = hybrid_skinning_cost_optimize(
            joint_pos=joint_pos, active_mask=active_mask,
            neural_scores=torch.randn(N, N) * 0.1,
            skinning_cosine=None, max_shared_weight=None,
            mode="density_normalized_mst", k=4,
        )
        stats = graph_stats(pred, active_mask)
        assert stats["cycle_count"] == 0


def _edge_count(edge_mask):
    tri = torch.triu(torch.ones_like(edge_mask, dtype=torch.bool), diagonal=1)
    return int((edge_mask & tri).sum().item())


# ---------------------------------------------------------------------------
# Smoke test on real data (requires test cache)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TEST_CACHE_PATH.exists(), reason="Test cache not available")
class TestSmoke:
    @pytest.fixture(scope="class")
    def cached(self):
        return torch.load(str(TEST_CACHE_PATH), map_location="cpu", weights_only=False)

    def test_cache_not_empty(self, cached):
        assert len(cached) > 0

    def test_cache_has_skinning_fields(self, cached):
        s = cached[0]
        assert "skinning_cosine" in s, "Cache must include skinning_cosine"
        assert "max_shared_weight" in s, "Cache must include max_shared_weight"

    def test_mean_f1_above_threshold(self, cached):
        """v4.1 mean F1 must not drop below the smoke threshold."""
        f1s = []
        for s in cached[:50]:  # First 50 samples for speed
            pred = hybrid_skinning_cost_optimize(
                joint_pos=s["joint_pos"], active_mask=s["active_mask"],
                neural_scores=s["neural_scores"],
                skinning_cosine=s.get("skinning_cosine"),
                max_shared_weight=s.get("max_shared_weight"),
                mode="density_normalized_mst", k=16,
                distance_weight=1.0, neural_weight=1.0,
                skin_cosine_weight=4.0, shared_weight=4.0,
                degree_penalty=0.05, long_edge_penalty=0.25,
                mutual_bonus=0.2, max_degree=4,
                candidate_mask=s["candidate_mask"],
            )
            prf = edge_prf(pred, s["gt_adj"], s["active_mask"])
            f1s.append(prf["f1"])

        mean_f1 = float(np.mean(f1s))
        assert mean_f1 >= SMOKE_F1_THRESHOLD, (
            f"v4.1 mean F1 = {mean_f1:.4f}, below smoke threshold {SMOKE_F1_THRESHOLD}"
        )

    def test_no_cycles_on_real_data(self, cached):
        """v4.1 should produce zero cycles on real data."""
        total_cycles = 0
        for s in cached[:50]:
            pred = hybrid_skinning_cost_optimize(
                joint_pos=s["joint_pos"], active_mask=s["active_mask"],
                neural_scores=s["neural_scores"],
                skinning_cosine=s.get("skinning_cosine"),
                max_shared_weight=s.get("max_shared_weight"),
                mode="density_normalized_mst", k=16,
                distance_weight=1.0, neural_weight=1.0,
                skin_cosine_weight=4.0, shared_weight=4.0,
                candidate_mask=s["candidate_mask"],
            )
            stats = graph_stats(pred, s["active_mask"])
            total_cycles += stats["cycle_count"]
        assert total_cycles == 0, f"Expected 0 total cycles, got {total_cycles}"

    def test_selected_to_gt_near_one(self, cached):
        """sel/GT ratio should be near 1.0 (MST produces N-1 edges)."""
        ratios = []
        for s in cached[:50]:
            pred = hybrid_skinning_cost_optimize(
                joint_pos=s["joint_pos"], active_mask=s["active_mask"],
                neural_scores=s["neural_scores"],
                skinning_cosine=s.get("skinning_cosine"),
                max_shared_weight=s.get("max_shared_weight"),
                mode="density_normalized_mst", k=16,
                distance_weight=1.0, neural_weight=1.0,
                skin_cosine_weight=4.0, shared_weight=4.0,
                candidate_mask=s["candidate_mask"],
            )
            sel = _edge_count(pred)
            gt = _edge_count(s["gt_adj"])
            if gt > 0:
                ratios.append(sel / gt)

        mean_ratio = float(np.mean(ratios))
        assert 0.95 <= mean_ratio <= 1.05, (
            f"Mean sel/GT ratio = {mean_ratio:.3f}, expected near 1.0"
        )
