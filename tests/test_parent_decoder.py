from __future__ import annotations

import numpy as np

from hyperbone.rigs.parent_decoder import ParentDecodeConfig, decode_parent_graph


def _logits_from_targets(targets: list[int], root_class: int) -> np.ndarray:
    n_nodes = len(targets)
    logits = np.full((n_nodes, root_class + 1), -10.0, dtype=np.float32)
    for i, t in enumerate(targets):
        logits[i, int(t)] = 10.0
    return logits


def _decode_from_targets(targets: list[int], active_mask: list[bool], mode: str = "parent_argmax_acyclic") -> dict:
    n_nodes = len(targets)
    root_class = n_nodes
    positions = np.zeros((n_nodes, 3), dtype=np.float32)
    active_prob = np.asarray([1.0 if x else 0.0 for x in active_mask], dtype=np.float32)
    parent_logits = _logits_from_targets(targets, root_class)
    cfg = ParentDecodeConfig(active_threshold=0.5, decode_mode=mode, max_degree=8)
    return decode_parent_graph(positions, active_prob, parent_logits, config=cfg)


def test_root_class_creates_no_parent_edge() -> None:
    # Node 0 is root, node 1 attaches to root.
    decoded = _decode_from_targets(targets=[2, 0], active_mask=[True, True])
    parent_ptr = decoded["parent_ptr"]
    assert int(parent_ptr[0]) == -1
    assert int(parent_ptr[1]) == 0


def test_nonroot_parent_creates_edge() -> None:
    # Root class is 3, make 1->0 and 2->1.
    decoded = _decode_from_targets(targets=[3, 0, 1], active_mask=[True, True, True])
    edges = set(decoded["edges"])
    assert (0, 1) in edges
    assert (1, 2) in edges
    assert len(edges) == 2


def test_self_parent_rejected() -> None:
    # Node 1 predicts itself, decoder should reject to root/no-parent.
    decoded = _decode_from_targets(targets=[2, 1], active_mask=[True, True])
    parent_ptr = decoded["parent_ptr"]
    assert int(parent_ptr[1]) == -1


def test_cycle_is_broken_in_acyclic_mode() -> None:
    # 0->1 and 1->0 forms a cycle; decoder should break it.
    decoded = _decode_from_targets(targets=[1, 0], active_mask=[True, True], mode="parent_argmax_acyclic")
    parent_ptr = decoded["parent_ptr"]
    # At least one node must become root to break the cycle.
    assert np.sum(parent_ptr < 0) >= 1
    assert float(decoded["metadata"]["cycle_rate"]) == 0.0


def test_gt_parent_array_reconstructs_expected_edge_count() -> None:
    # Tree with 5 nodes: root at 0, and 1/2 children of 0, 3/4 children of 2.
    decoded = _decode_from_targets(targets=[5, 0, 0, 2, 2], active_mask=[True, True, True, True, True])
    edges = set(decoded["edges"])
    expected = {(0, 1), (0, 2), (2, 3), (2, 4)}
    assert edges == expected
    assert int(decoded["edge_count"]) == 4


def test_padded_nodes_ignored() -> None:
    # Node 3 is padded/inactive; even if it has logits it must not create edges.
    decoded = _decode_from_targets(targets=[4, 0, 1, 2], active_mask=[True, True, True, False])
    edges = set(decoded["edges"])
    assert all(3 not in e for e in edges)
    assert int(decoded["edge_count"]) == 2
