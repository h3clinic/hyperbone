from __future__ import annotations

import numpy as np

from hyperbone.rigs.parent_decoder import ParentDecodeConfig, decode_parent_graph


def _perfect_logits(expected_parent: list[int], active_mask: list[bool]) -> np.ndarray:
    n = len(expected_parent)
    root_class = n
    logits = np.full((n, n + 1), -20.0, dtype=np.float32)
    for child in range(n):
        if not active_mask[child]:
            logits[child, root_class] = 20.0
            continue
        p = int(expected_parent[child])
        if p < 0 or p >= n or p == child or not active_mask[p]:
            logits[child, root_class] = 20.0
        else:
            logits[child, p] = 20.0
    return logits


def _decode(expected_parent: list[int], active_mask: list[bool], mode: str = "parent_argmax_acyclic") -> dict:
    n = len(expected_parent)
    pos = np.zeros((n, 3), dtype=np.float32)
    active_prob = np.asarray([1.0 if a else 0.0 for a in active_mask], dtype=np.float32)
    logits = _perfect_logits(expected_parent, active_mask)
    cfg = ParentDecodeConfig(active_threshold=0.5, decode_mode=mode, max_degree=n)
    return decode_parent_graph(pos, active_prob, logits, config=cfg)


def _edge_set(parent_ptr: np.ndarray, active_mask: np.ndarray) -> set[tuple[int, int]]:
    edges = set()
    for child, parent in enumerate(parent_ptr.tolist()):
        if parent >= 0 and active_mask[child] and active_mask[parent]:
            a, b = sorted((int(parent), int(child)))
            edges.add((a, b))
    return edges


def test_single_root_tree_identity() -> None:
    # 0 is root, 1->0, 2->1, 3->1
    expected = [-1, 0, 1, 1]
    active = [True, True, True, True]
    decoded = _decode(expected, active)
    parent_ptr = decoded["parent_ptr"]
    np.testing.assert_array_equal(parent_ptr, np.array(expected, dtype=np.int64))


def test_two_root_forest_identity() -> None:
    # roots: 0 and 3; 1->0, 2->1, 4->3
    expected = [-1, 0, 1, -1, 3]
    active = [True, True, True, True, True]
    decoded = _decode(expected, active)
    np.testing.assert_array_equal(decoded["parent_ptr"], np.array(expected, dtype=np.int64))


def test_padded_nodes_ignored() -> None:
    expected = [-1, 0, 1, 2]
    active = [True, True, True, False]
    decoded = _decode(expected, active)
    edges = decoded["edges"]
    assert all(3 not in e for e in edges)


def test_self_parent_rejected_to_root() -> None:
    # child 1 self-parent should become root/no-parent.
    n = 3
    root_class = n
    pos = np.zeros((n, 3), dtype=np.float32)
    active_prob = np.ones((n,), dtype=np.float32)
    logits = np.full((n, n + 1), -20.0, dtype=np.float32)
    logits[0, root_class] = 20.0
    logits[1, 1] = 20.0
    logits[2, 1] = 20.0
    cfg = ParentDecodeConfig(active_threshold=0.5, decode_mode="parent_argmax_acyclic", max_degree=n)
    decoded = decode_parent_graph(pos, active_prob, logits, config=cfg)
    assert int(decoded["parent_ptr"][1]) == -1


def test_active_mask_suppresses_padded_nodes() -> None:
    expected = [-1, 0, 1, 2, 3]
    active = [True, True, True, False, False]
    decoded = _decode(expected, active)
    parent_ptr = decoded["parent_ptr"]
    assert int(parent_ptr[3]) == -1
    assert int(parent_ptr[4]) == -1


def test_perfect_logits_reconstruct_expected_edges() -> None:
    expected = [-1, 0, 0, 2, 2]
    active = [True, True, True, True, True]
    decoded = _decode(expected, active)
    pred_edges = _edge_set(decoded["parent_ptr"], decoded["active_mask"])
    gt_edges = {(0, 1), (0, 2), (2, 3), (2, 4)}
    assert pred_edges == gt_edges
