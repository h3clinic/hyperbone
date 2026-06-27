from __future__ import annotations

import numpy as np

from hyperbone.rigs.parent_targets import normalize_parent_index


def _active(n: int, inactive: list[int] | None = None) -> np.ndarray:
    a = np.ones((n,), dtype=bool)
    if inactive:
        for i in inactive:
            a[i] = False
    return a


def test_valid_tree_unchanged() -> None:
    p = np.array([-1, 0, 1, 1], dtype=np.int64)
    n, root, valid, meta = normalize_parent_index(p, _active(4))
    np.testing.assert_array_equal(n, p)
    np.testing.assert_array_equal(root, np.array([1, 0, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(valid, np.array([0, 1, 1, 1], dtype=np.float32))
    assert meta["cycles_detected"] == 0


def test_valid_forest_unchanged() -> None:
    p = np.array([-1, 0, -1, 2], dtype=np.int64)
    n, root, valid, _ = normalize_parent_index(p, _active(4))
    np.testing.assert_array_equal(n, p)
    np.testing.assert_array_equal(root, np.array([1, 0, 1, 0], dtype=np.float32))
    np.testing.assert_array_equal(valid, np.array([0, 1, 0, 1], dtype=np.float32))


def test_self_parent_becomes_root() -> None:
    p = np.array([-1, 1, 1], dtype=np.int64)
    n, root, valid, meta = normalize_parent_index(p, _active(3))
    assert int(n[1]) == -1
    assert root[1] == 1.0
    assert valid[1] == 0.0
    assert meta["self_parent_count"] >= 1


def test_invalid_parent_becomes_root() -> None:
    p = np.array([-1, 99, 1], dtype=np.int64)
    n, root, valid, meta = normalize_parent_index(p, _active(3))
    assert int(n[1]) == -1
    assert root[1] == 1.0
    assert valid[1] == 0.0
    assert meta["invalid_parent_count"] >= 1


def test_no_root_chain_gets_root() -> None:
    # 0->1->2->1 has no explicit root after canonicalization.
    p = np.array([1, 2, 1], dtype=np.int64)
    n, root, valid, meta = normalize_parent_index(p, _active(3))
    assert int(np.sum(root > 0.5)) >= 1
    assert (meta["no_root_component_count"] >= 1 and meta["roots_added"] >= 1) or (meta["cycles_broken"] >= 1)


def test_two_cycle_breaks_to_valid_forest() -> None:
    p = np.array([1, 0], dtype=np.int64)
    n, root, valid, meta = normalize_parent_index(p, _active(2))
    # Deterministic lowest-index break => node 0 becomes root.
    assert int(n[0]) == -1
    assert int(n[1]) == 0
    assert int(np.sum(root > 0.5)) == 1
    assert meta["cycles_detected"] >= 1
    assert meta["cycles_broken"] >= 1


def test_three_cycle_breaks_to_valid_forest() -> None:
    p = np.array([1, 2, 0], dtype=np.int64)
    n, root, valid, meta = normalize_parent_index(p, _active(3))
    assert int(n[0]) == -1
    assert int(n[1]) == 2
    assert int(n[2]) == 0
    assert int(np.sum(root > 0.5)) == 1
    assert meta["cycles_detected"] >= 1


def test_padded_nodes_ignored() -> None:
    p = np.array([-1, 0, 1, 2, 3], dtype=np.int64)
    a = _active(5, inactive=[3, 4])
    n, root, valid, _ = normalize_parent_index(p, a)
    assert int(n[3]) == -1
    assert int(n[4]) == -1
    assert root[3] == 0.0
    assert root[4] == 0.0
    assert valid[3] == 0.0
    assert valid[4] == 0.0


def test_active_parent_required() -> None:
    p = np.array([-1, 2, -1], dtype=np.int64)
    a = _active(3, inactive=[2])
    n, root, valid, _ = normalize_parent_index(p, a)
    assert int(n[1]) == -1
    assert root[1] == 1.0
    assert valid[1] == 0.0
