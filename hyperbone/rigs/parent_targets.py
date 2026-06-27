from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ParentTargetSet:
    parent_index: np.ndarray
    root_mask: np.ndarray
    child_count: np.ndarray
    bone_vector_to_parent: np.ndarray
    depth_from_root: np.ndarray
    valid_parent_mask: np.ndarray


def normalize_parent_index(
    parent_index: np.ndarray,
    active_mask: np.ndarray,
    strategy: str = "forest_break_cycles",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Normalize parent pointers into a valid forest on active nodes.

    Conventions:
    - inactive nodes stay ignored with parent=-1
    - invalid/self/inactive-parent pointers become ROOT (-1)
    - each active component has at least one root
    - cycles are broken deterministically at the lowest cycle index
    """
    if strategy != "forest_break_cycles":
        raise ValueError(f"Unsupported normalization strategy: {strategy}")

    p_raw = np.asarray(parent_index, dtype=np.int64)
    active = np.asarray(active_mask, dtype=bool)
    n = int(p_raw.shape[0])

    p_norm = np.full(n, -1, dtype=np.int64)

    invalid_parent_count = 0
    self_parent_count = 0

    for i in range(n):
        if not active[i]:
            continue
        p = int(p_raw[i])
        if p == i:
            self_parent_count += 1
            continue
        if p < 0 or p >= n or (not active[p]):
            invalid_parent_count += 1
            continue
        p_norm[i] = p

    active_ids = np.where(active)[0].tolist()

    # Break directed cycles deterministically.
    cycles_detected = 0
    cycles_broken = 0
    state = np.zeros(n, dtype=np.int8)  # 0 unvisited, 1 visiting, 2 done
    pos_in_path = {}

    for start in active_ids:
        if state[start] != 0:
            continue
        path = []
        node = start
        pos_in_path.clear()
        while node >= 0 and active[node] and state[node] != 2:
            if state[node] == 1:
                # Back-edge to an active visiting node => cycle.
                cycle_start = pos_in_path[node]
                cycle_nodes = path[cycle_start:]
                if cycle_nodes:
                    cycles_detected += 1
                    break_node = min(cycle_nodes)
                    if p_norm[break_node] >= 0:
                        p_norm[break_node] = -1
                        cycles_broken += 1
                break

            state[node] = 1
            pos_in_path[node] = len(path)
            path.append(node)
            nxt = int(p_norm[node])
            if nxt < 0 or (not active[nxt]):
                break
            node = nxt

        for v in path:
            state[v] = 2

    # Ensure each connected component has at least one root.
    undirected = defaultdict(set)
    for i in active_ids:
        p = int(p_norm[i])
        if p >= 0:
            undirected[i].add(p)
            undirected[p].add(i)

    components = []
    seen = set()
    for i in active_ids:
        if i in seen:
            continue
        stack = [i]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            for nxt in undirected.get(cur, set()):
                if nxt not in seen:
                    stack.append(nxt)
        components.append(comp)

    no_root_component_count = 0
    roots_added = 0
    for comp in components:
        comp_roots = [node for node in comp if p_norm[node] < 0]
        if comp_roots:
            continue
        no_root_component_count += 1
        root_node = min(comp)
        p_norm[root_node] = -1
        roots_added += 1

    raw_edges = set()
    norm_edges = set()
    for i in active_ids:
        pr = int(p_raw[i])
        if 0 <= pr < n and active[pr] and pr != i:
            raw_edges.add((pr, i))
        pn = int(p_norm[i])
        if 0 <= pn < n and active[pn] and pn != i:
            norm_edges.add((pn, i))

    root_mask = np.zeros(n, dtype=np.float32)
    valid_parent_mask = np.zeros(n, dtype=np.float32)
    for i in active_ids:
        if p_norm[i] < 0:
            root_mask[i] = 1.0
        else:
            valid_parent_mask[i] = 1.0

    metadata = {
        "cycles_detected": int(cycles_detected),
        "cycles_broken": int(cycles_broken),
        "invalid_parent_count": int(invalid_parent_count),
        "self_parent_count": int(self_parent_count),
        "no_root_component_count": int(no_root_component_count),
        "roots_added": int(roots_added),
        "edges_preserved": int(len(raw_edges & norm_edges)),
        "edges_removed": int(len(raw_edges - norm_edges)),
    }
    return p_norm, root_mask, valid_parent_mask, metadata


def extract_parent_targets(joints: np.ndarray, conns: np.ndarray) -> ParentTargetSet:
    """Extract parent-pointer supervision from GT joints and parent indices.

    Rules:
    - parent_index[j] = parent joint index or -1 for roots/invalid/secondary roots
    - root_mask marks roots across all components
    - largest connected component is treated as the primary tree; other components
      are preserved as secondary roots by zeroing their parent pointers.
    """
    joints = np.asarray(joints, dtype=np.float32)
    conns = np.asarray(conns, dtype=np.int64)
    n_joints = int(joints.shape[0])

    parent_index = np.full(n_joints, -1, dtype=np.int64)
    root_mask = np.zeros(n_joints, dtype=np.float32)
    valid_parent_mask = np.zeros(n_joints, dtype=np.float32)
    bone_vector_to_parent = np.zeros((n_joints, 3), dtype=np.float32)
    depth_from_root = np.zeros(n_joints, dtype=np.int64)
    child_count = np.zeros(n_joints, dtype=np.int64)

    if n_joints == 0:
        return ParentTargetSet(
            parent_index=parent_index,
            root_mask=root_mask,
            child_count=child_count,
            bone_vector_to_parent=bone_vector_to_parent,
            depth_from_root=depth_from_root,
            valid_parent_mask=valid_parent_mask,
        )

    active_mask = np.ones(n_joints, dtype=bool)
    parent_index, root_mask, valid_parent_mask, _ = normalize_parent_index(
        conns[:n_joints],
        active_mask,
        strategy="forest_break_cycles",
    )

    # Recompute child counts and depths on the adjusted forest.
    children = defaultdict(list)
    roots = [j for j in range(n_joints) if active_mask[j] and parent_index[j] < 0]
    for j in range(n_joints):
        p = int(parent_index[j])
        if p >= 0:
            children[p].append(j)
    for j in range(n_joints):
        child_count[j] = len(children.get(j, []))

    depth_from_root[:] = 0
    queue = deque([(r, 0) for r in roots])
    visited = set()
    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        depth_from_root[node] = depth
        for child in children.get(node, []):
            queue.append((child, depth + 1))

    # Parent bone vectors for valid parents only.
    for j in range(n_joints):
        p = int(parent_index[j])
        if p >= 0:
            bone_vector_to_parent[j] = joints[p] - joints[j]

    return ParentTargetSet(
        parent_index=parent_index,
        root_mask=root_mask,
        child_count=child_count,
        bone_vector_to_parent=bone_vector_to_parent,
        depth_from_root=depth_from_root,
        valid_parent_mask=valid_parent_mask,
    )
