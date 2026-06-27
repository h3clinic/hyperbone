"""Graph extraction from skeleton images.

Converts a 1-pixel-wide skeleton into a node/edge graph,
then prunes tiny branches.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


def skeleton_to_graph(skeleton: np.ndarray, min_branch_length: int = 10) -> Dict:
    """Convert a skeleton image to a graph dict with nodes and edges.

    Returns:
        {
            "nodes": [{"id": int, "type": str, "xy": [x, y]}, ...],
            "edges": [{"parent": int, "child": int, "length_px": float}, ...]
        }
    """
    # Find skeleton points
    skel_binary = (skeleton > 127).astype(np.uint8)

    if skel_binary.sum() == 0:
        return {"nodes": [], "edges": []}

    # Classify pixels by neighbor count
    # Kernel to count 8-connected neighbors
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(skel_binary, -1, kernel)
    neighbor_count = neighbor_count * skel_binary  # only skeleton pixels

    # Endpoints: 1 neighbor
    endpoints = np.argwhere((neighbor_count == 1) & (skel_binary == 1))
    # Junctions: 3+ neighbors
    junctions = np.argwhere((neighbor_count >= 3) & (skel_binary == 1))

    # Build node list from endpoints and junctions
    nodes = []
    node_positions = {}  # (y, x) -> node_id

    for pt in junctions:
        nid = len(nodes)
        nodes.append({"id": nid, "type": "junction", "xy": [int(pt[1]), int(pt[0])]})
        node_positions[(int(pt[0]), int(pt[1]))] = nid

    for pt in endpoints:
        nid = len(nodes)
        nodes.append({"id": nid, "type": "endpoint", "xy": [int(pt[1]), int(pt[0])]})
        node_positions[(int(pt[0]), int(pt[1]))] = nid

    # If no special points found but skeleton exists, create single node at centroid
    if not nodes:
        ys, xs = np.where(skel_binary == 1)
        cx, cy = int(xs.mean()), int(ys.mean())
        nodes.append({"id": 0, "type": "center", "xy": [cx, cy]})
        return {"nodes": nodes, "edges": []}

    # Trace edges between nodes by walking along skeleton branches
    edges = _trace_edges(skel_binary, nodes, node_positions)

    # Prune short branches (edges to endpoints shorter than threshold)
    edges, nodes = _prune_short_branches(edges, nodes, min_branch_length)

    # Re-index nodes
    nodes, edges = _reindex(nodes, edges)

    return {"nodes": nodes, "edges": edges}


def _trace_edges(
    skel: np.ndarray, nodes: List[Dict], node_positions: Dict
) -> List[Dict]:
    """Trace skeleton paths between nodes to form edges."""
    h, w = skel.shape
    visited_edges = set()
    edges = []

    # For each node, walk outward along skeleton pixels
    for node in nodes:
        x, y = node["xy"]
        nid = node["id"]

        # Find neighboring skeleton pixels
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] == 1:
                    # Walk until we hit another node or dead end
                    target_id, length = _walk_to_node(
                        skel, (ny, nx), (y, x), node_positions, h, w
                    )
                    if target_id is not None and target_id != nid:
                        edge_key = (min(nid, target_id), max(nid, target_id))
                        if edge_key not in visited_edges:
                            visited_edges.add(edge_key)
                            edges.append({
                                "parent": nid,
                                "child": target_id,
                                "length_px": float(length),
                            })

    return edges


def _walk_to_node(
    skel: np.ndarray,
    start: Tuple[int, int],
    came_from: Tuple[int, int],
    node_positions: Dict,
    h: int, w: int,
    max_steps: int = 5000,
) -> Tuple[int, int]:
    """Walk along skeleton from start until hitting a node. Return (node_id, path_length)."""
    cy, cx = start
    prev_y, prev_x = came_from
    length = 1

    for _ in range(max_steps):
        # Check if current pixel is a node
        if (cy, cx) in node_positions:
            return node_positions[(cy, cx)], length

        # Find next pixel (not the one we came from)
        found_next = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if (ny, nx) == (prev_y, prev_x):
                    continue
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] == 1:
                    prev_y, prev_x = cy, cx
                    cy, cx = ny, nx
                    length += 1
                    found_next = True
                    break
            if found_next:
                break

        if not found_next:
            break

    return None, length


def _prune_short_branches(
    edges: List[Dict], nodes: List[Dict], min_length: int
) -> Tuple[List[Dict], List[Dict]]:
    """Remove edges to endpoint nodes that are shorter than min_length."""
    endpoint_ids = {n["id"] for n in nodes if n["type"] == "endpoint"}
    kept_edges = []
    removed_node_ids = set()

    for edge in edges:
        is_short = edge["length_px"] < min_length
        touches_endpoint = edge["parent"] in endpoint_ids or edge["child"] in endpoint_ids
        if is_short and touches_endpoint:
            # Remove the endpoint node
            if edge["parent"] in endpoint_ids:
                removed_node_ids.add(edge["parent"])
            if edge["child"] in endpoint_ids:
                removed_node_ids.add(edge["child"])
        else:
            kept_edges.append(edge)

    kept_nodes = [n for n in nodes if n["id"] not in removed_node_ids]
    return kept_edges, kept_nodes


def _reindex(nodes: List[Dict], edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Re-assign sequential IDs after pruning."""
    old_to_new = {}
    for i, node in enumerate(nodes):
        old_to_new[node["id"]] = i
        node["id"] = i

    new_edges = []
    for edge in edges:
        if edge["parent"] in old_to_new and edge["child"] in old_to_new:
            edge["parent"] = old_to_new[edge["parent"]]
            edge["child"] = old_to_new[edge["child"]]
            new_edges.append(edge)

    return nodes, new_edges
