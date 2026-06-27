from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StructuredDecodeConfig:
	active_threshold: float = 0.5
	edge_threshold: float = 0.5
	edge_decode_mode: str = "threshold"
	decoder_k: int = 6
	max_degree: int = 4
	max_components: int = 1
	edge_alpha: float = 1.0
	edge_beta: float = 0.5
	edge_gamma: float = 1.0
	edge_eta: float = 0.25
	min_edge_length: float = 1e-4
	max_edge_length_fraction: float = 0.95
	long_edge_override_prob: float = 0.9
	outlier_long_mult: float = 4.0
	outlier_short_mult: float = 0.1


def decode_structured_graph(
	positions: np.ndarray,
	active_prob: np.ndarray,
	edge_prob: np.ndarray,
	point_cloud: np.ndarray | None = None,
	node_confidence: np.ndarray | None = None,
	config: StructuredDecodeConfig | None = None,
) -> dict:
	"""Decode a sparse graph from dense node and edge predictions."""
	if config is None:
		config = StructuredDecodeConfig()

	n_nodes = positions.shape[0]
	if node_confidence is None:
		node_confidence = active_prob

	active_mask = active_prob > config.active_threshold
	active_idx = np.where(active_mask)[0]
	adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)

	if active_idx.shape[0] < 2:
		return {
			"active_mask": active_mask,
			"adjacency": adj,
			"selected_edges": [],
			"edge_scores": {},
			"metadata": _graph_metadata(active_idx, []),
		}

	active_pos = positions[active_idx]
	bbox_diag = np.linalg.norm(active_pos.max(axis=0) - active_pos.min(axis=0))
	bbox_diag = float(max(bbox_diag, 1e-6))
	pair_dists = _pairwise_distances(active_pos)
	upper_pair_dists = pair_dists[np.triu_indices(len(active_idx), k=1)]
	p90_dist = float(np.percentile(upper_pair_dists, 90)) if upper_pair_dists.size else 0.0
	median_dist = float(np.median(upper_pair_dists)) if upper_pair_dists.size else 0.0

	knn_neighbors = _knn_neighbors(active_pos, max(config.decoder_k, 1))
	candidates = []
	for i_local, i_global in enumerate(active_idx):
		for j_local in range(i_local + 1, len(active_idx)):
			j_global = int(active_idx[j_local])
			dist = float(np.linalg.norm(positions[i_global] - positions[j_global]))
			if dist <= config.min_edge_length:
				continue

			prob = float(edge_prob[i_global, j_global])
			if dist > bbox_diag * config.max_edge_length_fraction and prob < config.long_edge_override_prob:
				continue

			if median_dist > 0.0:
				if dist > median_dist * config.outlier_long_mult:
					continue
				if dist < median_dist * config.outlier_short_mult:
					continue

			norm_dist = dist / bbox_diag
			excessive_penalty = max(0.0, dist - p90_dist) / bbox_diag if p90_dist > 0.0 else 0.0
			pair_conf = 0.5 * (float(node_confidence[i_global]) + float(node_confidence[j_global]))
			score = (
				config.edge_alpha * prob
				- config.edge_beta * norm_dist
				- config.edge_gamma * excessive_penalty
				+ config.edge_eta * pair_conf
			)
			candidates.append({
				"i_local": i_local,
				"j_local": j_local,
				"i_global": int(i_global),
				"j_global": j_global,
				"dist": dist,
				"prob": prob,
				"score": float(score),
				"knn_ok": (j_local in knn_neighbors[i_local]) or (i_local in knn_neighbors[j_local]),
			})

	mode = config.edge_decode_mode
	if mode == "threshold":
		selected = [c for c in candidates if c["prob"] >= config.edge_threshold]
	elif mode == "mst":
		selected = _kruskal_select(candidates, len(active_idx), None, config.max_components)
	elif mode == "knn_mst":
		knn_candidates = [c for c in candidates if c["knn_ok"]]
		selected = _kruskal_select(knn_candidates, len(active_idx), config.max_degree, 1)
		selected = _bridge_components_if_needed(selected, candidates, len(active_idx), config.max_degree)
	elif mode == "forest_mst":
		selected = _kruskal_select(candidates, len(active_idx), config.max_degree, config.max_components)
	elif mode == "degree_limited":
		selected = _kruskal_select(candidates, len(active_idx), config.max_degree, 1)
	else:
		raise ValueError(f"Unknown edge decode mode: {mode}")

	selected_edges = []
	edge_scores = {}
	for c in selected:
		i_global = c["i_global"]
		j_global = c["j_global"]
		adj[i_global, j_global] = 1.0
		adj[j_global, i_global] = 1.0
		selected_edges.append((i_global, j_global))
		edge_scores[f"{i_global}-{j_global}"] = c["score"]

	metadata = _graph_metadata(active_idx, selected_edges)
	metadata.update({
		"mode": mode,
		"bbox_diag": bbox_diag,
		"median_candidate_distance": median_dist,
		"p90_candidate_distance": p90_dist,
	})
	return {
		"active_mask": active_mask,
		"adjacency": adj,
		"selected_edges": selected_edges,
		"edge_scores": edge_scores,
		"metadata": metadata,
	}


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
	diff = points[:, None, :] - points[None, :, :]
	return np.linalg.norm(diff, axis=-1)


def _knn_neighbors(points: np.ndarray, k: int) -> list[set[int]]:
	dists = _pairwise_distances(points)
	neighbors = []
	for i in range(points.shape[0]):
		order = np.argsort(dists[i])
		order = [int(j) for j in order if j != i][:k]
		neighbors.append(set(order))
	return neighbors


class _UnionFind:
	def __init__(self, n: int):
		self.parent = list(range(n))
		self.rank = [0] * n

	def find(self, x: int) -> int:
		while self.parent[x] != x:
			self.parent[x] = self.parent[self.parent[x]]
			x = self.parent[x]
		return x

	def union(self, a: int, b: int) -> bool:
		ra, rb = self.find(a), self.find(b)
		if ra == rb:
			return False
		if self.rank[ra] < self.rank[rb]:
			ra, rb = rb, ra
		self.parent[rb] = ra
		if self.rank[ra] == self.rank[rb]:
			self.rank[ra] += 1
		return True


def _kruskal_select(candidates: list[dict], n_active: int, max_degree: int | None, max_components: int) -> list[dict]:
	uf = _UnionFind(n_active)
	degrees = [0] * n_active
	selected = []
	n_components = n_active
	for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
		i_local = c["i_local"]
		j_local = c["j_local"]
		if max_degree is not None and (degrees[i_local] >= max_degree or degrees[j_local] >= max_degree):
			continue
		if uf.union(i_local, j_local):
			selected.append(c)
			degrees[i_local] += 1
			degrees[j_local] += 1
			n_components -= 1
			if n_components <= max_components:
				break
	return selected


def _bridge_components_if_needed(selected: list[dict], candidates: list[dict], n_active: int, max_degree: int | None) -> list[dict]:
	uf = _UnionFind(n_active)
	degrees = [0] * n_active
	for c in selected:
		uf.union(c["i_local"], c["j_local"])
		degrees[c["i_local"]] += 1
		degrees[c["j_local"]] += 1

	if len({uf.find(i) for i in range(n_active)}) <= 1:
		return selected

	remaining = [c for c in sorted(candidates, key=lambda x: x["score"], reverse=True) if c not in selected]
	for c in remaining:
		i_local = c["i_local"]
		j_local = c["j_local"]
		if uf.find(i_local) == uf.find(j_local):
			continue
		if max_degree is not None and (degrees[i_local] >= max_degree or degrees[j_local] >= max_degree):
			continue
		uf.union(i_local, j_local)
		degrees[i_local] += 1
		degrees[j_local] += 1
		selected.append(c)
		if len({uf.find(i) for i in range(n_active)}) <= 1:
			break
	return selected


def _graph_metadata(active_idx: np.ndarray, selected_edges: list[tuple[int, int]]) -> dict:
	n_active = int(active_idx.shape[0])
	degrees = {int(i): 0 for i in active_idx.tolist()}
	for i, j in selected_edges:
		degrees[i] += 1
		degrees[j] += 1
	comp_count, largest_ratio = _component_stats(active_idx.tolist(), selected_edges)
	avg_degree = float(np.mean(list(degrees.values()))) if degrees else 0.0
	max_degree = int(max(degrees.values())) if degrees else 0
	return {
		"pred_edge_count": len(selected_edges),
		"component_count": comp_count,
		"connected_ratio": largest_ratio,
		"average_degree": avg_degree,
		"max_degree": max_degree,
	}


def _component_stats(active_nodes: list[int], edges: list[tuple[int, int]]) -> tuple[int, float]:
	if not active_nodes:
		return 0, 0.0
	adj = {n: set() for n in active_nodes}
	for i, j in edges:
		if i in adj and j in adj:
			adj[i].add(j)
			adj[j].add(i)
	seen = set()
	comp_sizes = []
	for n in active_nodes:
		if n in seen:
			continue
		stack = [n]
		size = 0
		while stack:
			cur = stack.pop()
			if cur in seen:
				continue
			seen.add(cur)
			size += 1
			stack.extend(adj[cur] - seen)
		comp_sizes.append(size)
	largest = max(comp_sizes) if comp_sizes else 0
	return len(comp_sizes), largest / max(len(active_nodes), 1)
