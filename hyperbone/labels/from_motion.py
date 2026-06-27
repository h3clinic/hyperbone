"""
Motion-derived articulation label extraction.

Given tracked points inside an object over time, cluster by coherent motion
to find functional articulation points.

Pipeline:
1. Sample points inside object mask
2. Track over frames (given or from optical flow)
3. Compute pairwise motion similarity
4. Cluster into rigid-ish parts
5. Find boundaries between clusters
6. Estimate articulation node at each boundary
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hyperbone.labels.schema import (
    GraphLabel,
    HyperNodeLabel,
    HyperEdgeLabel,
    NodeType,
    EdgeType,
    LabelSource,
)


@dataclass
class MotionArticulationConfig:
    """Configuration for motion-based articulation detection."""
    num_sample_points: int = 200
    min_frames: int = 5
    motion_threshold: float = 0.1   # min relative motion to be articulation
    cluster_method: str = "spectral"  # "spectral" or "kmeans"
    max_clusters: int = 8
    min_cluster_size: int = 10
    boundary_confidence_scale: float = 1.0


def detect_articulations_from_tracks(
    point_tracks: np.ndarray,
    masks: Optional[np.ndarray] = None,
    config: MotionArticulationConfig = MotionArticulationConfig(),
    sample_id: str = "motion",
) -> GraphLabel:
    """
    Detect articulation nodes from point tracks.

    Args:
        point_tracks: [T, N, 2] tracked point positions over T frames, N points.
        masks: [T, H, W] optional object masks per frame.
        config: detection parameters.
        sample_id: identifier for this sample.

    Returns:
        GraphLabel with ARTICULATION nodes at cluster boundaries.
    """
    T, N, _ = point_tracks.shape

    if T < config.min_frames:
        return GraphLabel(sample_id=sample_id, nodes=[], edges=[],
                          metadata={"source": "motion", "error": "insufficient_frames"})

    # Compute motion vectors for each point
    # Use displacement relative to mean motion (remove global translation)
    displacements = np.diff(point_tracks, axis=0)  # [T-1, N, 2]
    global_motion = displacements.mean(axis=1, keepdims=True)  # [T-1, 1, 2]
    relative_motion = displacements - global_motion  # [T-1, N, 2]

    # Motion feature per point: concatenated relative displacements
    # Flatten: [N, (T-1)*2]
    motion_features = relative_motion.transpose(1, 0, 2).reshape(N, -1)

    # Compute pairwise motion similarity
    affinity = _compute_motion_affinity(motion_features)

    # Cluster points
    labels = _cluster_points(affinity, config)

    # Find articulation candidates at cluster boundaries
    nodes, edges = _find_boundary_articulations(
        point_tracks, labels, config, sample_id
    )

    return GraphLabel(
        sample_id=sample_id,
        nodes=nodes,
        edges=edges,
        metadata={
            "source": "motion_articulation",
            "num_points": N,
            "num_frames": T,
            "num_clusters": int(labels.max() + 1) if len(labels) > 0 else 0,
        },
    )


def _compute_motion_affinity(motion_features: np.ndarray) -> np.ndarray:
    """
    Compute N×N affinity matrix from motion features.

    Points with similar motion patterns get high affinity.
    """
    N = len(motion_features)

    # Normalize features
    norms = np.linalg.norm(motion_features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = motion_features / norms

    # Cosine similarity
    similarity = normalized @ normalized.T  # [N, N]

    # Convert to affinity (ensure non-negative)
    affinity = (similarity + 1) / 2  # map [-1,1] → [0,1]

    return affinity


def _cluster_points(
    affinity: np.ndarray,
    config: MotionArticulationConfig,
) -> np.ndarray:
    """
    Cluster points based on motion affinity.

    Returns cluster labels [N].
    """
    N = affinity.shape[0]

    if N < config.min_cluster_size * 2:
        return np.zeros(N, dtype=np.int32)

    try:
        from sklearn.cluster import SpectralClustering, KMeans
    except ImportError:
        # Fallback: simple thresholding
        return _simple_cluster(affinity, config)

    # Determine optimal number of clusters
    n_clusters = min(config.max_clusters, N // config.min_cluster_size)
    n_clusters = max(2, n_clusters)

    if config.cluster_method == "spectral":
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42,
        )
        labels = clustering.fit_predict(affinity)
    else:
        # Embed with affinity eigenvectors then kmeans
        from scipy.linalg import eigh
        # Laplacian
        D = np.diag(affinity.sum(axis=1))
        L = D - affinity
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(affinity.sum(axis=1)) + 1e-8))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        eigenvalues, eigenvectors = eigh(L_norm)
        # Use first n_clusters eigenvectors
        embedding = eigenvectors[:, :n_clusters]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embedding)

    return labels.astype(np.int32)


def _simple_cluster(affinity: np.ndarray, config: MotionArticulationConfig) -> np.ndarray:
    """Fallback clustering without sklearn: connected components on thresholded affinity."""
    N = affinity.shape[0]
    threshold = 0.7
    labels = -np.ones(N, dtype=np.int32)
    current_label = 0

    for i in range(N):
        if labels[i] >= 0:
            continue
        # BFS
        queue = [i]
        labels[i] = current_label
        while queue:
            node = queue.pop(0)
            for j in range(N):
                if labels[j] < 0 and affinity[node, j] > threshold:
                    labels[j] = current_label
                    queue.append(j)
        current_label += 1

    # Assign any remaining to nearest cluster
    for i in range(N):
        if labels[i] < 0:
            labels[i] = 0

    return labels


def _find_boundary_articulations(
    point_tracks: np.ndarray,
    cluster_labels: np.ndarray,
    config: MotionArticulationConfig,
    sample_id: str,
) -> tuple[list[HyperNodeLabel], list[HyperEdgeLabel]]:
    """
    Find articulation nodes at boundaries between motion clusters.

    For each pair of adjacent clusters, the articulation is estimated
    at the boundary centroid.
    """
    T, N, _ = point_tracks.shape
    # Use mean position of each point over time
    mean_positions = point_tracks.mean(axis=0)  # [N, 2]

    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # Compute cluster centroids
    cluster_centroids = {}
    for c in unique_clusters:
        mask = cluster_labels == c
        cluster_centroids[c] = mean_positions[mask].mean(axis=0)

    nodes: list[HyperNodeLabel] = []
    edges: list[HyperEdgeLabel] = []
    node_id = 0
    edge_id = 0

    # For each pair of clusters, find boundary
    processed_pairs: set[tuple[int, int]] = set()

    for i, c1 in enumerate(unique_clusters):
        for c2 in unique_clusters[i + 1:]:
            pair = (int(min(c1, c2)), int(max(c1, c2)))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            # Find boundary points: points in c1 close to points in c2
            pts1 = mean_positions[cluster_labels == c1]
            pts2 = mean_positions[cluster_labels == c2]

            if len(pts1) == 0 or len(pts2) == 0:
                continue

            # Compute cross-distances
            dists = np.linalg.norm(pts1[:, None] - pts2[None, :], axis=2)
            min_dist = dists.min()

            # Only consider adjacent clusters
            median_intra1 = np.median(np.linalg.norm(pts1 - pts1.mean(0), axis=1))
            median_intra2 = np.median(np.linalg.norm(pts2 - pts2.mean(0), axis=1))
            adjacency_threshold = (median_intra1 + median_intra2)

            if min_dist > adjacency_threshold:
                continue

            # Boundary region: closest points between clusters
            close_mask1 = dists.min(axis=1) < adjacency_threshold * 0.5
            close_mask2 = dists.min(axis=0) < adjacency_threshold * 0.5

            boundary_pts1 = pts1[close_mask1]
            boundary_pts2 = pts2[close_mask2]

            if len(boundary_pts1) == 0 and len(boundary_pts2) == 0:
                continue

            # Articulation at midpoint of boundary
            all_boundary = np.concatenate([boundary_pts1, boundary_pts2], axis=0)
            articulation_xy = all_boundary.mean(axis=0)

            # Confidence from relative motion between clusters
            motion1 = point_tracks[:, cluster_labels == c1].mean(axis=1)  # [T, 2]
            motion2 = point_tracks[:, cluster_labels == c2].mean(axis=1)  # [T, 2]
            relative_disp = np.linalg.norm(np.diff(motion1 - motion2, axis=0), axis=1)
            motion_magnitude = relative_disp.mean()

            conf = min(0.95, motion_magnitude * config.boundary_confidence_scale)
            conf = max(0.3, conf)

            nodes.append(HyperNodeLabel(
                id=node_id,
                node_type=NodeType.ARTICULATION,
                xy=articulation_xy.tolist(),
                confidence=conf,
                label_sources={LabelSource.MOTION_ARTICULATION.value: conf},
                semantic=f"articulation_c{c1}_c{c2}",
                uncertainty_reason="motion_derived" if conf < 0.6 else None,
            ))
            node_id += 1

    # Connect articulations that share a cluster
    # (forms a chain of articulation nodes)
    if len(nodes) >= 2:
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Simple proximity-based edge
                xy_i = np.array(nodes[i].xy)
                xy_j = np.array(nodes[j].xy)
                dist = np.linalg.norm(xy_i - xy_j)

                # Connect if reasonably close
                overall_span = np.linalg.norm(mean_positions.max(0) - mean_positions.min(0))
                if dist < overall_span * 0.6:
                    edges.append(HyperEdgeLabel(
                        id=edge_id,
                        source_node_id=nodes[i].id,
                        target_node_id=nodes[j].id,
                        edge_type=EdgeType.DEFORMATION_LINK,
                        confidence=min(nodes[i].confidence, nodes[j].confidence) * 0.8,
                        label_sources={LabelSource.MOTION_ARTICULATION.value: 0.7},
                        length=float(dist),
                    ))
                    edge_id += 1

    return nodes, edges


def sample_points_in_mask(
    mask: np.ndarray,
    num_points: int = 200,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample random points inside a binary mask.

    Returns: [N, 2] array of (x, y) coordinates.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.zeros((0, 2))

    if len(ys) <= num_points:
        indices = np.arange(len(ys))
    else:
        indices = rng.choice(len(ys), size=num_points, replace=False)

    return np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)
