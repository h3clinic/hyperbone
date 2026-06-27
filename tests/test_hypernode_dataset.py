"""Tests for HyperNodeNet dataset loader."""
import pytest
import torch
import numpy as np
from pathlib import Path

from hyperbone.labels.schema import (
    GraphLabel,
    HyperNodeLabel,
    HyperEdgeLabel,
    NodeType,
    EdgeType,
    save_graph_labels,
)
from hyperbone.hypernodes.dataset import (
    HyperNodeDataset,
    rasterize_graph,
    normalize_positions,
    split_train_val,
    NODE_TYPES,
    NUM_NODE_TYPES,
    NUM_EDGE_TYPES,
)


def _make_sample_graph(sample_id="test_001", num_nodes=5) -> GraphLabel:
    """Create a minimal test graph."""
    nodes = []
    for i in range(num_nodes):
        angle = 2 * 3.14159 * i / num_nodes
        nodes.append(HyperNodeLabel(
            id=i,
            node_type=NODE_TYPES[i % NUM_NODE_TYPES],
            xy=[0.3 + 0.2 * np.cos(angle), 0.3 + 0.2 * np.sin(angle)],
            xyz=[np.cos(angle), np.sin(angle), 0.0],
            confidence=0.9,
            radius=0.05,
            label_sources={"procedural": 0.9},
        ))
    edges = []
    for i in range(num_nodes - 1):
        edges.append(HyperEdgeLabel(
            id=i,
            source_node_id=i,
            target_node_id=i + 1,
            edge_type=EdgeType.BRANCH,
            confidence=0.8,
        ))
    return GraphLabel(
        sample_id=sample_id,
        nodes=nodes,
        edges=edges,
        metadata={"source": "procedural_branch"},
    )


def _make_graphs(n=10) -> list[GraphLabel]:
    return [_make_sample_graph(f"test_{i:03d}", num_nodes=3 + i % 5) for i in range(n)]


class TestRasterize:
    def test_basic(self):
        g = _make_sample_graph()
        img = rasterize_graph(g, 64)
        assert img.shape == (64, 64)
        assert img.max() > 0

    def test_empty_graph(self):
        g = GraphLabel(sample_id="empty")
        img = rasterize_graph(g, 64)
        assert img.shape == (64, 64)
        assert img.max() == 0

    def test_augment(self):
        g = _make_sample_graph()
        img = rasterize_graph(g, 64, augment=True)
        assert img.shape == (64, 64)


class TestDataset:
    def test_from_graphs(self):
        graphs = _make_graphs(5)
        ds = HyperNodeDataset(graphs_path="", resolution=64, max_nodes=32, graphs=graphs)
        assert len(ds) == 5

    def test_getitem_shapes(self):
        graphs = _make_graphs(3)
        ds = HyperNodeDataset(graphs_path="", resolution=64, max_nodes=32, graphs=graphs)
        sample = ds[0]
        assert sample["image"].shape == (1, 64, 64)
        assert sample["heatmaps"].shape == (NUM_NODE_TYPES, 64, 64)
        assert sample["radius_map"].shape == (1, 64, 64)
        assert sample["node_active"].shape == (32,)
        assert sample["node_xy"].shape == (32, 2)
        assert sample["node_xyz"].shape == (32, 3)
        assert sample["node_type"].shape == (32,)
        assert sample["node_confidence"].shape == (32,)
        assert sample["edge_active"].shape == (32, 32)
        assert sample["edge_type"].shape == (32, 32)

    def test_heatmap_has_peaks(self):
        graphs = _make_graphs(1)
        ds = HyperNodeDataset(graphs_path="", resolution=64, max_nodes=32, graphs=graphs)
        sample = ds[0]
        hm = sample["heatmaps"]
        assert hm.max() > 0.5  # Should have clear peaks

    def test_node_active(self):
        graphs = [_make_sample_graph(num_nodes=4)]
        ds = HyperNodeDataset(graphs_path="", resolution=64, max_nodes=32, graphs=graphs)
        sample = ds[0]
        assert sample["node_active"].sum() == 4

    def test_edge_active(self):
        graphs = [_make_sample_graph(num_nodes=4)]
        ds = HyperNodeDataset(graphs_path="", resolution=64, max_nodes=32, graphs=graphs)
        sample = ds[0]
        # 3 edges (symmetric), so 6 entries
        assert sample["edge_active"].sum() == 6

    def test_input_source_rasterized(self):
        graphs = _make_graphs(1)
        ds = HyperNodeDataset(graphs_path="", resolution=64, max_nodes=32, graphs=graphs)
        sample = ds[0]
        assert sample["input_source"] == "graph_rasterized"

    def test_loads_from_file(self, tmp_path):
        graphs = _make_graphs(5)
        path = tmp_path / "test.jsonl"
        save_graph_labels(graphs, path)
        ds = HyperNodeDataset(graphs_path=str(path), resolution=64, max_nodes=32)
        assert len(ds) == 5

    def test_loads_trainable_graphs(self):
        """Integration: loads actual trainable_graphs.jsonl if it exists."""
        path = Path("outputs/labelforge_v05/graphs/trainable_graphs.jsonl")
        if not path.exists():
            pytest.skip("trainable_graphs.jsonl not available")
        ds = HyperNodeDataset(graphs_path=str(path), resolution=64, max_nodes=64)
        assert len(ds) > 0
        sample = ds[0]
        assert sample["image"].shape[0] == 1


class TestSplit:
    def test_no_leak(self):
        graphs = _make_graphs(20)
        train, val = split_train_val(graphs, val_fraction=0.2, seed=42)
        train_ids = {g.sample_id for g in train}
        val_ids = {g.sample_id for g in val}
        assert len(train_ids & val_ids) == 0
        assert len(train) + len(val) == len(graphs)

    def test_deterministic(self):
        graphs = _make_graphs(20)
        t1, v1 = split_train_val(graphs, seed=42)
        t2, v2 = split_train_val(graphs, seed=42)
        assert [g.sample_id for g in t1] == [g.sample_id for g in t2]
