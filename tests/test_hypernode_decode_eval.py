"""Tests for HyperNodeNet decoder and evaluation."""
import pytest
import torch
import numpy as np

from hyperbone.labels.schema import (
    GraphLabel,
    HyperNodeLabel,
    HyperEdgeLabel,
    NodeType,
    EdgeType,
)
from hyperbone.hypernodes.decode import decode_predictions, DecodeConfig
from hyperbone.hypernodes.eval import compute_metrics, MetricsResult
from hyperbone.hypernodes.dataset import NUM_NODE_TYPES, NUM_EDGE_TYPES
from hyperbone.hypernodes.losses import HyperNodeLoss
from hyperbone.models.hypernode_net import HyperNodeNet
from hyperbone.hypernodes.dataset import HyperNodeDataset


def _make_pred_tensors(B=2, N=32, active_n=5):
    """Create mock prediction tensors with active nodes."""
    active_logits = torch.full((B, N), -5.0)
    active_logits[:, :active_n] = 3.0  # sigmoid(3) > 0.95

    node_xy = torch.rand(B, N, 2)
    # Spread nodes so they don't get NMS'd
    for b in range(B):
        for i in range(active_n):
            node_xy[b, i] = torch.tensor([0.1 + 0.2 * i, 0.5])

    return {
        "active_logits": active_logits,
        "node_xy": node_xy,
        "node_xyz": torch.randn(B, N, 3),
        "node_type_logits": torch.randn(B, N, NUM_NODE_TYPES),
        "node_confidence": torch.ones(B, N) * 0.8,
        "edge_logits": torch.full((B, N, N), -5.0),
        "edge_type_logits": torch.randn(B, N, N, NUM_EDGE_TYPES),
    }


class TestDecoder:
    def test_basic_decode(self):
        pred = _make_pred_tensors(B=2, active_n=5)
        # Set some edges active
        pred["edge_logits"][:, 0, 1] = 3.0
        pred["edge_logits"][:, 1, 2] = 3.0

        graphs = decode_predictions(pred, sample_ids=["s1", "s2"])
        assert len(graphs) == 2
        for g in graphs:
            assert g.metadata.get("valid", False)
            assert len(g.nodes) >= 2

    def test_variable_node_count(self):
        pred = _make_pred_tensors(B=2, active_n=3)
        # Make second batch have more active
        pred["active_logits"][1, 3:6] = 3.0
        for i in range(3, 6):
            pred["node_xy"][1, i] = torch.tensor([0.1 + 0.2 * i, 0.3])

        graphs = decode_predictions(pred)
        assert len(graphs[0].nodes) != len(graphs[1].nodes)

    def test_empty_prediction(self):
        pred = _make_pred_tensors(B=1, active_n=0)
        pred["active_logits"][:] = -10.0
        graphs = decode_predictions(pred)
        assert len(graphs) == 1
        assert not graphs[0].metadata.get("valid", True)

    def test_nms_dedup(self):
        pred = _make_pred_tensors(B=1, active_n=3)
        # Place nodes very close together
        pred["node_xy"][0, 0] = torch.tensor([0.5, 0.5])
        pred["node_xy"][0, 1] = torch.tensor([0.501, 0.501])
        pred["node_xy"][0, 2] = torch.tensor([0.9, 0.9])

        config = DecodeConfig(nms_radius=0.05)
        graphs = decode_predictions(pred, config)
        # Two of the first three are duplicates
        assert len(graphs[0].nodes) == 2

    def test_edge_threshold(self):
        pred = _make_pred_tensors(B=1, active_n=4)
        # All edges below threshold
        pred["edge_logits"][:] = -5.0
        graphs = decode_predictions(pred)
        assert len(graphs[0].edges) == 0


class TestEvalMetrics:
    def _make_gt_graph(self, n=5) -> GraphLabel:
        nodes = []
        for i in range(n):
            nodes.append(HyperNodeLabel(
                id=i,
                node_type=NodeType.ENDPOINT,
                xy=[0.1 + 0.2 * i, 0.5],
                confidence=1.0,
                label_sources={"test": 1.0},
            ))
        edges = [HyperEdgeLabel(
            id=i, source_node_id=i, target_node_id=i + 1,
            edge_type=EdgeType.BRANCH, confidence=1.0,
        ) for i in range(n - 1)]
        return GraphLabel(sample_id="gt", nodes=nodes, edges=edges,
                         metadata={"source": "procedural_branch"})

    def test_perfect_match(self):
        gt = self._make_gt_graph(5)
        metrics = compute_metrics([gt], [gt])
        assert metrics.node_f1 == 1.0
        assert metrics.node_precision == 1.0
        assert metrics.node_recall == 1.0

    def test_no_predictions(self):
        gt = self._make_gt_graph(5)
        empty = GraphLabel(sample_id="empty", metadata={"valid": False})
        metrics = compute_metrics([empty], [gt])
        assert metrics.node_f1 == 0.0
        assert metrics.invalid_graph_rate == 1.0

    def test_partial_match(self):
        gt = self._make_gt_graph(5)
        # Predict only 3 nodes at same positions
        pred_nodes = [HyperNodeLabel(
            id=i, node_type=NodeType.ENDPOINT,
            xy=[0.1 + 0.2 * i, 0.5], confidence=1.0, label_sources={},
        ) for i in range(3)]
        pred = GraphLabel(sample_id="pred", nodes=pred_nodes,
                         metadata={"valid": True})
        metrics = compute_metrics([pred], [gt])
        assert 0 < metrics.node_precision <= 1.0
        assert 0 < metrics.node_recall < 1.0
        assert 0 < metrics.node_f1 < 1.0

    def test_metrics_result_to_dict(self):
        r = MetricsResult(node_f1=0.5, edge_f1=0.3)
        d = r.to_dict()
        assert d["node_f1"] == 0.5
        assert "per_node_type_f1" in d


class TestOverfitSmoke:
    """Tiny overfit test: 4 samples, verify loss decreases."""

    def test_overfit_4_samples(self):
        # Create 4 simple graphs
        graphs = []
        for i in range(4):
            nodes = [
                HyperNodeLabel(id=0, node_type=NodeType.ENDPOINT,
                              xy=[0.2, 0.3], confidence=1.0, label_sources={"p": 1.0}),
                HyperNodeLabel(id=1, node_type=NodeType.BRANCH,
                              xy=[0.5, 0.5], confidence=1.0, label_sources={"p": 1.0}),
                HyperNodeLabel(id=2, node_type=NodeType.ENDPOINT,
                              xy=[0.8, 0.7], confidence=1.0, label_sources={"p": 1.0}),
            ]
            edges = [
                HyperEdgeLabel(id=0, source_node_id=0, target_node_id=1,
                              edge_type=EdgeType.BRANCH, confidence=1.0),
                HyperEdgeLabel(id=1, source_node_id=1, target_node_id=2,
                              edge_type=EdgeType.BRANCH, confidence=1.0),
            ]
            graphs.append(GraphLabel(
                sample_id=f"overfit_{i}",
                nodes=nodes, edges=edges,
                metadata={"source": "test"},
            ))

        ds = HyperNodeDataset(graphs_path="", resolution=64, max_nodes=16, graphs=graphs)

        model = HyperNodeNet(in_channels=1, base_channels=8, max_nodes=16)
        loss_fn = HyperNodeLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Collate function
        from scripts.train_hypernode_net_v0 import collate_fn
        from torch.utils.data import DataLoader

        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(loader))

        # Train 50 steps
        model.train()
        losses = []
        for step in range(50):
            images = batch["image"]
            targets = {
                "heatmaps": batch["heatmaps"],
                "radius_map": batch["radius_map"],
                "node_active": batch["node_active"],
                "node_xy": batch["node_xy"],
                "node_type": batch["node_type"],
                "edge_active": batch["edge_active"],
                "edge_type": batch["edge_type"],
            }
            pred = model(images)
            total, _ = loss_fn(pred, targets)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            losses.append(total.item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
