"""Tests for HyperNodeNet losses."""
import pytest
import torch

from hyperbone.hypernodes.losses import (
    HyperNodeLoss,
    LossWeights,
    focal_mse_loss,
    chamfer_loss_2d,
)
from hyperbone.hypernodes.dataset import NUM_NODE_TYPES, NUM_EDGE_TYPES


def _make_batch(B=2, N=32, T=NUM_NODE_TYPES, E=NUM_EDGE_TYPES, res=64):
    """Create a mock batch."""
    return {
        "heatmaps": torch.rand(B, T, res, res),
        "radius_map": torch.rand(B, 1, res, res),
        "node_active": torch.zeros(B, N).float(),
        "node_xy": torch.rand(B, N, 2),
        "node_type": torch.randint(0, T, (B, N)),
        "edge_active": torch.zeros(B, N, N).float(),
        "edge_type": torch.zeros(B, N, N, dtype=torch.long),
    }


def _make_pred(B=2, N=32, T=NUM_NODE_TYPES, E=NUM_EDGE_TYPES, res=64):
    """Create mock predictions."""
    return {
        "heatmaps": torch.rand(B, T, res, res),
        "radius_map": torch.rand(B, 1, res, res),
        "active_logits": torch.randn(B, N),
        "node_xy": torch.rand(B, N, 2),
        "node_xyz": torch.randn(B, N, 3),
        "node_type_logits": torch.randn(B, N, T),
        "node_confidence": torch.rand(B, N),
        "edge_logits": torch.randn(B, N, N),
        "edge_type_logits": torch.randn(B, N, N, E),
    }


class TestFocalMSE:
    def test_zero_target(self):
        pred = torch.rand(2, 8, 16, 16)
        target = torch.zeros(2, 8, 16, 16)
        loss = focal_mse_loss(pred, target)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_perfect_match(self):
        x = torch.rand(2, 8, 16, 16)
        loss = focal_mse_loss(x, x)
        assert loss.item() == 0.0


class TestChamferLoss:
    def test_basic(self):
        pred_xy = torch.rand(2, 32, 2)
        pred_active = torch.zeros(2, 32)
        pred_active[:, :5] = 1.0
        target_xy = torch.rand(2, 32, 2)
        target_active = torch.zeros(2, 32)
        target_active[:, :5] = 1.0
        loss = chamfer_loss_2d(pred_xy, pred_active, target_xy, target_active)
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_same_points(self):
        xy = torch.rand(2, 32, 2)
        active = torch.zeros(2, 32)
        active[:, :5] = 1.0
        loss = chamfer_loss_2d(xy, active, xy, active)
        assert loss.item() < 0.001

    def test_no_active(self):
        pred_xy = torch.rand(2, 32, 2)
        pred_active = torch.zeros(2, 32)
        target_active = torch.zeros(2, 32)
        loss = chamfer_loss_2d(pred_xy, pred_active, pred_xy, target_active)
        assert loss.item() == 0.0


class TestHyperNodeLoss:
    def test_computes_without_nan(self):
        batch = _make_batch()
        # Set some nodes as active
        batch["node_active"][:, :5] = 1.0
        batch["edge_active"][:, 0, 1] = 1.0
        batch["edge_active"][:, 1, 0] = 1.0

        pred = _make_pred()
        loss_fn = HyperNodeLoss()
        total, loss_dict = loss_fn(pred, batch)
        assert not torch.isnan(total)
        assert total.item() > 0
        assert "total" in loss_dict
        assert "heatmap" in loss_dict
        assert "node_active" in loss_dict
        assert "node_xy" in loss_dict
        assert "node_type" in loss_dict
        assert "edge_active" in loss_dict
        assert "edge_type" in loss_dict
        assert "chamfer" in loss_dict
        assert "radius" in loss_dict

    def test_no_active_nodes(self):
        """Should not crash when no nodes are active."""
        batch = _make_batch()
        pred = _make_pred()
        loss_fn = HyperNodeLoss()
        total, loss_dict = loss_fn(pred, batch)
        assert not torch.isnan(total)

    def test_backward(self):
        """Loss should support backprop."""
        batch = _make_batch()
        batch["node_active"][:, :3] = 1.0
        pred = _make_pred()
        # Make pred require grad
        for v in pred.values():
            v.requires_grad_(True)
        loss_fn = HyperNodeLoss()
        total, _ = loss_fn(pred, batch)
        total.backward()

    def test_custom_weights(self):
        batch = _make_batch()
        batch["node_active"][:, :3] = 1.0
        pred = _make_pred()
        w = LossWeights(heatmap=0.0, node_active=0.0, node_xy=0.0,
                        node_type=0.0, edge_active=0.0, edge_type=0.0,
                        radius=0.0, chamfer=0.0)
        loss_fn = HyperNodeLoss(w)
        total, _ = loss_fn(pred, batch)
        assert total.item() == 0.0
