"""Tests for HyperNodeNet model."""
import pytest
import torch

from hyperbone.models.hypernode_net import HyperNodeNet
from hyperbone.hypernodes.dataset import NUM_NODE_TYPES, NUM_EDGE_TYPES


class TestHyperNodeNet:
    @pytest.fixture
    def model(self):
        return HyperNodeNet(
            in_channels=1,
            base_channels=16,
            max_nodes=32,
            num_node_types=NUM_NODE_TYPES,
            num_edge_types=NUM_EDGE_TYPES,
        )

    def test_forward_shapes(self, model):
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out["heatmaps"].shape == (2, NUM_NODE_TYPES, 64, 64)
        assert out["radius_map"].shape == (2, 1, 64, 64)
        assert out["active_logits"].shape == (2, 32)
        assert out["node_xy"].shape == (2, 32, 2)
        assert out["node_xyz"].shape == (2, 32, 3)
        assert out["node_type_logits"].shape == (2, 32, NUM_NODE_TYPES)
        assert out["node_confidence"].shape == (2, 32)
        assert out["edge_logits"].shape == (2, 32, 32)
        assert out["edge_type_logits"].shape == (2, 32, 32, NUM_EDGE_TYPES)

    def test_forward_192(self):
        model = HyperNodeNet(in_channels=1, base_channels=16, max_nodes=64)
        x = torch.randn(1, 1, 192, 192)
        out = model(x)
        assert out["heatmaps"].shape == (1, NUM_NODE_TYPES, 192, 192)

    def test_node_xy_bounded(self, model):
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        # node_xy uses sigmoid, should be [0, 1]
        assert out["node_xy"].min() >= 0.0
        assert out["node_xy"].max() <= 1.0

    def test_node_confidence_bounded(self, model):
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out["node_confidence"].min() >= 0.0
        assert out["node_confidence"].max() <= 1.0

    def test_param_count(self, model):
        count = model.param_count()
        assert count > 0
        # Small model should be < 10M params
        assert count < 10_000_000

    def test_gradient_flow(self, model):
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        loss = out["heatmaps"].mean() + out["active_logits"].mean()
        loss.backward()
        # Check gradients exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_different_resolutions(self, model):
        """Model should handle any resolution divisible by 16."""
        for res in [64, 128, 192]:
            x = torch.randn(1, 1, res, res)
            out = model(x)
            assert out["heatmaps"].shape == (1, NUM_NODE_TYPES, res, res)
