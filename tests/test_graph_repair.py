"""Tests for graph repair module."""

import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.cv.graph_repair import repair_graph, _find_components, _node_dist


def _make_disconnected_graph_close():
    """Two components with endpoints 5px apart — should be bridged."""
    return {
        "nodes": [
            {"id": 0, "type": "endpoint", "xy": [10, 50]},
            {"id": 1, "type": "junction", "xy": [30, 50]},
            {"id": 2, "type": "endpoint", "xy": [50, 50]},
            # Second component — 5px away from node 2
            {"id": 3, "type": "endpoint", "xy": [55, 50]},
            {"id": 4, "type": "junction", "xy": [70, 50]},
            {"id": 5, "type": "endpoint", "xy": [90, 50]},
        ],
        "edges": [
            {"parent": 0, "child": 1, "length_px": 20.0},
            {"parent": 1, "child": 2, "length_px": 20.0},
            {"parent": 3, "child": 4, "length_px": 15.0},
            {"parent": 4, "child": 5, "length_px": 20.0},
        ],
    }


def _make_disconnected_graph_far():
    """Two components with endpoints 100px apart — should NOT be bridged."""
    return {
        "nodes": [
            {"id": 0, "type": "endpoint", "xy": [10, 10]},
            {"id": 1, "type": "endpoint", "xy": [30, 10]},
            # Second component far away
            {"id": 2, "type": "endpoint", "xy": [200, 200]},
            {"id": 3, "type": "endpoint", "xy": [220, 200]},
        ],
        "edges": [
            {"parent": 0, "child": 1, "length_px": 20.0},
            {"parent": 2, "child": 3, "length_px": 20.0},
        ],
    }


def _make_graph_with_spur():
    """Graph with a short endpoint spur."""
    return {
        "nodes": [
            {"id": 0, "type": "endpoint", "xy": [10, 50]},
            {"id": 1, "type": "junction", "xy": [50, 50]},
            {"id": 2, "type": "endpoint", "xy": [90, 50]},
            # Short spur from junction
            {"id": 3, "type": "endpoint", "xy": [52, 53]},
        ],
        "edges": [
            {"parent": 0, "child": 1, "length_px": 40.0},
            {"parent": 1, "child": 2, "length_px": 40.0},
            {"parent": 1, "child": 3, "length_px": 4.0},  # short spur
        ],
    }


def test_bridge_close_components():
    """Two close components get bridged."""
    graph = _make_disconnected_graph_close()
    result = repair_graph(graph, bridge_gap_px=10.0)

    assert result["components_before"] == 2
    assert result["components_after"] == 1
    assert result["bridges_added"] >= 1
    print("  PASS: close components bridged")


def test_far_components_not_bridged():
    """Two far components are NOT bridged."""
    graph = _make_disconnected_graph_far()
    result = repair_graph(graph, bridge_gap_px=10.0)

    assert result["components_before"] == 2
    assert result["components_after"] == 2
    assert result["bridges_added"] == 0
    print("  PASS: far components not bridged")


def test_spur_pruned():
    """Short endpoint spur gets pruned (or merged if very close)."""
    graph = _make_graph_with_spur()
    result = repair_graph(graph, min_branch_length=6)

    repaired = result["graph"]
    # The 4px spur should be removed (via pruning or merge of close nodes)
    assert result["spurs_pruned"] >= 1 or result["nodes_merged"] >= 1
    # Original graph had 4 nodes; after removing spur, should have 3
    assert len(repaired["nodes"]) == 3
    print("  PASS: short spur pruned")


def test_junction_connector_not_pruned():
    """Junction-to-junction edge is never pruned even if short."""
    graph = {
        "nodes": [
            {"id": 0, "type": "junction", "xy": [10, 50]},
            {"id": 1, "type": "junction", "xy": [15, 50]},
            {"id": 2, "type": "endpoint", "xy": [50, 50]},
            {"id": 3, "type": "endpoint", "xy": [0, 50]},
        ],
        "edges": [
            {"parent": 0, "child": 1, "length_px": 5.0},  # short but junction-junction
            {"parent": 1, "child": 2, "length_px": 35.0},
            {"parent": 0, "child": 3, "length_px": 10.0},
        ],
    }
    result = repair_graph(graph, min_branch_length=8)
    repaired = result["graph"]

    # Junction-to-junction edge should survive
    # Both junctions should still be present
    junction_count = sum(1 for n in repaired["nodes"] if n["type"] == "junction")
    assert junction_count == 2, f"Both junctions should survive, got {junction_count}"
    print("  PASS: junction connector not pruned")


def test_repaired_has_fewer_components():
    """Repaired graph has fewer components than unrepaired."""
    graph = _make_disconnected_graph_close()
    components_before = len(_find_components(graph["nodes"], graph["edges"]))
    result = repair_graph(graph, bridge_gap_px=10.0)
    assert result["components_after"] < components_before
    print("  PASS: repair reduces component count")


def test_repair_diagnostics_complete():
    """Repair returns all required diagnostic fields."""
    graph = _make_disconnected_graph_close()
    result = repair_graph(graph)

    required = [
        "components_before", "components_after", "bridges_added",
        "nodes_merged", "spurs_pruned", "nodes_before", "nodes_after",
        "edges_before", "edges_after", "graph",
    ]
    for key in required:
        assert key in result, f"Missing key: {key}"
    print("  PASS: repair diagnostics complete")


def test_quality_improves_after_repair():
    """Quality score improves (or acceptance changes) after repair."""
    from hyperbone.quality.score import score_graph

    # Disconnected graph that should fail quality
    graph = _make_disconnected_graph_close()
    mask = np.ones((100, 100), dtype=np.uint8) * 255

    quality_before = score_graph(mask, graph, (100, 100), (0, 0, 100, 100))
    assert not quality_before["accepted"], "Should be rejected before repair"

    # Repair
    result = repair_graph(graph, bridge_gap_px=10.0)
    repaired = result["graph"]

    quality_after = score_graph(mask, repaired, (100, 100), (0, 0, 100, 100))
    # After bridging, it should at least not be rejected for disconnected
    disconnected_reasons_after = [r for r in quality_after["reject_reasons"] if "disconnected" in r]
    assert len(disconnected_reasons_after) == 0, "Should not be rejected for disconnected after repair"
    print("  PASS: quality improves after repair (no disconnected rejection)")


def test_single_component_graph_unchanged():
    """Already-connected graph passes through unchanged."""
    graph = {
        "nodes": [
            {"id": 0, "type": "endpoint", "xy": [10, 50]},
            {"id": 1, "type": "junction", "xy": [50, 50]},
            {"id": 2, "type": "endpoint", "xy": [90, 50]},
        ],
        "edges": [
            {"parent": 0, "child": 1, "length_px": 40.0},
            {"parent": 1, "child": 2, "length_px": 40.0},
        ],
    }
    result = repair_graph(graph, bridge_gap_px=10.0)
    assert result["components_before"] == 1
    assert result["components_after"] == 1
    assert result["bridges_added"] == 0
    print("  PASS: single component unchanged")


def run_all():
    print("[HyperBone Graph Repair Tests]")
    test_bridge_close_components()
    test_far_components_not_bridged()
    test_spur_pruned()
    test_junction_connector_not_pruned()
    test_repaired_has_fewer_components()
    test_repair_diagnostics_complete()
    test_quality_improves_after_repair()
    test_single_component_graph_unchanged()
    print(f"\nAll 8 tests passed.")


if __name__ == "__main__":
    run_all()
