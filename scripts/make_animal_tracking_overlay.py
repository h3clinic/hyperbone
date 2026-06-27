"""Create animal tracking overlay video.

Composites skeleton graphs, bboxes, centroid trail on each frame.
"""

import sys, json, argparse
import numpy as np
import cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbone.io.video import get_video_info, sample_frames


def load_graph_records(path: str) -> list:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def draw_animal_overlay(frame: np.ndarray, record: dict, centroid_trail: list) -> np.ndarray:
    """Draw skeleton overlay on a frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    bbox = record.get("bbox_xywh", [0, 0, 0, 0])
    accepted = record.get("accepted", False)
    nodes = record.get("nodes", [])
    edges = record.get("edges", [])
    node_count = record.get("node_count", len(nodes))
    edge_count = record.get("edge_count", len(edges))
    frame_idx = record.get("frame_idx", 0)
    label = record.get("object_label", "?")
    reject_reasons = record.get("reject_reasons", [])

    color = (0, 255, 0) if accepted else (0, 0, 255)

    # Bbox
    x, y, bw, bh = bbox
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)

    # Edges
    node_map = {}
    for node in nodes:
        nid = node.get("id", id(node))
        nx, ny = int(node["xy"][0]), int(node["xy"][1])
        node_map[nid] = (nx, ny)

    for edge in edges:
        p = node_map.get(edge.get("parent"))
        c = node_map.get(edge.get("child"))
        if p and c:
            cv2.line(vis, p, c, (255, 128, 0), 1, cv2.LINE_AA)

    # Nodes
    for nid, (nx, ny) in node_map.items():
        cv2.circle(vis, (nx, ny), 2, (0, 255, 255), -1)

    # Centroid trail
    if len(centroid_trail) > 1:
        for i in range(1, len(centroid_trail)):
            pt1 = (int(centroid_trail[i - 1][0]), int(centroid_trail[i - 1][1]))
            pt2 = (int(centroid_trail[i][0]), int(centroid_trail[i][1]))
            alpha = i / len(centroid_trail)
            cv2.line(vis, pt1, pt2, (int(255 * alpha), int(200 * alpha), 0), 2, cv2.LINE_AA)

    # Text
    status = "ACCEPTED" if accepted else "REJECTED"
    cv2.putText(vis, f"Frame {frame_idx} | {label} | {status}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(vis, f"Nodes: {node_count} | Edges: {edge_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if not accepted and reject_reasons:
        cv2.putText(vis, f"Reason: {reject_reasons[0][:50]}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    return vis


def make_animal_tracking_overlay(
    video_path: str,
    graphs_path: str,
    output_path: str,
    sample_fps: float = 5.0,
):
    """Create overlay video from graph records."""
    info = get_video_info(video_path)
    records = load_graph_records(graphs_path)
    print(f"[Overlay] Video: {info['path']}")
    print(f"[Overlay] Graphs: {len(records)} records")

    records_by_frame = {}
    for r in records:
        fidx = r.get("frame_idx", -1)
        if fidx not in records_by_frame:
            records_by_frame[fidx] = []
        records_by_frame[fidx].append(r)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    centroid_trail = []

    for frame_idx, ts, frame in sample_frames(video_path, sample_fps):
        h, w = frame.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), fourcc, sample_fps, (w, h))

        frame_records = records_by_frame.get(frame_idx, [])

        if frame_records:
            record = frame_records[0]
            nodes = record.get("nodes", [])
            if nodes:
                xs = [n["xy"][0] for n in nodes]
                ys = [n["xy"][1] for n in nodes]
                centroid_trail.append((np.mean(xs), np.mean(ys)))
            else:
                bbox = record.get("bbox_xywh", [0, 0, w, h])
                centroid_trail.append((bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2))
            centroid_trail = centroid_trail[-30:]
            vis = draw_animal_overlay(frame, record, centroid_trail)
        else:
            vis = frame.copy()
            cv2.putText(vis, f"Frame {frame_idx} | NO GRAPH", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        writer.write(vis)

    if writer:
        writer.release()
    print(f"[Overlay] Written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Create animal tracking overlay video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    args = parser.parse_args()

    make_animal_tracking_overlay(args.video, args.graphs, args.out, args.sample_fps)


if __name__ == "__main__":
    main()
