"""Diagnose which step in the M2.2 pipeline hangs."""
import sys, time
sys.path.insert(0, ".")

import numpy as np

print("[DIAG] Step 1: Import SAM2 adapter...", flush=True)
t0 = time.time()
from hyperbone.cv.masks import get_mask_generator
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

print("[DIAG] Step 2: Load SAM2 model...", flush=True)
t0 = time.time()
mask_gen = get_mask_generator(
    "sam2",
    checkpoint="checkpoints/sam2.1_hiera_tiny.pt",
    model_cfg=r"C:\Users\ritayan\miniconda3\Lib\site-packages\sam2\configs\sam2.1\sam2.1_hiera_t.yaml",
    device="cuda",
)
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

print("[DIAG] Step 3: Load a frame...", flush=True)
t0 = time.time()
from hyperbone.io.video import sample_frames
frame = None
for frame_idx, ts, f in sample_frames("HyperVid/HyperVid.mp4", 1.0):
    frame = f
    break
print(f"  Frame shape: {frame.shape}, done in {time.time()-t0:.1f}s", flush=True)

print("[DIAG] Step 4: Generate SAM2 masks...", flush=True)
t0 = time.time()
mask_records = mask_gen.generate(frame)
print(f"  Got {len(mask_records)} masks in {time.time()-t0:.1f}s", flush=True)

print("[DIAG] Step 5: Process first mask through cleanup...", flush=True)
from hyperbone.cv.mask_cleanup import clean_mask_for_skeleton
from hyperbone.cv.skeletonize import skeletonize_mask
from hyperbone.cv.graph import skeleton_to_graph
from hyperbone.cv.graph_repair import repair_graph

for i, rec in enumerate(mask_records[:3]):
    mask = rec["mask"]
    mask_area = np.count_nonzero(mask)
    print(f"\n  --- Mask {i}: area={mask_area} ({mask_area/(frame.shape[0]*frame.shape[1])*100:.2f}%) ---", flush=True)

    t0 = time.time()
    cleanup_result = clean_mask_for_skeleton(mask, close_kernel=5, keep_largest=True)
    skel_mask = cleanup_result["clean_mask"]
    clean_area = np.count_nonzero(skel_mask)
    print(f"  5a. Cleanup: {time.time()-t0:.2f}s (area {mask_area}->{clean_area})", flush=True)

    t0 = time.time()
    skeleton = skeletonize_mask(skel_mask)
    skel_pixels = np.count_nonzero(skeleton)
    print(f"  5b. Skeletonize: {time.time()-t0:.2f}s ({skel_pixels} skeleton pixels)", flush=True)

    t0 = time.time()
    graph = skeleton_to_graph(skeleton, min_branch_length=10)
    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])
    print(f"  5c. Graph extract: {time.time()-t0:.2f}s ({n_nodes} nodes, {n_edges} edges)", flush=True)

    if not graph["nodes"]:
        print("  (empty graph, skipping repair)", flush=True)
        continue

    t0 = time.time()
    repair_result = repair_graph(graph, mask_shape=frame.shape[:2], bridge_gap_px=8.0, min_branch_length=10)
    repaired = repair_result["graph"]
    print(f"  5d. Graph repair: {time.time()-t0:.2f}s "
          f"(components {repair_result['components_before']}->{repair_result['components_after']}, "
          f"bridges={repair_result['bridges_added']})", flush=True)

print("\n[DIAG] DONE — no hang detected.", flush=True)
