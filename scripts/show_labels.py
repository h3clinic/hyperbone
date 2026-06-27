import json, sys

with open("outputs/grounded_sam2_test/videos/HyperVid/quality.jsonl") as f:
    for line in f:
        d = json.loads(line)
        print(f"frame={d['frame_idx']:5d} obj={d['object_id']:2d} "
              f"class={d.get('object_class','?'):10s} "
              f"conf={d.get('label_confidence',0):.2f} "
              f"accepted={d['accepted']} "
              f"nodes={d['skeleton_node_count']:3d} "
              f"reasons={d['reject_reasons']}")
