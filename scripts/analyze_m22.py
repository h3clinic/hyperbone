"""Quick analysis of M2.2 repair results."""
import json, sys

with open("outputs/sam2_smoke_m22_repair/videos/HyperVid/quality.jsonl") as f:
    lines = [json.loads(l) for l in f]

print("=== REPAIR DIAGNOSTICS ===")
comp_before = [l.get("components_before_repair", 0) for l in lines if l.get("graph_repair_applied")]
comp_after = [l.get("components_after_repair", 0) for l in lines if l.get("graph_repair_applied")]
bridges = [l.get("bridges_added", 0) for l in lines if l.get("graph_repair_applied")]

print(f"Objects processed with repair: {len(comp_before)}")
print(f"Avg components BEFORE repair: {sum(comp_before)/max(len(comp_before),1):.1f}")
print(f"Avg components AFTER repair:  {sum(comp_after)/max(len(comp_after),1):.1f}")
print(f"Total bridges added: {sum(bridges)}")
print(f"Avg bridges per object: {sum(bridges)/max(len(bridges),1):.1f}")

print("\n=== ACCEPTED (4) ===")
for l in lines:
    if l["accepted"]:
        print(f"  frame={l['frame_idx']:3d} obj={l['object_id']} "
              f"nodes={l['skeleton_node_count']:3d} edges={l['skeleton_edge_count']:3d} "
              f"comp_before={l.get('components_before_repair','?')} "
              f"comp_after={l.get('components_after_repair','?')} "
              f"bridges={l.get('bridges_added','?')}")

print("\n=== REJECTED TOP REASONS ===")
reasons = {}
for l in lines:
    if not l["accepted"]:
        for r in l["reject_reasons"]:
            reasons[r] = reasons.get(r, 0) + 1
for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f"  {c:2d}x {r}")
