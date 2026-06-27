"""Quick analysis of DINO smoke results."""
import json
from collections import Counter

with open("outputs/dino_custom_smoke/graphs/graphs.jsonl") as f:
    records = [json.loads(l) for l in f if l.strip()]

accepted = [r for r in records if r["accepted"]]
rejected = [r for r in records if not r["accepted"]]

print("=== DINO SMOKE RESULTS ===")
print(f"Total: {len(records)}, Accepted: {len(accepted)}, Rejected: {len(rejected)}")
print(f"Acceptance rate: {len(accepted)/len(records)*100:.1f}%")
print()

labels_a = Counter(r["object_label"] for r in accepted)
labels_r = Counter(r["object_label"] for r in rejected)

print("Accepted by label:")
for l, c in labels_a.most_common(15):
    print(f"  {l}: {c}")

print("\nRejected by label:")
for l, c in labels_r.most_common(10):
    print(f"  {l}: {c}")

print(f"\nUnique labels: {len(set(r['object_label'] for r in records))}")

# Example accepted
if accepted:
    ex = accepted[0]
    print(f"\nExample accepted: label={ex['object_label']} conf={ex['object_label_confidence']:.2f} nodes={ex['node_count']} edges={ex['edge_count']} runtime={ex['runtime_ms']:.1f}ms")

# Example rejected
if rejected:
    ex = rejected[0]
    print(f"Example rejected: label={ex['object_label']} conf={ex['object_label_confidence']:.2f} nodes={ex['node_count']} reasons={ex['reject_reasons']}")

# Runtime stats
import numpy as np
runtimes = [r["runtime_ms"] for r in records]
print(f"\nRuntime: avg={np.mean(runtimes):.1f}ms, p50={np.median(runtimes):.1f}ms, max={np.max(runtimes):.1f}ms")
