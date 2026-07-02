"""HyperBone Track C3 — motion plausibility classifier (rules vs learned).

Classifies micro-motion clips as valid / one-of-6-corruptions using the C2
mobility features. Reports:
  * validity F1 (invalid = positive)   -- gate: >= 0.90
  * corruption_type macro F1 (6 classes) -- gate: >= 0.75
  * per-class recall, esp. the LENGTH-PRESERVING corruptions
    (impossible_large_rotation, swapped_limb_motion)
  * localization accuracy (culprit joint/edge vs the actually-affected joints)

Models: rules-only baseline (thresholds fit on train) vs learned MLP and
RandomForest. Asset-level split (no clip leakage).

Usage:
    python scripts/train_motion_plausibility.py \
        --features outputs/track_c_micro_motion_1000/features.npz \
        --dataset  outputs/track_c_micro_motion_1000 \
        --out outputs/track_c_micro_motion_1000
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hyperbone.motion.mobility import annotate_clip, FEATURE_NAMES

LEN_PRESERVING = ["impossible_large_rotation", "swapped_limb_motion"]


def fi(name):
    return FEATURE_NAMES.index(name)


def rules_predict(X, thr, classes):
    """Hand-crafted decision procedure over C2 features. Returns class idx."""
    out = np.zeros(len(X), dtype=np.int64)
    ci = {c: classes.index(c) for c in classes}
    for i, f in enumerate(X):
        bld = f[fi("bone_length_deviation")]
        mbld = f[fi("mean_bone_length_deviation")]
        maxang = f[fi("max_joint_angle_range")]
        sym = f[fi("motion_symmetry")]
        smooth = f[fi("temporal_smoothness")]
        cons = f[fi("parent_child_consistency")]
        movefrac = f[fi("moving_joint_fraction")]
        energy = f[fi("motion_energy")]

        if smooth < thr["smooth_lo"] or cons < thr["cons_lo"]:
            out[i] = ci["temporal_jitter"]
        elif bld > thr["len_hi"]:
            if mbld > thr["mbld_hi"]:
                out[i] = ci["bone_length_scale_error"]
            elif bld > thr["len_wrongparent"] and maxang > thr["ang_wrongparent"]:
                out[i] = ci["wrong_parent_motion"]
            else:
                out[i] = ci["detached_child"]
        elif maxang > thr["ang_hi"]:
            out[i] = ci["impossible_large_rotation"]
        elif sym < thr["sym_lo"] and movefrac > thr["movefrac_swap"] and energy > thr["energy_swap"]:
            out[i] = ci["swapped_limb_motion"]
        else:
            out[i] = ci["valid"]
    return out


def fit_rules_thresholds(Xtr, ytr, classes):
    ci = {c: classes.index(c) for c in classes}
    val = Xtr[ytr == ci["valid"]]
    swap = Xtr[ytr == ci["swapped_limb_motion"]]

    def p(arr, feat, q):
        return float(np.percentile(arr[:, fi(feat)], q)) if len(arr) else 0.0
    return {
        "len_hi": max(0.15, p(val, "bone_length_deviation", 99)),
        "mbld_hi": 0.12,
        "len_wrongparent": 1.6,
        "ang_wrongparent": 1.1,
        "smooth_lo": min(0.75, p(val, "temporal_smoothness", 2)),
        "cons_lo": min(0.85, p(val, "parent_child_consistency", 2)),
        "ang_hi": max(1.4, p(val, "max_joint_angle_range", 99)),
        # swapped: low symmetry + more than one limb moving + real energy
        "sym_lo": 0.20,
        "movefrac_swap": max(0.06, p(swap, "moving_joint_fraction", 25) * 0.6),
        "energy_swap": max(0.05, p(swap, "motion_energy", 25) * 0.5),
    }


def metrics(y_true, y_pred, classes):
    ci = {c: classes.index(c) for c in classes}
    valid_idx = ci["valid"]
    tv = (y_true != valid_idx).astype(int)   # invalid = positive
    pv = (y_pred != valid_idx).astype(int)
    validity_f1 = f1_score(tv, pv)
    corr_ids = [ci[c] for c in classes if c != "valid"]
    macro = f1_score(y_true, y_pred, labels=corr_ids, average="macro", zero_division=0)
    per = {}
    for c in classes:
        idx = ci[c]
        per[c] = f1_score((y_true == idx).astype(int), (y_pred == idx).astype(int),
                          zero_division=0)
    rec = {}
    for c in LEN_PRESERVING:
        idx = ci[c]
        mask = y_true == idx
        rec[c] = float((y_pred[mask] == idx).mean()) if mask.any() else 0.0
    return {"validity_f1": float(validity_f1),
            "corruption_macro_f1": float(macro),
            "per_class_f1": per,
            "length_preserving_recall": rec}


def localization_eval(dataset, test_assets, classes):
    """For length-based and impossible corruptions, does the flagged culprit
    joint fall in the actually-affected joint set (moving_joint_ids)?"""
    paths = sorted(glob.glob(str(Path(dataset) / "clips" / "*.npz")))
    hit = tot = 0
    for cp in paths:
        z = np.load(cp, allow_pickle=True)
        if int(z["asset_idx"]) not in test_assets:
            continue
        if bool(z["is_valid_motion"]):
            continue
        ctype = str(z["corruption_type"])
        world = z["joints_world"].astype(np.float64); edges = z["edges"].astype(np.int64)
        parents = z["parents"].astype(np.int64); rest = z["joints_rest"].astype(np.float64)
        mv = set(int(x) for x in z["moving_joint_ids"].tolist())
        if not mv or edges.shape[0] == 0:
            continue
        ann = annotate_clip(world, edges, parents, rest, int(z["fps"]))
        loc = ann["localization"]
        if ctype in ("bone_length_scale_error", "detached_child", "wrong_parent_motion",
                     "temporal_jitter"):
            e = loc["max_len_dev_edge"]
            culprit = int(edges[e, 1]) if e >= 0 else -1
        else:  # impossible/swapped -> angle
            culprit = loc["max_angle_joint"]
        tot += 1
        # allow culprit or its parent/children to be in affected set
        neigh = {culprit}
        if 0 <= culprit < len(parents):
            neigh.add(int(parents[culprit]))
        if culprit in mv or neigh & mv:
            hit += 1
    return hit / tot if tot else 0.0, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X = d["X"].astype(np.float64)
    yc = d["y_class"].astype(np.int64)
    assets = d["asset_idx"].astype(np.int64)
    classes = [str(c) for c in d["classes"]]
    print(f"Loaded {len(X)} clips, {len(set(assets))} assets, {len(classes)} classes",
          flush=True)

    # asset-level split (no clip leakage)
    uniq = np.array(sorted(set(assets.tolist())))
    rng = np.random.default_rng(args.seed); rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * args.test_frac)))
    test_assets = set(uniq[:n_test].tolist())
    tr = np.array([a not in test_assets for a in assets])
    te = ~tr
    Xtr, Xte, ytr, yte = X[tr], X[te], yc[tr], yc[te]
    print(f"train clips {tr.sum()} ({len(uniq)-n_test} assets)  "
          f"test clips {te.sum()} ({n_test} assets)", flush=True)

    results = {}

    # ---- rules baseline ----
    thr = fit_rules_thresholds(Xtr, ytr, classes)
    ypred_rules = rules_predict(Xte, thr, classes)
    results["rules"] = metrics(yte, ypred_rules, classes)
    results["rules"]["thresholds"] = {k: float(v) for k, v in thr.items()}

    # ---- learned: MLP + RandomForest ----
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                        random_state=args.seed).fit(Xtr_s, ytr)
    results["mlp"] = metrics(yte, mlp.predict(Xte_s), classes)
    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed).fit(Xtr, ytr)
    ypred_rf = rf.predict(Xte)
    results["random_forest"] = metrics(yte, ypred_rf, classes)

    # ---- localization (uses annotate on test clips) ----
    loc_acc, loc_n = localization_eval(args.dataset, test_assets, classes)
    results["localization"] = {"culprit_in_affected_acc": loc_acc, "n": loc_n}

    # ---- print ----
    print("\n================ RESULTS (held-out assets) ================", flush=True)
    for m in ["rules", "mlp", "random_forest"]:
        r = results[m]
        print(f"\n[{m}]  validity_F1={r['validity_f1']:.3f}  "
              f"corruption_macroF1={r['corruption_macro_f1']:.3f}", flush=True)
        print("  per-class F1: " + "  ".join(
            f"{c[:14]}={r['per_class_f1'][c]:.2f}" for c in classes), flush=True)
        print("  length-preserving recall: " + "  ".join(
            f"{c}={r['length_preserving_recall'][c]:.2f}" for c in LEN_PRESERVING), flush=True)
    print(f"\nlocalization (culprit in affected set): "
          f"{loc_acc:.3f}  (n={loc_n})", flush=True)

    # ---- gate verdict (best learned model) ----
    best = max(["mlp", "random_forest"], key=lambda m: results[m]["corruption_macro_f1"])
    r = results[best]
    gate = {
        "best_learned_model": best,
        "validity_f1_ge_0.90": r["validity_f1"] >= 0.90,
        "corruption_macro_f1_ge_0.75": r["corruption_macro_f1"] >= 0.75,
        "impossible_rotation_detected": r["length_preserving_recall"]["impossible_large_rotation"] >= 0.5,
        "swapped_limb_detected": r["length_preserving_recall"]["swapped_limb_motion"] >= 0.5,
    }
    gate["PASS"] = all(gate[k] for k in gate if k != "best_learned_model")
    results["gate"] = gate
    results["classes"] = classes
    results["split"] = {"train_assets": len(uniq) - n_test, "test_assets": n_test,
                        "train_clips": int(tr.sum()), "test_clips": int(te.sum())}
    print(f"\n== GATE ({best}) ==", flush=True)
    for k, v in gate.items():
        if k != "best_learned_model":
            print(f"  [{'PASS' if v else 'FAIL'}] {k}", flush=True)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "plausibility_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport -> {out_dir/'plausibility_report.json'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
