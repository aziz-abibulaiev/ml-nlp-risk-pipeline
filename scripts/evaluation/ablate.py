#!/usr/bin/env python3
"""
Fusion-weight ablation on a finished mixed run.

For each alpha, thresholds are re-selected on the validation run and applied to the
stored outputs of the mixed run. Writes runs/<mixed>/ablations.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_thresholds as st  # noqa: E402

ALPHAS = {"ml_only (alpha=1.0)": 1.0, "alpha=0.6": 0.6, "nlp_only (alpha=0.0)": 0.0}


def load(run):
    evs = [json.loads(l) for l in open(ROOT / "runs" / run / "events.jsonl")]
    return [e for e in evs if e.get("http_status") == 200 and "fusion" in (e.get("response") or {})]


def set_alpha(evs, alpha):
    for e in evs:
        f = e["response"]["fusion"]
        f["final_score"] = alpha * float(f["agg_ml"]) + (1 - alpha) * float(f["agg_nlp"])


def select(fusion, ben, att, evs, fpr_t=0.01, rec_t=0.99):
    finals = sorted({round(float(e["response"]["fusion"]["final_score"]), 6) for e in evs} | {0.0, 1.0})
    if st.rate(fusion, ben, 0.0, st.fix_levels(finals[-1]), st.MEDIUM_PLUS) > fpr_t:
        l_low, feasible = 1.0, False
    else:
        feasible, lo, hi = True, 0, len(finals) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if st.rate(fusion, ben, 0.0, st.fix_levels(finals[mid]), st.MEDIUM_PLUS) <= fpr_t:
                hi = mid
            else:
                lo = mid + 1
        l_low = finals[lo]
    levels = st.fix_levels(l_low)
    cand = [c for c in finals if c <= l_low]
    theta = 0.0
    if st.rate(fusion, att, cand[0], levels, st.LOW_PLUS) >= rec_t:
        lo, hi = 0, len(cand) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if st.rate(fusion, att, cand[mid], levels, st.LOW_PLUS) >= rec_t:
                lo = mid
            else:
                hi = mid - 1
        theta = cand[lo]
    return theta, levels, feasible


def confusion(fusion, evs, theta, levels):
    out = {}
    for name, acc in (("Medium+", st.MEDIUM_PLUS), ("Low+", st.LOW_PLUS)):
        tp = fp = fn = tn = 0
        for e in evs:
            p = st.decide(fusion, e, theta, levels) in acc
            if e["label"] == 1:
                tp += p; fn += (not p)
            else:
                fp += p; tn += (not p)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        out[name] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "Precision": prec, "Recall": rec,
                     "F1": 2 * prec * rec / (prec + rec) if prec + rec else float("nan"),
                     "FPR": fp / (fp + tn) if fp + tn else float("nan")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--mixed", required=True)
    a = ap.parse_args()
    fusion = st.load_fusion()
    res = {"note": "thresholds re-selected on validation per alpha", "variants": {}}
    for name, alpha in ALPHAS.items():
        val, mix = load(a.val), load(a.mixed)
        set_alpha(val, alpha); set_alpha(mix, alpha)
        ben = [e for e in val if e["label"] == 0]; att = [e for e in val if e["label"] == 1]
        theta, levels, feasible = select(fusion, ben, att, val)
        res["variants"][name] = {"alpha": alpha, "FUSION_NORMAL_THR": theta, "FUSION_LEVELS": levels,
                                 "medium_boundary_feasible": feasible,
                                 "validation": confusion(fusion, val, theta, levels),
                                 "mixed": confusion(fusion, mix, theta, levels)}
        print(name, json.dumps(res["variants"][name]["mixed"]), flush=True)
    (ROOT / "runs" / a.mixed / "ablations.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
