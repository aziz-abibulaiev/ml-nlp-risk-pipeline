#!/usr/bin/env python3
"""
Select thresholds and fusion weights on the validation run.

Decisions use the functions of the fusion service, so the script runs in the fusion image.
Writes runs/<run>/frozen_config.json and frozen.env.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V001_LEVELS = [0.3, 0.5, 0.6, 0.8]
MEDIUM_PLUS = {"medium", "high", "critical"}
LOW_PLUS = {"low", "medium", "high", "critical"}


def load_fusion():
    sys.path.insert(0, "/app")
    import main as fusion  # the fusion service module baked into the image
    return fusion


def decide(fusion, ev: dict, normal_thr: float, levels: list[float]) -> str:
    fusion.NORMAL_THR = normal_thr
    fusion.L_LOW, fusion.L_MED, fusion.L_HIGH = levels[0], levels[1], levels[2]
    fus = ev["response"]["fusion"]
    ctx = fus.get("context") or {}
    topics = ctx.get("topics") or []
    level = fusion.risk_level_from_matrix(float(fus["final_score"]), ctx, topics)
    min_level, _, _ = fusion._topic_overrides(topics) if topics else (None, None, None)
    order = ["normal", "low", "medium", "high", "critical"]
    if min_level and order.index(min_level) > order.index(level):
        level = min_level
    return level


def fix_levels(l_low: float) -> list[float]:
    out = [l_low]
    for v in V001_LEVELS[1:]:
        out.append(v if v > out[-1] else min(1.0, round(out[-1] + 0.01, 6)))
    return out


def rate(fusion, evs, normal_thr, levels, accept) -> float:
    return sum(decide(fusion, e, normal_thr, levels) in accept for e in evs) / max(1, len(evs))


def roc_auc(y: list[int], s: list[float]) -> float:
    """Mann-Whitney ROC-AUC with average ranks for ties (no numpy in the fusion image)."""
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = sum(y); n0 = len(y) - n1
    r1 = sum(r for r, t in zip(ranks, y) if t == 1)
    return (r1 - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def select_alpha(evs: list[dict]) -> tuple[float, dict]:
    """Alpha with the highest ROC-AUC of the fused score, beta = 1 - alpha. Ties go to the value closer to 0.5."""
    y = [e["label"] for e in evs]
    ml = [float(e["response"]["fusion"]["agg_ml"]) for e in evs]
    nl = [float(e["response"]["fusion"]["agg_nlp"]) for e in evs]
    aucs = {}
    for k in range(11):
        al = k / 10
        aucs[al] = roc_auc(y, [al * m + (1 - al) * n for m, n in zip(ml, nl)])
    best = max(aucs, key=lambda al: (round(aucs[al], 12), -abs(al - 0.5)))
    for e, m, n in zip(evs, ml, nl):
        e["response"]["fusion"]["final_score"] = best * m + (1 - best) * n
    return best, aucs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--recall-target", type=float, default=0.99)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--impact-policy-host", default="./policies/impact_policy.json")
    a = ap.parse_args()
    run = ROOT / "runs" / a.run
    evs = [json.loads(l) for l in open(run / "events.jsonl")]
    bad = [e for e in evs if e.get("http_status") != 200 or "fusion" not in (e.get("response") or {})]
    if bad:
        raise SystemExit(f"{len(bad)} events failed; fix the stack before selecting thresholds")
    ben = [e for e in evs if e["label"] == 0]
    att = [e for e in evs if e["label"] == 1]
    fusion = load_fusion()
    alpha, alpha_aucs = select_alpha(evs)
    mode = "weighted"

    finals = sorted({round(float(e["response"]["fusion"]["final_score"]), 6) for e in evs} | {0.0, 1.0})
    # Both rates are monotone in the swept value, so a binary search over the observed
    # fused scores finds the same value as a full sweep.
    # 1) Medium boundary L_LOW: smallest value with Medium+ FPR <= target (NORMAL_THR = 0)
    lo, hi = 0, len(finals) - 1
    if rate(fusion, ben, 0.0, fix_levels(finals[hi]), MEDIUM_PLUS) > a.fpr_target:
        l_low, feasible = 1.0, False
    else:
        feasible = True
        while lo < hi:
            mid = (lo + hi) // 2
            if rate(fusion, ben, 0.0, fix_levels(finals[mid]), MEDIUM_PLUS) <= a.fpr_target:
                hi = mid
            else:
                lo = mid + 1
        l_low = finals[lo]
    levels = fix_levels(l_low)
    # 2) θ_low: largest value <= L_LOW with Low+ recall >= target
    cand = [c for c in finals if c <= l_low]
    lo, hi = 0, len(cand) - 1
    if rate(fusion, att, cand[0], levels, LOW_PLUS) < a.recall_target:
        theta_low = 0.0
    else:
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if rate(fusion, att, cand[mid], levels, LOW_PLUS) >= a.recall_target:
                lo = mid
            else:
                hi = mid - 1
        theta_low = cand[lo]
    # 3) θ_ml from calibrate_ml.py
    tml = json.loads((ROOT / a.models_dir / "ml/theta_ml.json").read_text())
    theta_ml, ml_agg = tml["ML_THRESHOLD"], tml.get("ML_AGGREGATE", "max")
    # 4) θ_nlp: max F1 on NLP validation
    nlpd = ROOT / a.models_dir / "nlp"
    thr_lr = json.loads((nlpd / "nlp_tfidf_lr.meta.json").read_text())["recommended_threshold"]
    thr_char = json.loads((nlpd / "nlp_char_sgd.meta.json").read_text())["recommended_threshold"]
    theta_nlp = float(max(thr_lr, thr_char))

    val_metrics = {}
    for name, acc in (("medium_plus", MEDIUM_PLUS), ("low_plus", LOW_PLUS)):
        val_metrics[name] = {"fpr_val_benign": rate(fusion, ben, theta_low, levels, acc),
                             "recall_val_attacks": rate(fusion, att, theta_low, levels, acc)}
    cfg = {
        "selected_on": a.run, "n_val_benign": len(ben), "n_val_attacks": len(att),
        "FUSION_NORMAL_THR": theta_low, "FUSION_LEVELS": levels, "medium_boundary_feasible": feasible,
        "ML_THRESHOLD": theta_ml, "NLP_THRESHOLD": theta_nlp,
        "nlp_thresholds_val": {"tfidf_lr": thr_lr, "char_sgd": thr_char},
        "fusion_mode": mode, "alpha": alpha, "beta": round(1 - alpha, 6), "alpha_roc_auc_val": alpha_aucs,
        "ML_AGGREGATE": ml_agg, "models_dir": a.models_dir,
        "impact_matrix": fusion._IMPACT.get("matrix"),
        "topic_escalation": fusion.TOPIC_ESCALATION,
        "validation_metrics_at_frozen_values": val_metrics,
        "criteria": {"medium_plus_fpr_max": a.fpr_target, "low_plus_recall_min": a.recall_target},
    }
    (run / "frozen_config.json").write_text(json.dumps(cfg, indent=2))
    (run / "frozen.env").write_text(
        f"FUSION_NORMAL_THR={theta_low}\nFUSION_LEVELS={','.join(str(v) for v in levels)}\n"
        f"ML_THRESHOLD={theta_ml:.6f}\nNLP_THRESHOLD={theta_nlp:.6f}\nML_AGGREGATE={ml_agg}\n"
        f"FUSION_MODE={mode}\nFUSION_ALPHA={alpha}\nFUSION_BETA={round(1 - alpha, 6)}\nMODELS_DIR=./{a.models_dir}\n"
        f"IMPACT_POLICY_HOST={a.impact_policy_host}\n")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
