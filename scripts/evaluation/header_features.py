#!/usr/bin/env python3
"""
Header-feature diagnostic on a mixed run.

score   recomputes the ML score of every attack with the header features of a random
        benign event (retrain image).
decide  applies the frozen thresholds to these scores (fusion image).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
HEADER_FEATS = ("header_count", "headers_entropy")
SEED = 20261110


def load(run):
    return [json.loads(l) for l in open(ROOT / "runs" / run / "events.jsonl")]


def cmd_score(a):
    import joblib
    import numpy as np
    from common.ml_scoring import aggregate, calibrate_bundle, raw_anomaly

    agg = json.loads((ROOT / a.models_dir / "ml" / "theta_ml.json").read_text())["ML_AGGREGATE"]
    bundles = [joblib.load(ROOT / a.models_dir / "ml" / f"{k}_model.joblib") for k in ("if", "ocsvm", "ae", "lstm_ae")]
    order = bundles[0]["feature_order"]
    evs = load(a.mixed)
    ben = [e["response"]["preprocess"]["numeric_feats"] for e in evs if e["label"] == 0]
    att = [e for e in evs if e["label"] == 1]
    rng = random.Random(SEED)

    def vec(nf):
        return [float(nf.get(f, 0.0)) for f in order]

    X_orig, X_cf = [], []
    for e in att:
        nf = dict(e["response"]["preprocess"]["numeric_feats"])
        X_orig.append(vec(nf))
        donor = rng.choice(ben)
        for f in HEADER_FEATS:
            if f in donor:
                nf[f] = donor[f]
            else:
                nf.pop(f, None)
        X_cf.append(vec(nf))

    def scores(X):
        X = np.asarray(X, dtype="float32")
        return aggregate([np.asarray(calibrate_bundle(b, raw_anomaly(b, X))) for b in bundles], agg)

    s_orig, s_cf = scores(X_orig), scores(X_cf)
    recorded = np.array([float(e["response"]["ml"]["ml_score"]) for e in att])
    out = {"feature_order": order, "aggregate": agg, "replaced": list(HEADER_FEATS), "seed": SEED,
           "max_abs_diff_recomputed_vs_recorded": float(np.max(np.abs(s_orig - recorded))),
           "attacks": [{"seq": e["seq"], "attack_group": e["attack_group"],
                        "ml_recorded": float(r), "ml_counterfactual": float(c)}
                       for e, r, c in zip(att, recorded, s_cf)]}
    (ROOT / "runs" / a.mixed / "header_features_ml.json").write_text(json.dumps(out, indent=1))
    q = np.quantile(s_cf, [0.05, 0.25, 0.5, 0.75])
    print(json.dumps({"check_max_abs_diff": out["max_abs_diff_recomputed_vs_recorded"],
                      "cf_quantiles_05_25_50_75": q.round(4).tolist()}))


def cmd_decide(a):
    import ablate as ab
    import select_thresholds as st

    fusion = st.load_fusion()
    cf = {x["seq"]: x["ml_counterfactual"] for x in
          json.loads((ROOT / "runs" / a.mixed / "header_features_ml.json").read_text())["attacks"]}
    frozen = json.loads((ROOT / "runs" / a.val / "frozen_config.json").read_text())
    res = {"note": "header features of attacks replaced by those of random benign events",
           "frozen": {}, "per_alpha_reselected": {}}

    def apply_cf(evs):
        for e in evs:
            if e["label"] == 1 and e["seq"] in cf:
                e["response"]["fusion"]["agg_ml"] = cf[e["seq"]]

    mix = ab.load(a.mixed); apply_cf(mix); ab.set_alpha(mix, float(frozen["alpha"]))
    res["frozen"] = ab.confusion(fusion, mix, float(frozen["FUSION_NORMAL_THR"]), frozen["FUSION_LEVELS"])
    print("frozen", json.dumps(res["frozen"]), flush=True)
    for name, alpha in ab.ALPHAS.items():
        val, mix = ab.load(a.val), ab.load(a.mixed)
        apply_cf(mix)
        ab.set_alpha(val, alpha); ab.set_alpha(mix, alpha)
        ben = [e for e in val if e["label"] == 0]; att = [e for e in val if e["label"] == 1]
        theta, levels, feasible = ab.select(fusion, ben, att, val)
        res["per_alpha_reselected"][name] = {"alpha": alpha, "FUSION_NORMAL_THR": theta, "FUSION_LEVELS": levels,
                                             "mixed_counterfactual": ab.confusion(fusion, mix, theta, levels)}
        print(name, json.dumps(res["per_alpha_reselected"][name]["mixed_counterfactual"]), flush=True)
    (ROOT / "runs" / a.mixed / "header_features.json").write_text(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--mixed", required=True); s.add_argument("--models-dir", default="models")
    d = sub.add_parser("decide"); d.add_argument("--val", required=True); d.add_argument("--mixed", required=True)
    a = ap.parse_args()
    {"score": cmd_score, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()
