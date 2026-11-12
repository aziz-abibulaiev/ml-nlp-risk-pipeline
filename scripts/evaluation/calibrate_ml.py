#!/usr/bin/env python3
"""
Calibration of the ML detectors and choice of the aggregator.

Each detector gets an ECDF calibration on benign validation data, the aggregator is chosen by
ROC-AUC on validation data. Updates models/ml in place and writes aggregate.json and
theta_ml.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.ml_scoring import aggregate, ecdf, raw_anomaly  # noqa: E402

KINDS = ("if", "ocsvm", "ae", "lstm_ae")
N_Q = 2001


def load(p: Path, order):
    return pd.read_csv(p)[order].to_numpy(dtype="float32")


def ecdf_table(raw_benign: np.ndarray) -> dict:
    probs = np.linspace(0.0, 1.0, N_Q)
    x = np.quantile(raw_benign, probs)
    # collapse ties: keep the largest probability for each distinct x (right-continuous CDF)
    ux, idx = np.unique(x[::-1], return_index=True)
    up = probs[::-1][idx]
    return {"x": ux.tolist(), "p": up.tolist(), "n": int(raw_benign.size), "quantiles": N_Q}


def main() -> None:
    src = dst = ROOT / "models"
    fd = ROOT / "data/splits/features"

    cal_vb, cal_va, report = [], [], {"kinds": {}}
    order = None
    for k in KINDS:
        b = joblib.load(src / "ml" / f"{k}_model.joblib")
        order = b["feature_order"]
        if k == "if":
            Xvb, Xva = load(fd / "ml_val_benign.csv", order), load(fd / "ml_val_attacks.csv", order)
        r_vb, r_va = raw_anomaly(b, Xvb), raw_anomaly(b, Xva)
        b["calibration"] = {**b["calibration"], "ecdf": ecdf_table(r_vb), "method": "ecdf_ml_val_benign"}
        s_vb = ecdf(r_vb, b["calibration"]["ecdf"]["x"], b["calibration"]["ecdf"]["p"])
        s_va = ecdf(r_va, b["calibration"]["ecdf"]["x"], b["calibration"]["ecdf"]["p"])
        cal_vb.append(s_vb); cal_va.append(s_va)
        out = dst / "ml" / f"{k}_model.joblib"
        joblib.dump(b, out)
        meta = json.loads((src / "ml" / f"{k}_model.meta.json").read_text())
        meta["calibration"] = {"method": "ecdf", "on": "ML-val benign", "quantiles": N_Q}
        meta["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
        (dst / "ml" / f"{k}_model.meta.json").write_text(json.dumps(meta, indent=2))
        print(k, "done", flush=True)

    y = np.r_[np.zeros(len(cal_vb[0]), int), np.ones(len(cal_va[0]), int)]
    cand = {}
    for how in ("max", "mean", "median"):
        s = np.r_[aggregate(cal_vb, how), aggregate(cal_va, how)]
        cand[how] = {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}
    best = max(cand, key=lambda h: cand[h]["roc_auc"])
    agg_vb = aggregate(cal_vb, best)
    theta_ml = float(np.quantile(agg_vb, 0.99))
    agg_va = aggregate(cal_va, best)
    report.update({"candidates": cand, "selected": best, "criterion": "max ROC-AUC on full ML-val",
                   "ML_THRESHOLD": theta_ml,
                   "fpr_val_benign_at_thr": float((agg_vb >= theta_ml).mean()),
                   "recall_val_attacks_at_thr": float((agg_va >= theta_ml).mean())})
    (dst / "ml" / "aggregate.json").write_text(json.dumps(report, indent=2))
    (dst / "ml" / "theta_ml.json").write_text(json.dumps({"ML_THRESHOLD": theta_ml, "ML_AGGREGATE": best}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
