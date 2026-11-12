#!/usr/bin/env python3
"""
Evaluation of the frozen ML branch on the held-out ML test split.

Writes <models-dir>/ml/ml_test_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.ml_scoring import aggregate, calibrate_bundle, raw_anomaly  # noqa: E402

KINDS = ("if", "ocsvm", "ae", "lstm_ae")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    a = ap.parse_args()
    md = ROOT / a.models_dir / "ml"
    frozen = json.loads((md / "theta_ml.json").read_text())
    thr, how = float(frozen["ML_THRESHOLD"]), frozen["ML_AGGREGATE"]
    df = pd.read_csv(ROOT / "data/splits/features/ml_test.csv")
    y = df.label.to_numpy(int)
    per, cal = {}, []
    for k in KINDS:
        b = joblib.load(md / f"{k}_model.joblib")
        s = np.asarray(calibrate_bundle(b, raw_anomaly(b, df[b["feature_order"]].to_numpy("float32"))))
        cal.append(s)
        per[k] = {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}
    s = aggregate(cal, how)
    pred = s >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    out = {"n": int(len(y)), "pos": int(y.sum()), "aggregate": how, "ML_THRESHOLD": thr,
           "per_model": per,
           "aggregated": {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
                          "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                          "recall": tp / (tp + fn), "fpr": fp / (fp + tn),
                          "precision": tp / (tp + fp) if tp + fp else float("nan")},
           "by_source": {src: {"n": int((df.source == src).sum()),
                               "flagged_rate": float(pred[(df.source == src).to_numpy()].mean())}
                         for src in sorted(df.source.unique())}}
    (md / "ml_test_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
