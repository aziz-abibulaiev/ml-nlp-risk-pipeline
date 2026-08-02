"""Part E — external FPR/recall READ (a smoke, not the full validation).

Runs the FROZEN binary core (threshold from calib) on external held-out sets:
 (i)  a 200k random sample of zanbil (real organic benign)  -> alert rate
 (ii) SR-BH benign                                          -> alert rate
 (iii) SR-BH attacks                                        -> recall
Nothing is tuned to these numbers. A catastrophic zanbil alert rate (>20%) is reported
prominently, not adjusted. Full external validation + rule baseline + ablations are TASK 06.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from . import config as C

ZANBIL = "data/corpus/phase2/external/zanbil.parquet"
SRBH = "data/corpus/phase2/external/srbh.parquet"
SAMPLE_N = 200_000


def _reqs(df):
    return df[["url_path_raw", "url_query", "request_body", "headers"]].to_dict("records")


def _alert_rate(det, df, thr, batch=50_000):
    n, alerts = 0, 0
    for i in range(0, len(df), batch):
        part = df.iloc[i:i + batch]
        s = det.p_content(_reqs(part))
        alerts += int((s >= thr).sum())
        n += len(part)
    return alerts / n if n else 0.0, n


def main():
    det = joblib.load(os.path.join(C.MODEL_DIR, "content_detector.joblib"))
    thr = det.threshold(C.PRIMARY_FPR)
    out = {"primary_fpr_budget": C.PRIMARY_FPR, "threshold": thr}

    z = pd.read_parquet(ZANBIL, columns=["url_path_raw", "url_query", "request_body", "headers", "label"])
    z = z.sample(n=min(SAMPLE_N, len(z)), random_state=C.SEED)
    zr, zn = _alert_rate(det, z, thr)
    out["zanbil_sample"] = {"n": zn, "alert_rate": zr}

    s = pd.read_parquet(SRBH, columns=["url_path_raw", "url_query", "request_body", "headers", "label"])
    sb = s[s.label == "benign"]; sa = s[s.label == "attack"]
    br, bn = _alert_rate(det, sb, thr)
    ar, an = _alert_rate(det, sa, thr)
    out["srbh_benign"] = {"n": bn, "alert_rate": br}
    out["srbh_attack"] = {"n": an, "recall": ar}

    with open(os.path.join(C.MODEL_DIR, "eval_external.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nREAD @ FPR-budget {C.PRIMARY_FPR} (threshold {thr:.4f}):")
    print(f"  zanbil sample alert rate : {out['zanbil_sample']['alert_rate']*100:.2f}%  (n={zn})")
    print(f"  SR-BH benign alert rate  : {out['srbh_benign']['alert_rate']*100:.2f}%  (n={bn})")
    print(f"  SR-BH attack recall      : {out['srbh_attack']['recall']*100:.2f}%  (n={an})")
    return out


if __name__ == "__main__":
    main()
