#!/usr/bin/env python3
"""
Numeric ML features for the data splits.

Uses the same featurizer as the preprocess service. Writes data/splits/features/<split>.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.featurize import numeric_features, request_to_event, simple_detect_method  # noqa: E402

FEATURE_ORDER = os.getenv(
    "FEATURE_ORDER",
    "endpoint_len,header_count,payload_size,headers_entropy,query_param_count,"
    "path_depth,suspicious_char_ratio,is_json_payload,http_method_code",
).split(",")

SPLITS = ["ml_train_benign", "ml_val_benign", "ml_val_attacks", "ml_test"]


def featurize_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        ev = request_to_event(r.method, r.url, r.user_agent, r.referrer, r.status, r.body)
        f = numeric_features(ev, *simple_detect_method(ev["method"]))
        rows.append([f[k] for k in FEATURE_ORDER])
    out = pd.DataFrame(rows, columns=FEATURE_ORDER)
    out["label"] = df["label"].to_numpy()
    out["source"] = df["source"].to_numpy()
    out["row_id"] = df["row_id"].to_numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/splits"))
    a = ap.parse_args()
    data = Path(a.data)
    (data / "features").mkdir(exist_ok=True)
    for s in SPLITS:
        df = pd.read_csv(data / f"{s}.csv", dtype=str, keep_default_na=False)
        df = df.assign(label=df["label"].astype(int))
        feats = featurize_frame(df)
        feats.to_csv(data / "features" / f"{s}.csv", index=False)
        print(s, len(feats), feats.label.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
