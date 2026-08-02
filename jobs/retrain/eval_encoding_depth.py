"""Part D prototype: the `encoding_depth` feature.

encoding_depth(s) = number of urllib.parse.unquote_plus passes required before the
string stops changing, capped at 4. Computed over the raw URI and the raw body; the
per-request feature is max(depth_uri, depth_body). A benign single-encoded URI decodes
once (depth 1); a double-encoded evasion (%252F -> %2F -> /) needs two.

TASK-01 measured that residual %XX after one pass appears in 7.94 % of SR-BH attacks and
0.0008 % of SR-BH benign — signal absent from the current 24 features
(pct_encoded_ratio measures encoded-token density, not nesting depth).

This script only gathers evidence. The feature is NOT added to FEATURE_COLS — that would
break feature-order parity with every trained model. If the gain is real it becomes a
versioned feature-set bump in a later task, with retraining.

Measurements:
  1. standalone single-feature ROC-AUC for attack detection on SR-BH (full corpus) and
     on CSIC (full corpus, raw text from the kagglehub cache);
  2. incremental value on top of the existing 24: HistGradientBoostingClassifier
     (random_state=42) trained on srbh_supervised_train with and without the feature,
     evaluated once on srbh_test, paired bootstrap CI (2,000 resamples, seed 42) on the
     ROC-AUC delta.

Deterministic; touches nothing under models/ or data/ml_*.csv.
"""

import json
import logging
import os
import sys
from urllib.parse import unquote_plus

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "jobs", "retrain"))
from ml_http_features import FEATURE_COLS  # noqa: E402

log = logging.getLogger("eval_encoding_depth")

SEED = 42
N_BOOT = 2000
MAX_PASSES = 4

SRBH_DIR = os.path.join(_REPO, "data", "srbh2020")
SPLITS_DIR = os.path.join(SRBH_DIR, "splits")
OUT_PATH = os.path.join(SRBH_DIR, "transfer", "encoding_depth_results.json")

CSIC_CACHED = os.path.expanduser(
    "~/.cache/kagglehub/datasets/ispangler/csic-2010-web-application-attacks/versions/1/csic_database.csv"
)


def encoding_depth(s: str) -> int:
    """Passes of unquote_plus until fixpoint, capped at MAX_PASSES."""
    if not s or ("%" not in s and "+" not in s):
        return 0
    depth = 0
    cur = s
    while depth < MAX_PASSES:
        try:
            nxt = unquote_plus(cur)
        except Exception:
            break
        if nxt == cur:
            break
        depth += 1
        cur = nxt
    return depth


def depth_series(uri: pd.Series, body: pd.Series) -> np.ndarray:
    du = np.fromiter((encoding_depth(u) for u in uri), dtype=np.int64, count=len(uri))
    db = np.fromiter((encoding_depth(b) for b in body), dtype=np.int64, count=len(body))
    return np.maximum(du, db)


def auc_single(feature: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, feature))


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    results: dict = {"seed": SEED, "max_passes": MAX_PASSES, "definition": "max(depth(uri), depth(body)), unquote_plus fixpoint"}

    # ---- standalone AUC on SR-BH (full corpus) ----
    text = pd.read_parquet(os.path.join(SRBH_DIR, "srbh_text.parquet"))
    d_srbh = depth_series(text["text_uri"], text["text_body"])
    y_srbh = text["label"].to_numpy()
    results["srbh_full"] = {
        "rows": int(len(text)),
        "auc_standalone": auc_single(d_srbh, y_srbh),
        "depth_distribution_by_label": {
            str(lbl): {str(k): int(v) for k, v in zip(*np.unique(d_srbh[y_srbh == lbl], return_counts=True))}
            for lbl in (0, 1)
        },
    }
    log.info("SR-BH standalone AUC: %.4f", results["srbh_full"]["auc_standalone"])

    # ---- standalone AUC on CSIC (full corpus, raw text) ----
    if not os.path.exists(CSIC_CACHED):
        raise FileNotFoundError(f"CSIC raw CSV not in kagglehub cache: {CSIC_CACHED}")
    csic = pd.read_csv(CSIC_CACHED, usecols=["URL", "content", "classification"], dtype=str, na_filter=False)
    y_csic = (pd.to_numeric(csic["classification"], errors="coerce").fillna(0) != 0).astype(int).to_numpy()
    uri_csic = csic["URL"].map(lambda s: s.split()[0] if s.strip() else "")
    d_csic = depth_series(uri_csic, csic["content"])
    results["csic_full"] = {
        "rows": int(len(csic)),
        "auc_standalone": auc_single(d_csic, y_csic),
        "depth_distribution_by_label": {
            str(lbl): {str(k): int(v) for k, v in zip(*np.unique(d_csic[y_csic == lbl], return_counts=True))}
            for lbl in (0, 1)
        },
    }
    log.info("CSIC standalone AUC: %.4f", results["csic_full"]["auc_standalone"])

    # ---- incremental value on top of the 24, SR-BH supervised protocol ----
    tr = pd.read_parquet(os.path.join(SPLITS_DIR, "srbh_supervised_train.parquet"))
    te = pd.read_parquet(os.path.join(SPLITS_DIR, "srbh_test.parquet"))
    tr_text = pd.read_parquet(os.path.join(SPLITS_DIR, "srbh_supervised_train_text.parquet"))
    te_text = pd.read_parquet(os.path.join(SPLITS_DIR, "srbh_test_text.parquet"))
    assert list(tr["row_id"]) == list(tr_text["row_id"])
    assert list(te["row_id"]) == list(te_text["row_id"])

    d_tr = depth_series(tr_text["text_uri"], tr_text["text_body"])
    d_te = depth_series(te_text["text_uri"], te_text["text_body"])

    X_tr24 = tr[FEATURE_COLS].to_numpy(dtype=float)
    X_te24 = te[FEATURE_COLS].to_numpy(dtype=float)
    y_tr = tr["label"].to_numpy(dtype=int)
    y_te = te["label"].to_numpy(dtype=int)

    clf24 = HistGradientBoostingClassifier(random_state=SEED)
    clf24.fit(X_tr24, y_tr)
    p24 = clf24.predict_proba(X_te24)[:, 1]

    X_tr25 = np.column_stack([X_tr24, d_tr])
    X_te25 = np.column_stack([X_te24, d_te])
    clf25 = HistGradientBoostingClassifier(random_state=SEED)
    clf25.fit(X_tr25, y_tr)
    p25 = clf25.predict_proba(X_te25)[:, 1]

    auc24 = float(roc_auc_score(y_te, p24))
    auc25 = float(roc_auc_score(y_te, p25))

    rng = np.random.RandomState(SEED)
    n = len(y_te)
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.randint(0, n, size=n)
        yb = y_te[idx]
        if yb.min() == yb.max():
            deltas[i] = np.nan
            continue
        deltas[i] = roc_auc_score(yb, p25[idx]) - roc_auc_score(yb, p24[idx])
    deltas = deltas[np.isfinite(deltas)]

    results["srbh_supervised_delta"] = {
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "auc_24_features": auc24,
        "auc_24_plus_encoding_depth": auc25,
        "delta_auc": auc25 - auc24,
        "delta_auc_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "bootstrap": {"method": "paired_percentile", "samples": int(len(deltas)), "seed": SEED},
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
