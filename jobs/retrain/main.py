import os, json, time
from pathlib import Path
from typing import List
import pandas as pd
import joblib
import logging
import string
import math, collections
import numpy as np
import hashlib
import re2 as re
import html, urllib.parse
import matplotlib.pyplot as plt 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from tensorflow import keras
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from common.text_norm import norm_text
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, roc_curve

logger = logging.getLogger(__name__)

# Paths to datasets
NLP_TRAIN_FILE   = os.getenv("NLP_TRAIN_FILE", "/data/nlp_train.csv")
ML_NUM_FILE      = Path(os.getenv("ML_NUM_FILE", "/data/ml_numeric.csv"))
ML_VALIDATION_FILE = Path(os.getenv("ML_VALIDATION_FILE", "/data/ml_validation.csv"))

# ML-models paths
ML_RULES_CSV        = Path(os.getenv("ML_RULES_CSV", "/data/ml_rules_silver.csv"))
ML_RULES_CLF_OUT    = os.getenv("ML_RULES_CLF_OUT", "/models/ml/ml_rules_clf.joblib")
ML_RULES_META_PATH  = os.getenv("ML_RULES_META_PATH", "/models/ml/ml_rules_clf.meta.json")

# NLP-models paths
NLP_TFIDF_LR_OUT = os.getenv("NLP_TFIDF_LR_PATH", "/models/nlp/nlp_tfidf_lr.joblib")
NLP_CHAR_SGD_OUT = os.getenv("NLP_CHAR_SGD_PATH", "/models/nlp/nlp_char_sgd.joblib")

# ML-anomaly models output paths
IF_OUT           = os.getenv("IF_OUT", "/models/ml/if_model.joblib")
OCSVM_OUT        = os.getenv("OCSVM_OUT", "/models/ml/ocsvm_model.joblib")
AE_OUT           = os.getenv("AE_OUT", "/models/ml/ae_model.joblib")
LSTM_AE_OUT      = os.getenv("LSTM_AE_OUT", "/models/ml/lstm_ae_model.joblib")
SUP_LR_OUT       = os.getenv("SUP_LR_OUT", "/models/ml/sup_lr_model.joblib")
SUP_HGB_OUT      = os.getenv("SUP_HGB_OUT", "/models/ml/sup_hgb_model.joblib")
SUP_RF_OUT       = os.getenv("SUP_RF_OUT", "/models/ml/sup_rf_model.joblib")
SUP_ET_OUT       = os.getenv("SUP_ET_OUT", "/models/ml/sup_et_model.joblib")

# NLP meta path
NLP_META_PATH    = os.getenv("NLP_META_PATH", "/models/nlp/nlp_meta.json")
TRAIN_TARGET_RAW = os.getenv("TRAIN_TARGET", "all")
TRAIN_TARGET     = (TRAIN_TARGET_RAW or "all").strip().lower()

# Minimum positive samples per attack type for ML rules classifier
MIN_POS_PER_ATTACK = int(os.getenv("MIN_POS_PER_ATTACK", "3"))

# Feature order for ML numeric training
FEATURE_ORDER_RAW = os.getenv(
    "FEATURE_ORDER",
    "endpoint_len,header_count,payload_size,headers_entropy,"
    "query_param_count,path_depth,suspicious_char_ratio,is_json_payload,http_method_code,"
    "query_len,query_entropy,query_value_len_mean,query_value_len_max,empty_query_value_ratio,"
    "path_entropy,path_digit_ratio,path_upper_ratio,body_len,body_entropy,body_digit_ratio,"
    "body_punct_ratio,pct_encoded_ratio,header_value_total_len,header_value_max_len"
)

# NLP rules paths and settings
RULES_FILE = Path(os.getenv("RULES_FILE", "/app/rules/base_rules.json"))
SILVER_MIN_PARTIAL = float(os.getenv("SILVER_MIN_PARTIAL", "0.50"))
SILVER_OUT_CSV = Path(os.getenv("NLP_TRAIN_SILVER", "/data/nlp_train_silver.csv"))

def parse_feature_order(raw: str) -> List[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]

FEATURE_ORDER = parse_feature_order(FEATURE_ORDER_RAW)
Path("/models").mkdir(parents=True, exist_ok=True)
Path("/models/ml").mkdir(parents=True, exist_ok=True)
Path("/models/nlp").mkdir(parents=True, exist_ok=True)

ML_TEST_FILE = Path(os.getenv("ML_TEST_FILE", "/data/ml_test.csv"))
NLP_TEST_FILE = os.getenv("NLP_TEST_FILE", "")  # optional, can be empty
OCSVM_EXTRA_BENIGN_FILE = os.getenv("OCSVM_EXTRA_BENIGN_FILE", "").strip()
OCSVM_EXTRA_BENIGN_MAX_ROWS = int(os.getenv("OCSVM_EXTRA_BENIGN_MAX_ROWS", "60000"))

ANOM_SIGMOID_Z_CLIP = float(os.getenv("ANOM_SIGMOID_Z_CLIP", "30.0"))
ANOM_LOGIT_C = float(os.getenv("ANOM_LOGIT_C", "0.5"))
ANOM_LOGIT_CLASS_WEIGHT = os.getenv("ANOM_LOGIT_CLASS_WEIGHT", "none").strip().lower()
ANOM_ECDF_KNOTS = int(os.getenv("ANOM_ECDF_KNOTS", "2048"))
if ANOM_ECDF_KNOTS < 64:
    ANOM_ECDF_KNOTS = 64
ML_ENSEMBLE_AGG = os.getenv("ML_ENSEMBLE_AGG", "mean_logit").strip().lower()
def _parse_kind_list(raw: str | None, default: list[str]) -> list[str]:
    vals = [x.strip().lower() for x in str(raw or "").split(",") if x.strip()]
    if not vals:
        vals = [x.strip().lower() for x in default]
    out: list[str] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out

ML_PRIMARY_CORE_KINDS = _parse_kind_list(
    os.getenv("ML_PRIMARY_CORE_KINDS", "sup_hgb,sup_rf,sup_et"),
    ["sup_hgb", "sup_rf", "sup_et"],
)
ML_AUX_NOVELTY_KINDS = _parse_kind_list(
    os.getenv("ML_AUX_NOVELTY_KINDS", "if,ocsvm,ae,lstm_ae"),
    ["if", "ocsvm", "ae", "lstm_ae"],
)
ML_PRIMARY_ENSEMBLE_AGG = os.getenv("ML_PRIMARY_ENSEMBLE_AGG", ML_ENSEMBLE_AGG or "mean_logit").strip().lower()
ML_AUX_NOVELTY_AGG = os.getenv("ML_AUX_NOVELTY_AGG", "mean_logit").strip().lower()
ML_VAL_MAX_ROWS = int(os.getenv("ML_VAL_MAX_ROWS", "300000"))
ML_TEST_MAX_ROWS = int(os.getenv("ML_TEST_MAX_ROWS", "200000"))
ML_AUTO_SPLIT_VAL_FOR_TEST = os.getenv("ML_AUTO_SPLIT_VAL_FOR_TEST", "true").strip().lower() in ("1", "true", "yes", "on")
ML_VAL_TEST_SPLIT = float(os.getenv("ML_VAL_TEST_SPLIT", "0.2"))
ML_VAL_TEST_RANDOM_STATE = int(os.getenv("ML_VAL_TEST_RANDOM_STATE", "42"))
ML_CALIB_SPLIT_FILE = Path(os.getenv("ML_CALIB_SPLIT_FILE", "/models/ml/ml_calibration_split.csv"))
ML_TEST_SPLIT_FILE = Path(os.getenv("ML_TEST_SPLIT_FILE", "/models/ml/ml_test_split.csv"))
ML_EVAL_SHUFFLE = os.getenv("ML_EVAL_SHUFFLE", "true").strip().lower() in ("1", "true", "yes", "on")
ML_LABEL_CHECK_MAX_ROWS = int(os.getenv("ML_LABEL_CHECK_MAX_ROWS", "0"))
ML_NUM_MAX_ATTACK_FRAC = float(os.getenv("ML_NUM_MAX_ATTACK_FRAC", "0.001"))
ML_SPLIT_GROUP_BY_FEATURES = os.getenv("ML_SPLIT_GROUP_BY_FEATURES", "true").strip().lower() in ("1", "true", "yes", "on")
ML_SPLIT_GROUP_INCLUDE_LABEL = os.getenv("ML_SPLIT_GROUP_INCLUDE_LABEL", "false").strip().lower() in ("1", "true", "yes", "on")
ML_SOURCE_HOLDOUT_ENABLED = os.getenv("ML_SOURCE_HOLDOUT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
ML_SOURCE_HOLDOUT_COLUMN = os.getenv("ML_SOURCE_HOLDOUT_COLUMN", "source").strip()
ML_SOURCE_HOLDOUT_SOURCE = os.getenv("ML_SOURCE_HOLDOUT_SOURCE", "csic").strip().lower()
ML_SOURCE_HOLDOUT_MIN_ROWS = int(os.getenv("ML_SOURCE_HOLDOUT_MIN_ROWS", "1000"))
ML_SOURCE_HOLDOUT_STRICT = os.getenv("ML_SOURCE_HOLDOUT_STRICT", "true").strip().lower() in ("1", "true", "yes", "on")
ML_ENABLE_SUPERVISED_BASELINES = os.getenv("ML_ENABLE_SUPERVISED_BASELINES", "true").strip().lower() in ("1", "true", "yes", "on")
ML_SUPERVISED_TRAIN_FILE = os.getenv("ML_SUPERVISED_TRAIN_FILE", "").strip()
ML_SUPERVISED_TRAIN_FRACTION = float(os.getenv("ML_SUPERVISED_TRAIN_FRACTION", "0.60"))
if (not np.isfinite(ML_SUPERVISED_TRAIN_FRACTION)) or ML_SUPERVISED_TRAIN_FRACTION <= 0.05 or ML_SUPERVISED_TRAIN_FRACTION >= 0.95:
    ML_SUPERVISED_TRAIN_FRACTION = 0.60
ML_SUPERVISED_TRAIN_SPLIT_FILE = Path(
    os.getenv("ML_SUPERVISED_TRAIN_SPLIT_FILE", "/models/ml/ml_supervised_train_split.csv")
)
ML_SUPERVISED_CALIB_SPLIT_FILE = Path(
    os.getenv("ML_SUPERVISED_CALIB_SPLIT_FILE", "/models/ml/ml_supervised_calibration_split.csv")
)
ML_SUP_MAX_TRAIN_ROWS = int(os.getenv("ML_SUP_MAX_TRAIN_ROWS", "300000"))
ML_SUP_HGB_MAX_ITER = int(os.getenv("ML_SUP_HGB_MAX_ITER", "500"))
ML_SUP_RF_N_ESTIMATORS = int(os.getenv("ML_SUP_RF_N_ESTIMATORS", "500"))
ML_SUP_ET_N_ESTIMATORS = int(os.getenv("ML_SUP_ET_N_ESTIMATORS", "700"))
ANOM_PROB_EPS = float(os.getenv("ANOM_PROB_EPS", "1e-12"))
if (not np.isfinite(ANOM_PROB_EPS)) or ANOM_PROB_EPS <= 0.0 or ANOM_PROB_EPS >= 0.1:
    ANOM_PROB_EPS = 1e-12

ML_THRESHOLD_POLICY = os.getenv("ML_THRESHOLD_POLICY", "fpr_budget").strip().lower()
ML_TARGET_FPR = float(os.getenv("ML_TARGET_FPR", "0.05"))
if (not np.isfinite(ML_TARGET_FPR)) or ML_TARGET_FPR <= 0.0 or ML_TARGET_FPR >= 1.0:
    ML_TARGET_FPR = 0.05

# Scientific protocol switch:
# by default retrain performs train+calibration only; test evaluation is a separate stage.
RETRAIN_ENABLE_TEST_EVAL = os.getenv("RETRAIN_ENABLE_TEST_EVAL", "false").strip().lower() in ("1", "true", "yes", "on")

ML_BOOTSTRAP_CI_ENABLED = os.getenv("ML_BOOTSTRAP_CI_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
ML_BOOTSTRAP_CI_SAMPLES = int(os.getenv("ML_BOOTSTRAP_CI_SAMPLES", "500"))
if ML_BOOTSTRAP_CI_SAMPLES < 50:
    ML_BOOTSTRAP_CI_SAMPLES = 50
if ML_BOOTSTRAP_CI_SAMPLES > 5000:
    ML_BOOTSTRAP_CI_SAMPLES = 5000
ML_BOOTSTRAP_CI_ALPHA = float(os.getenv("ML_BOOTSTRAP_CI_ALPHA", "0.05"))
if (not np.isfinite(ML_BOOTSTRAP_CI_ALPHA)) or ML_BOOTSTRAP_CI_ALPHA <= 0.0 or ML_BOOTSTRAP_CI_ALPHA >= 1.0:
    ML_BOOTSTRAP_CI_ALPHA = 0.05
ML_BOOTSTRAP_CI_MAX_ROWS = int(os.getenv("ML_BOOTSTRAP_CI_MAX_ROWS", "50000"))
if ML_BOOTSTRAP_CI_MAX_ROWS < 5000:
    ML_BOOTSTRAP_CI_MAX_ROWS = 5000
ML_BOOTSTRAP_CI_RANDOM_STATE = int(os.getenv("ML_BOOTSTRAP_CI_RANDOM_STATE", "42"))

ML_DEPLOY_ATTACK_PRIOR = float(os.getenv("ML_DEPLOY_ATTACK_PRIOR", "-1.0"))
if (not np.isfinite(ML_DEPLOY_ATTACK_PRIOR)) or ML_DEPLOY_ATTACK_PRIOR <= 0.0 or ML_DEPLOY_ATTACK_PRIOR >= 1.0:
    ML_DEPLOY_ATTACK_PRIOR = -1.0

def split_train_val_test(
    X, y,
    val_size=0.2,
    test_size=0.1,
    random_state=42,
):
    """
    3-way split: train / val / test with stratification if possible.
    """
    y_arr = np.asarray(y)
    strat = y_arr if len(np.unique(y_arr)) > 1 else None

    # first split off test
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y_arr,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    # split tmp into train/val
    strat2 = y_tmp if len(np.unique(y_tmp)) > 1 else None
    rel_val = val_size / max(1e-9, (1.0 - test_size))
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp,
        test_size=rel_val,
        random_state=random_state,
        stratify=strat2,
    )
    return X_tr, y_tr, X_va, y_va, X_te, y_te

def best_threshold_f1(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """
    Return (best_thr, best_f1) maximizing F1 on provided set.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    prec, rec, thr = precision_recall_curve(y_true, prob)
    denom = prec + rec
    f1s = np.zeros_like(denom)
    m = denom > 0
    f1s[m] = 2.0 * prec[m] * rec[m] / denom[m]
    thr2 = np.append(thr, [thr[-1] if len(thr) else 0.5])
    idx = int(np.argmax(f1s)) if len(f1s) else 0
    return float(thr2[idx]) if len(thr2) else 0.5, float(f1s[idx]) if len(f1s) else 0.0

def threshold_for_target_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
) -> tuple[float | None, dict]:
    y = np.asarray(y_true).astype(int).reshape(-1)
    s = np.asarray(scores).astype(float).reshape(-1)
    m = np.isfinite(y) & np.isfinite(s)
    y = y[m]
    s = s[m]
    out = {
        "target_fpr": float(target_fpr),
        "actual_fpr": None,
        "actual_tpr": None,
        "benign_rows": int(np.sum(y == 0)),
        "attack_rows": int(np.sum(y == 1)),
        "threshold": None,
    }
    if y.size == 0 or np.sum(y == 0) <= 0:
        return None, out

    benign = s[y == 0]
    q = float(np.clip(1.0 - float(target_fpr), 0.0, 1.0))
    try:
        thr = float(np.quantile(benign, q, method="higher"))
    except TypeError:
        # NumPy <1.22 compatibility
        thr = float(np.quantile(benign, q, interpolation="higher"))
    out["threshold"] = thr
    out["actual_fpr"] = float(np.mean(benign >= thr))
    if np.any(y == 1):
        out["actual_tpr"] = float(np.mean(s[y == 1] >= thr))
    return thr, out

def choose_operating_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict]:
    y = np.asarray(y_true).astype(int).reshape(-1)
    s = np.asarray(scores).astype(float).reshape(-1)
    m = np.isfinite(y) & np.isfinite(s)
    y = y[m]
    s = s[m]

    f1_thr, f1_best = best_threshold_f1(y, s)
    info = {
        "policy": "best_f1",
        "selected_threshold": float(f1_thr),
        "best_f1_ref": float(f1_best),
        "best_f1_threshold_ref": float(f1_thr),
    }

    if ML_THRESHOLD_POLICY in ("fpr", "fpr_budget", "neyman_pearson", "neyman-pearson"):
        thr_fpr, fpr_info = threshold_for_target_fpr(y, s, ML_TARGET_FPR)
        if thr_fpr is not None:
            info = {
                "policy": "fpr_budget",
                "selected_threshold": float(thr_fpr),
                "best_f1_ref": float(f1_best),
                "best_f1_threshold_ref": float(f1_thr),
                **fpr_info,
            }
            return float(thr_fpr), info
    return float(f1_thr), info


def _safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def _binary_metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    y = np.asarray(y_true).astype(int).reshape(-1)
    s = np.asarray(scores).astype(float).reshape(-1)
    m = np.isfinite(y) & np.isfinite(s)
    y = y[m]
    s = s[m]

    pred = (s >= float(thr)).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    out = {
        "rows": int(y.size),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == 0)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_div(tp + tn, y.size),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "tpr": _safe_div(tp, tp + fn),
        "tnr": _safe_div(tn, tn + fp),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "f1": float(f1_score(y, pred, zero_division=0)) if y.size else None,
    }
    if len(np.unique(y)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def _downsample_for_bootstrap(y_true: np.ndarray, scores: np.ndarray, max_rows: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).reshape(-1)
    s = np.asarray(scores).astype(float).reshape(-1)
    m = np.isfinite(y) & np.isfinite(s)
    y = y[m]
    s = s[m]
    if max_rows <= 0 or y.size <= max_rows:
        return y, s

    idx_all = np.arange(y.size)
    idx_keep = None
    if len(np.unique(y)) >= 2:
        try:
            idx_keep, _ = train_test_split(
                idx_all,
                train_size=max_rows,
                random_state=random_state,
                stratify=y,
            )
        except Exception:
            idx_keep = None
    if idx_keep is None:
        rng = np.random.RandomState(random_state)
        idx_keep = rng.choice(idx_all, size=max_rows, replace=False)
    idx_keep = np.sort(np.asarray(idx_keep, dtype=int))
    return y[idx_keep], s[idx_keep]


def _bootstrap_ci_for_threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    thr: float,
    n_boot: int,
    alpha: float,
    random_state: int,
    max_rows: int,
) -> dict | None:
    if n_boot <= 0:
        return None
    y, s = _downsample_for_bootstrap(y_true, scores, max_rows=max_rows, random_state=random_state)
    n = int(y.size)
    if n < 2:
        return None

    metric_names = (
        "f1",
        "precision",
        "recall",
        "tpr",
        "tnr",
        "fpr",
        "fnr",
        "accuracy",
        "roc_auc",
        "pr_auc",
    )
    values = {k: [] for k in metric_names}
    rng = np.random.RandomState(random_state)
    idx_base = np.arange(n)

    for _ in range(int(n_boot)):
        idx = rng.choice(idx_base, size=n, replace=True)
        mm = _binary_metrics_at_threshold(y[idx], s[idx], thr)
        for k in metric_names:
            v = mm.get(k)
            if v is not None and np.isfinite(v):
                values[k].append(float(v))

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    ci_metrics = {}
    for k, seq in values.items():
        if not seq:
            continue
        arr = np.asarray(seq, dtype=float)
        ci_metrics[k] = {
            "low": float(np.percentile(arr, lo_q)),
            "high": float(np.percentile(arr, hi_q)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "n_valid": int(arr.size),
        }

    if not ci_metrics:
        return None
    return {
        "method": "bootstrap_percentile",
        "confidence": float(1.0 - alpha),
        "alpha": float(alpha),
        "samples": int(n_boot),
        "rows_used": int(n),
        "random_state": int(random_state),
        "metrics": ci_metrics,
    }

def metrics_on_threshold(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    out = _binary_metrics_at_threshold(y_true, prob, thr)
    if ML_BOOTSTRAP_CI_ENABLED:
        ci = _bootstrap_ci_for_threshold_metrics(
            y_true,
            prob,
            thr=thr,
            n_boot=ML_BOOTSTRAP_CI_SAMPLES,
            alpha=ML_BOOTSTRAP_CI_ALPHA,
            random_state=ML_BOOTSTRAP_CI_RANDOM_STATE,
            max_rows=ML_BOOTSTRAP_CI_MAX_ROWS,
        )
        if ci is not None:
            out["bootstrap_ci"] = ci
    return out

def _maybe_downsample_df(
    df: pd.DataFrame,
    y: np.ndarray | None,
    max_rows: int,
    tag: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    if max_rows <= 0 or len(df) <= max_rows:
        if ML_EVAL_SHUFFLE and y is not None and len(y) == len(df):
            rng = np.random.RandomState(random_state)
            idx = np.arange(len(df))
            rng.shuffle(idx)
            return df.iloc[idx].reset_index(drop=True), y[idx]
        return df, y

    idx_all = np.arange(len(df))
    idx_keep = None
    if y is not None and len(y) == len(df) and len(np.unique(y)) >= 2:
        try:
            idx_keep, _ = train_test_split(
                idx_all,
                train_size=max_rows,
                random_state=random_state,
                stratify=y,
            )
        except Exception:
            idx_keep = None

    if idx_keep is None:
        rng = np.random.RandomState(random_state)
        idx_keep = rng.choice(idx_all, size=max_rows, replace=False)

    idx_keep = np.sort(np.asarray(idx_keep, dtype=int))
    df_out = df.iloc[idx_keep].reset_index(drop=True)
    y_out = y[idx_keep] if y is not None and len(y) == len(df) else y
    logger.info(
        "[retrain][%s] downsampled rows: %s -> %s",
        tag,
        len(df),
        len(df_out),
    )
    return df_out, y_out

def _build_group_hash_keys(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    if not cols:
        return np.arange(len(df), dtype=np.int64)
    # Canonicalize numeric-like columns before hashing so that semantically
    # equal values (e.g., "0" vs 0) map to one group key.
    key_df = pd.DataFrame(index=df.index)
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.99:
            key_df[c] = s.astype(np.float64).round(12).fillna(-1e308)
        else:
            key_df[c] = df[c].astype(str).fillna("<NA>")

    # Stable hash of canonicalized columns; identical tuples map to one group.
    return pd.util.hash_pandas_object(key_df, index=False).to_numpy(dtype=np.uint64, copy=False)

def _select_groups_for_target_rows(
    group_counts: np.ndarray,
    target_rows: int,
    random_state: int,
) -> np.ndarray:
    counts = np.asarray(group_counts, dtype=int).reshape(-1)
    n_groups = counts.size
    selected = np.zeros(n_groups, dtype=bool)
    if n_groups == 0 or target_rows <= 0:
        return selected

    target = int(max(0, min(int(target_rows), int(np.sum(counts)))))
    if target == 0:
        return selected
    if target >= int(np.sum(counts)):
        selected[:] = True
        return selected

    rng = np.random.RandomState(int(random_state))
    # Randomized group order prevents "few largest groups dominate test split"
    # and keeps far richer feature diversity under strict group disjointness.
    order = np.argsort(rng.random_sample(n_groups), kind="mergesort")
    cum = np.cumsum(counts[order], dtype=np.int64)
    k_hi = int(np.searchsorted(cum, target, side="left"))
    k_hi = int(max(0, min(k_hi, n_groups - 1)))

    if k_hi == 0:
        cutoff = 0
        cur = int(cum[0])
    else:
        cur_hi = int(cum[k_hi])
        cur_lo = int(cum[k_hi - 1])
        if abs(cur_lo - target) <= abs(cur_hi - target):
            cutoff = k_hi - 1
            cur = cur_lo
        else:
            cutoff = k_hi
            cur = cur_hi

    selected[order[: cutoff + 1]] = True

    # one-step local adjustment at the boundary for better row-target match
    diff = int(target - cur)
    if diff > 0:
        pool = np.where(~selected)[0]
        if pool.size > 0:
            gi = int(pool[np.argmin(np.abs(counts[pool] - diff))])
            selected[gi] = True
    elif diff < 0:
        pool = np.where(selected)[0]
        if pool.size > 1:
            gi = int(pool[np.argmin(np.abs(counts[pool] + diff))])
            selected[gi] = False

    return selected

def build_calibration_and_test_from_validation(
    validation_path: str,
    required_cols: List[str],
    out_calib_path: Path,
    out_test_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[str, str]:
    """
    Build disjoint calibration/test splits from a single mixed validation file.
    Useful when a separate ML_TEST_FILE is not provided.
    """
    df_val = pd.read_csv(validation_path, low_memory=False)
    have = set(df_val.columns.tolist())
    need = set(required_cols)
    if not need.issubset(have):
        raise SystemExit(
            f"[retrain] ML validation {validation_path} has wrong columns for split. "
            f"Need at least: {sorted(need)}, have: {sorted(have)}"
        )

    y_val = _labels_from_df(df_val)
    if y_val is None or len(np.unique(y_val)) < 2:
        raise SystemExit(
            f"[retrain] Cannot split {validation_path} into calibration/test: "
            "label column missing or only one class."
        )

    ts = float(test_size)
    if (not np.isfinite(ts)) or ts <= 0.0 or ts >= 0.5:
        ts = 0.2

    idx_cal = None
    idx_test = None
    split_mode = "row_stratified"
    split_details: dict = {}
    group_cols = [c for c in required_cols if c in df_val.columns and c != "label"]
    if ML_SPLIT_GROUP_INCLUDE_LABEL and "label" in df_val.columns:
        group_cols = group_cols + ["label"]

    if (
        ML_SOURCE_HOLDOUT_ENABLED
        and ML_SOURCE_HOLDOUT_COLUMN
        and ML_SOURCE_HOLDOUT_COLUMN in df_val.columns
    ):
        src = (
            df_val[ML_SOURCE_HOLDOUT_COLUMN]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace("", "unknown")
        )
        min_rows = max(1, int(ML_SOURCE_HOLDOUT_MIN_ROWS))
        src_counts = src.value_counts().to_dict()

        selected_source = ""
        requested_source = (ML_SOURCE_HOLDOUT_SOURCE or "").strip().lower()
        if requested_source and requested_source != "auto":
            if requested_source in src_counts:
                selected_source = requested_source
            else:
                logger.info(
                    "[retrain] source holdout requested source=%s not found in column=%s; fallback to split",
                    requested_source,
                    ML_SOURCE_HOLDOUT_COLUMN,
                )
        else:
            # AUTO mode: choose the largest source that keeps both splits mixed.
            candidates = []
            for src_name, n_rows in src_counts.items():
                m = src == src_name
                y_test_src = y_val[m.to_numpy()]
                y_cal_src = y_val[~m.to_numpy()]
                if (
                    int(n_rows) >= min_rows
                    and len(np.unique(y_test_src)) >= 2
                    and len(np.unique(y_cal_src)) >= 2
                ):
                    candidates.append((int(n_rows), str(src_name)))
            if candidates:
                candidates.sort(reverse=True)
                selected_source = candidates[0][1]

        if selected_source:
            mask_test = (src == selected_source).to_numpy()
            idx_test_src = np.flatnonzero(mask_test)
            idx_cal_src = np.flatnonzero(~mask_test)
            y_test_src = y_val[idx_test_src]
            y_cal_src = y_val[idx_cal_src]
            if (
                len(idx_test_src) >= min_rows
                and len(idx_cal_src) > 0
                and len(np.unique(y_test_src)) >= 2
                and len(np.unique(y_cal_src)) >= 2
            ):
                idx_test = idx_test_src
                idx_cal = idx_cal_src
                split_mode = "source_holdout"
                split_details = {
                    "source_column": ML_SOURCE_HOLDOUT_COLUMN,
                    "holdout_source": selected_source,
                    "holdout_rows": int(len(idx_test_src)),
                    "holdout_pos": int(np.asarray(y_test_src).sum()),
                    "calib_rows": int(len(idx_cal_src)),
                    "calib_pos": int(np.asarray(y_cal_src).sum()),
                    "source_rows": {str(k): int(v) for k, v in src_counts.items()},
                }
            else:
                logger.info(
                    "[retrain] source holdout source=%s invalid for mixed calibration/test "
                    "(test_rows=%s test_classes=%s calib_rows=%s calib_classes=%s); fallback to split",
                    selected_source,
                    len(idx_test_src),
                    len(np.unique(y_test_src)),
                    len(idx_cal_src),
                    len(np.unique(y_cal_src)),
                )
        if ML_SOURCE_HOLDOUT_STRICT and (idx_cal is None or idx_test is None):
            raise SystemExit(
                "[retrain] strict source holdout enabled but valid source holdout split "
                f"was not created (column={ML_SOURCE_HOLDOUT_COLUMN}, source={ML_SOURCE_HOLDOUT_SOURCE or 'auto'})."
            )

    if idx_cal is None and idx_test is None and ML_SPLIT_GROUP_BY_FEATURES and group_cols:
        try:
            group_keys = _build_group_hash_keys(df_val, group_cols)
            test_mask = np.zeros(len(df_val), dtype=bool)
            groups_test_total = 0
            groups_all_total = 0
            for cls in sorted(np.unique(y_val).tolist()):
                cls_idx = np.flatnonzero(y_val == int(cls))
                if cls_idx.size == 0:
                    continue
                group_cls = group_keys[cls_idx]
                uniq_cls, inv_cls = np.unique(group_cls, return_inverse=True)
                cnt_cls = np.bincount(inv_cls)
                target_cls = int(round(ts * cls_idx.size))
                sel_groups = _select_groups_for_target_rows(
                    cnt_cls,
                    target_cls,
                    random_state=int(random_state) + int(cls),
                )
                rows_test_cls = np.isin(inv_cls, np.flatnonzero(sel_groups))
                test_mask[cls_idx[rows_test_cls]] = True
                groups_test_total += int(sel_groups.sum())
                groups_all_total += int(len(uniq_cls))

            # Enforce strict group-wise disjointness across classes.
            # Without this, identical feature groups that appear in both labels
            # can be split across calibration/test due to class-wise selection.
            if not ML_SPLIT_GROUP_INCLUDE_LABEL:
                _, inv_all = np.unique(group_keys, return_inverse=True)
                grp_total = np.bincount(inv_all)
                grp_test = np.bincount(inv_all, weights=test_mask.astype(int))
                grp_mixed = (grp_test > 0.0) & (grp_test < grp_total)
                n_mixed = int(np.sum(grp_mixed))
                if n_mixed > 0:
                    grp_to_test = grp_mixed & (grp_test >= (grp_total - grp_test))
                    grp_to_cal = grp_mixed & ~grp_to_test
                    if np.any(grp_to_test):
                        test_mask[np.isin(inv_all, np.flatnonzero(grp_to_test))] = True
                    if np.any(grp_to_cal):
                        test_mask[np.isin(inv_all, np.flatnonzero(grp_to_cal))] = False
                    split_details["n_mixed_groups_reassigned"] = n_mixed

            idx_test = np.flatnonzero(test_mask)
            idx_cal = np.flatnonzero(~test_mask)
            if (
                len(idx_cal) == 0
                or len(idx_test) == 0
                or len(np.unique(y_val[idx_cal])) < 2
                or len(np.unique(y_val[idx_test])) < 2
            ):
                idx_cal = None
                idx_test = None
            else:
                split_mode = "group_hash_classwise"
                split_details = {
                    "group_cols": list(group_cols),
                    "n_groups": int(groups_all_total),
                    "n_groups_test": int(groups_test_total),
                    "n_groups_cal": int(groups_all_total - groups_test_total),
                }
        except Exception as e:
            logger.info("[retrain] group split failed (%s), fallback to row stratified split", e)
            idx_cal = None
            idx_test = None

    if idx_cal is None or idx_test is None:
        idx = np.arange(len(df_val))
        idx_cal, idx_test = train_test_split(
            idx,
            test_size=ts,
            random_state=int(random_state),
            stratify=y_val,
        )

    idx_cal = np.sort(np.asarray(idx_cal, dtype=int))
    idx_test = np.sort(np.asarray(idx_test, dtype=int))

    df_cal = df_val.iloc[idx_cal].reset_index(drop=True)
    y_cal = y_val[idx_cal]
    df_tst = df_val.iloc[idx_test].reset_index(drop=True)
    y_tst = y_val[idx_test]

    if ML_EVAL_SHUFFLE:
        df_cal, y_cal = _maybe_downsample_df(
            df_cal,
            y_cal,
            max_rows=0,
            tag="split/calibration_shuffle",
            random_state=int(random_state),
        )
        df_tst, y_tst = _maybe_downsample_df(
            df_tst,
            y_tst,
            max_rows=0,
            tag="split/test_shuffle",
            random_state=int(random_state) + 1,
        )
        # keep label column synced in case any downstream reads it directly
        if "label" in df_cal.columns:
            df_cal["label"] = y_cal
        if "label" in df_tst.columns:
            df_tst["label"] = y_tst

    out_calib_path.parent.mkdir(parents=True, exist_ok=True)
    out_test_path.parent.mkdir(parents=True, exist_ok=True)
    df_cal.to_csv(out_calib_path, index=False)
    df_tst.to_csv(out_test_path, index=False)

    logger.info(
        "[retrain] built calib/test from validation: mode=%s src=%s calib=%s (%s rows, pos=%s) test=%s (%s rows, pos=%s)",
        split_mode,
        validation_path,
        out_calib_path,
        len(df_cal),
        int(np.asarray(y_cal).sum()),
        out_test_path,
        len(df_tst),
        int(np.asarray(y_tst).sum()),
    )
    if split_details:
        logger.info(
            "[retrain] split_details: group_cols=%s n_groups=%s n_groups_cal=%s n_groups_test=%s n_mixed_groups_reassigned=%s",
            split_details.get("group_cols"),
            split_details.get("n_groups"),
            split_details.get("n_groups_cal"),
            split_details.get("n_groups_test"),
            split_details.get("n_mixed_groups_reassigned", 0),
        )
    return str(out_calib_path), str(out_test_path)

def _normalize_anomaly_kind(kind: str | None) -> str:
    k = (kind or "").strip().lower()
    if "ocsvm" in k:
        return "ocsvm"
    if "if" in k:
        return "if"
    if "lstm" in k:
        return "lstm_ae"
    if "ae" in k:
        return "ae"
    return k or "if"

def _default_score_orientation(kind_norm: str) -> str:
    return "negate" if kind_norm in ("if", "ocsvm") else "identity"

def _resolve_score_orientation(
    kind: str | None,
    calibration: dict | None = None,
    override: str | None = None,
) -> str:
    candidates = [override]
    if isinstance(calibration, dict):
        candidates.append(calibration.get("orientation"))
        attack_log = calibration.get("attack_logistic")
        if isinstance(attack_log, dict):
            candidates.append(attack_log.get("orientation"))

    for candidate in candidates:
        if candidate is None:
            continue
        value = str(candidate).strip().lower()
        if value in ("negate", "flip", "reverse", "-1"):
            return "negate"
        if value in ("identity", "as_is", "none", "+1", "1"):
            return "identity"

    return _default_score_orientation(_normalize_anomaly_kind(kind))

def _oriented_stats(stats: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "negate":
        return -stats
    return stats

def _should_use_attack_logistic(
    logit_metrics: dict | None,
    baseline_eval_metrics: dict | None = None,
    runtime_eval_metrics: dict | None = None,
    min_roc_auc: float = 0.55,
    max_allowed_roc_drop: float = 0.005,
    max_allowed_pr_drop: float = 0.01,
    min_score_spread: float = 0.05,
    max_allowed_nll_increase: float = 0.02,
    max_allowed_brier_increase: float = 0.01,
    max_benign_sat_near1: float = 0.20,
) -> tuple[bool, str]:
    if not isinstance(logit_metrics, dict):
        return False, "missing_metrics"

    roc_logit = logit_metrics.get("roc_auc_attack")
    if isinstance(roc_logit, (float, int)):
        roc_logit = float(roc_logit)
        if roc_logit < min_roc_auc:
            return False, f"roc_below_min({roc_logit:.4f}<{min_roc_auc:.4f})"
    else:
        roc_logit = None

    if isinstance(baseline_eval_metrics, dict):
        roc_base = baseline_eval_metrics.get("roc_auc")
        if isinstance(roc_base, (float, int)) and roc_logit is not None:
            roc_base = float(roc_base)
            if roc_logit + max_allowed_roc_drop < roc_base:
                return False, (
                    f"roc_drop_too_large(logit={roc_logit:.4f},"
                    f" baseline={roc_base:.4f}, drop>{max_allowed_roc_drop:.4f})"
                )

    if isinstance(baseline_eval_metrics, dict) and isinstance(runtime_eval_metrics, dict):
        pr_base = baseline_eval_metrics.get("pr_auc")
        pr_run = runtime_eval_metrics.get("pr_auc")
        if isinstance(pr_base, (float, int)) and isinstance(pr_run, (float, int)):
            pr_base = float(pr_base)
            pr_run = float(pr_run)
            if pr_run + max_allowed_pr_drop < pr_base:
                return False, (
                    f"pr_drop_too_large(runtime={pr_run:.4f},"
                    f" baseline={pr_base:.4f}, drop>{max_allowed_pr_drop:.4f})"
                )

    if isinstance(runtime_eval_metrics, dict):
        p01 = runtime_eval_metrics.get("score_p01")
        p99 = runtime_eval_metrics.get("score_p99")
        if isinstance(p01, (float, int)) and isinstance(p99, (float, int)):
            spread = float(p99) - float(p01)
            if spread < min_score_spread:
                return False, f"spread_too_small({spread:.6f}<{min_score_spread:.6f})"

    if isinstance(baseline_eval_metrics, dict) and isinstance(runtime_eval_metrics, dict):
        nll_base = baseline_eval_metrics.get("log_loss")
        nll_run = runtime_eval_metrics.get("log_loss")
        if isinstance(nll_base, (float, int)) and isinstance(nll_run, (float, int)):
            nll_base = float(nll_base)
            nll_run = float(nll_run)
            if nll_run > nll_base + max_allowed_nll_increase:
                return False, (
                    f"nll_worse(runtime={nll_run:.4f},"
                    f" baseline={nll_base:.4f}, +>{max_allowed_nll_increase:.4f})"
                )

        brier_base = baseline_eval_metrics.get("brier")
        brier_run = runtime_eval_metrics.get("brier")
        if isinstance(brier_base, (float, int)) and isinstance(brier_run, (float, int)):
            brier_base = float(brier_base)
            brier_run = float(brier_run)
            if brier_run > brier_base + max_allowed_brier_increase:
                return False, (
                    f"brier_worse(runtime={brier_run:.4f},"
                    f" baseline={brier_base:.4f}, +>{max_allowed_brier_increase:.4f})"
                )

    if isinstance(runtime_eval_metrics, dict):
        benign_sat = runtime_eval_metrics.get("pct_near_1_0_benign_eps1e3")
        if isinstance(benign_sat, (float, int)) and float(benign_sat) > max_benign_sat_near1:
            return False, f"benign_saturation_too_high({float(benign_sat):.4f}>{max_benign_sat_near1:.4f})"

    return True, "ok"

def _eval_scores_only_with_labels(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compute the same core quality metrics as _eval_and_plot_scores_with_labels,
    but without writing CSV/plots. Used for objective candidate selection.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)

    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask]
    labels = labels[mask]
    if scores.size == 0 or labels.size == 0:
        return {}

    scores_prob = np.clip(scores, 1e-9, 1.0 - 1e-9)
    log_loss = _binary_logloss(labels, scores_prob)
    brier = float(np.mean((scores_prob - labels) ** 2))
    ece15 = _expected_calibration_error(labels, scores_prob, n_bins=15)

    try:
        roc = roc_auc_score(labels, scores)
    except Exception:
        roc = None
    try:
        ap = average_precision_score(labels, scores)
    except Exception:
        ap = None

    prec, rec, thr = precision_recall_curve(labels, scores)
    denom = prec + rec
    f1s = np.zeros_like(denom)
    valid = denom > 0
    f1s[valid] = 2.0 * prec[valid] * rec[valid] / denom[valid]
    best_f1 = float(f1s.max()) if len(f1s) else None
    thr_for_f1 = np.append(thr, [thr[-1] if len(thr) else 0.5])
    best_idx = int(np.argmax(f1s)) if len(f1s) else 0
    best_thr = float(thr_for_f1[best_idx]) if len(thr_for_f1) else 0.5

    return {
        "roc_auc": float(roc) if roc is not None else None,
        "pr_auc": float(ap) if ap is not None else None,
        "log_loss": float(log_loss),
        "brier": float(brier),
        "ece_15": float(ece15) if np.isfinite(ece15) else None,
        "best_f1": best_f1,
        "best_thr": best_thr,
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_p01": float(np.quantile(scores, 0.01)),
        "score_p99": float(np.quantile(scores, 0.99)),
        "pct_near_0_5_eps1e3": float(np.mean(np.abs(scores - 0.5) < 1e-3)),
        "pct_near_1_0_eps1e3": float(np.mean(scores > (1.0 - 1e-3))),
        "pct_near_1_0_benign_eps1e3": float(np.mean(scores[labels == 0] > (1.0 - 1e-3))) if np.any(labels == 0) else None,
        "pct_near_1_0_attack_eps1e3": float(np.mean(scores[labels == 1] > (1.0 - 1e-3))) if np.any(labels == 1) else None,
        "uniq_1e4": int(np.unique(np.round(scores, 4)).size),
        "rows": int(len(labels)),
        "positives": int(labels.sum()),
    }

def _finite_metric(value, fallback: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    if not np.isfinite(out):
        return float(fallback)
    return out

def _select_best_runtime_calibration_candidate(
    candidates: list[dict],
) -> tuple[dict, dict, str]:
    """
    Select runtime calibration candidate by proper scoring rules.
    Primary: lower log-loss; secondary: lower Brier; then lower ECE.
    Additional tie-breakers reduce saturation and prefer stronger ranking.
    """
    if not candidates:
        return {}, {}, "no_candidates"

    def rank(c: dict) -> tuple:
        m = c.get("eval", {}) if isinstance(c, dict) else {}
        p01 = _finite_metric(m.get("score_p01"), 0.0)
        p99 = _finite_metric(m.get("score_p99"), 0.0)
        spread = max(0.0, p99 - p01)
        return (
            _finite_metric(m.get("log_loss"), float("inf")),
            _finite_metric(m.get("brier"), float("inf")),
            _finite_metric(m.get("ece_15"), float("inf")),
            _finite_metric(m.get("pct_near_1_0_benign_eps1e3"), float("inf")),
            _finite_metric(m.get("pct_near_0_5_eps1e3"), float("inf")),
            -spread,
            -_finite_metric(m.get("pr_auc"), float("-inf")),
            -_finite_metric(m.get("roc_auc"), float("-inf")),
        )

    best = min(candidates, key=rank)
    comparison = {}
    for c in candidates:
        name = str(c.get("name", "candidate"))
        ev = dict(c.get("eval") or {})
        comparison[name] = {
            "log_loss": ev.get("log_loss"),
            "brier": ev.get("brier"),
            "ece_15": ev.get("ece_15"),
            "pr_auc": ev.get("pr_auc"),
            "roc_auc": ev.get("roc_auc"),
            "score_p01": ev.get("score_p01"),
            "score_p99": ev.get("score_p99"),
            "pct_near_0_5_eps1e3": ev.get("pct_near_0_5_eps1e3"),
            "pct_near_1_0_benign_eps1e3": ev.get("pct_near_1_0_benign_eps1e3"),
            "pct_near_1_0_attack_eps1e3": ev.get("pct_near_1_0_attack_eps1e3"),
            "uniq_1e4": ev.get("uniq_1e4"),
            "temperature_scaling": c.get("temperature_scaling"),
        }

    reason = (
        f"selected={best.get('name')} "
        f"log_loss={_finite_metric((best.get('eval') or {}).get('log_loss'), float('nan')):.6f} "
        f"brier={_finite_metric((best.get('eval') or {}).get('brier'), float('nan')):.6f}"
    )
    return best, comparison, reason

def _safe_logit(p: np.ndarray, eps: float | None = None) -> np.ndarray:
    if eps is None:
        eps = ANOM_PROB_EPS
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def _apply_prior_correction_probs(
    probs: np.ndarray,
    prior_cfg: dict | None,
) -> np.ndarray:
    p = np.asarray(probs, dtype=float).reshape(-1)
    if not isinstance(prior_cfg, dict):
        return p
    if not bool(prior_cfg.get("enabled", False)):
        return p
    try:
        p_cal = float(prior_cfg.get("calibration_prior"))
        p_dep = float(prior_cfg.get("deployment_prior"))
    except Exception:
        return p
    if (
        (not np.isfinite(p_cal))
        or (not np.isfinite(p_dep))
        or p_cal <= 0.0
        or p_cal >= 1.0
        or p_dep <= 0.0
        or p_dep >= 1.0
    ):
        return p
    odds = np.exp(_safe_logit(np.clip(p, ANOM_PROB_EPS, 1.0 - ANOM_PROB_EPS)))
    prior_ratio = (p_dep / (1.0 - p_dep)) / (p_cal / (1.0 - p_cal))
    odds_adj = odds * prior_ratio
    return odds_adj / (1.0 + odds_adj)

def _attach_prior_correction_to_calibration(
    calibration: dict | None,
    labels: np.ndarray | None,
    kind: str = "",
) -> tuple[dict, dict]:
    calib = dict(calibration or {})
    info = {
        "enabled": False,
        "deployment_prior": None,
        "calibration_prior": None,
        "kind": str(kind),
        "reason": "disabled",
    }
    if ML_DEPLOY_ATTACK_PRIOR <= 0.0 or ML_DEPLOY_ATTACK_PRIOR >= 1.0:
        calib.pop("prior_correction", None)
        info["reason"] = "deploy_prior_not_set"
        return calib, info
    if labels is None:
        calib.pop("prior_correction", None)
        info["reason"] = "labels_missing"
        return calib, info

    y = np.asarray(labels, dtype=int).reshape(-1)
    y = y[np.isfinite(y)]
    if y.size == 0:
        calib.pop("prior_correction", None)
        info["reason"] = "labels_empty"
        return calib, info

    p_cal = float(np.mean(y))
    if p_cal <= 0.0 or p_cal >= 1.0:
        calib.pop("prior_correction", None)
        info["reason"] = "calibration_prior_invalid"
        info["calibration_prior"] = p_cal
        return calib, info

    prior_cfg = {
        "enabled": True,
        "calibration_prior": float(p_cal),
        "deployment_prior": float(ML_DEPLOY_ATTACK_PRIOR),
        "method": "odds_correction",
    }
    calib["prior_correction"] = prior_cfg
    info = {
        **prior_cfg,
        "kind": str(kind),
        "reason": "ok",
    }
    return calib, info

def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def _compress_monotone_knots(x: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep monotone knot pairs with strictly increasing x.
    For repeated x values keep the last (highest p) one.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    if x.size == 0 or p.size == 0 or x.size != p.size:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    p = p[order]
    p = np.maximum.accumulate(p)
    if x.size == 1:
        return x, p
    keep = np.r_[x[1:] > x[:-1], True]
    return x[keep], p[keep]

def fit_ecdf_risk_mapping(
    benign_stats: np.ndarray,
    n_knots: int | None = None,
) -> dict | None:
    """
    Build smooth monotone ECDF-like mapping s_raw -> risk in (0,1)
    from benign-only statistics.
    """
    s = np.asarray(benign_stats, dtype=float).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size < 64:
        return None

    if n_knots is None:
        n_knots = ANOM_ECDF_KNOTS
    try:
        n_knots = int(n_knots)
    except Exception:
        n_knots = ANOM_ECDF_KNOTS
    n_knots = max(64, min(int(s.size), n_knots))

    q = np.linspace(0.0, 1.0, n_knots)
    x = np.quantile(s, q)
    p = np.clip(q, ANOM_PROB_EPS, 1.0 - ANOM_PROB_EPS)
    x, p = _compress_monotone_knots(x, p)
    if x.size < 2:
        return None

    return {
        "method": "ecdf_quantile",
        "x": [float(v) for v in x],
        "p": [float(v) for v in p],
        "n_ref": int(s.size),
        "n_knots": int(x.size),
    }

def _apply_ecdf_risk_mapping(
    stats: np.ndarray,
    mapping: dict | None,
) -> np.ndarray | None:
    if not isinstance(mapping, dict):
        return None
    x_raw = mapping.get("x")
    p_raw = mapping.get("p")
    if not isinstance(x_raw, list) or not isinstance(p_raw, list):
        return None
    if len(x_raw) < 2 or len(p_raw) < 2 or len(x_raw) != len(p_raw):
        return None
    try:
        x = np.asarray(x_raw, dtype=float).reshape(-1)
        p = np.asarray(p_raw, dtype=float).reshape(-1)
    except Exception:
        return None
    x, p = _compress_monotone_knots(x, p)
    if x.size < 2:
        return None
    s = np.asarray(stats, dtype=float).reshape(-1)
    out = np.interp(s, x, p)
    # Linear tail extrapolation avoids a hard constant cap above max benign.
    dx_l = float(x[1] - x[0])
    dx_r = float(x[-1] - x[-2])
    slope_l = float((p[1] - p[0]) / dx_l) if dx_l > 0.0 else 0.0
    slope_r = float((p[-1] - p[-2]) / dx_r) if dx_r > 0.0 else 0.0
    m_l = s < x[0]
    m_r = s > x[-1]
    if np.any(m_l):
        out[m_l] = float(p[0]) + slope_l * (s[m_l] - float(x[0]))
    if np.any(m_r):
        out[m_r] = float(p[-1]) + slope_r * (s[m_r] - float(x[-1]))
    return np.clip(out, ANOM_PROB_EPS, 1.0 - ANOM_PROB_EPS)

def _apply_temperature_scaling(scores: np.ndarray, temperature: float | None) -> np.ndarray:
    s = np.asarray(scores, dtype=float).reshape(-1)
    if s.size == 0:
        return s
    try:
        t = float(temperature)
    except Exception:
        t = 1.0
    if (not np.isfinite(t)) or t <= 1e-6:
        t = 1.0
    s = np.clip(s, 0.0, 1.0)
    if abs(t - 1.0) <= 1e-9:
        return s
    logits = _safe_logit(s, eps=ANOM_PROB_EPS)
    logits = logits / t
    return _sigmoid_stable(logits)

def _sigmoid_from_logit_with_temperature(
    logits: np.ndarray,
    temperature: float | None,
) -> np.ndarray:
    z = np.asarray(logits, dtype=float).reshape(-1)
    try:
        t = float(temperature)
    except Exception:
        t = 1.0
    if (not np.isfinite(t)) or t <= 1e-6:
        t = 1.0
    if abs(t - 1.0) > 1e-9:
        z = z / t
    return _sigmoid_stable(z)

def _binary_logloss(y_true: np.ndarray, prob: np.ndarray, eps: float = 1e-9) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(prob, dtype=float).reshape(-1)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

def _expected_calibration_error(
    y_true: np.ndarray,
    prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(prob, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if y.size == 0:
        return float("nan")

    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, max(2, int(n_bins) + 1))
    ece = 0.0
    n = float(len(y))
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not np.any(m):
            continue
        conf = float(np.mean(p[m]))
        acc = float(np.mean(y[m]))
        ece += float(np.sum(m)) / n * abs(acc - conf)
    return float(ece)

def _raw_anomaly_stat(
    model,
    X: np.ndarray,
    kind: str = "",
) -> np.ndarray:
    """
    Return a raw (pre-calibration) anomaly statistic for a model.
    For IF/OCSVM prefer score_samples (canonical sklearn anomaly score),
    then fallback to decision_function.
    """
    kind_norm = _normalize_anomaly_kind(kind)
    if kind_norm in ("if", "ocsvm") and hasattr(model, "score_samples"):
        try:
            return np.asarray(model.score_samples(X), dtype=float).reshape(-1)
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).reshape(-1)
    if hasattr(model, "score_samples"):
        return np.asarray(model.score_samples(X), dtype=float).reshape(-1)
    raise AttributeError(f"model for kind={kind_norm} has no decision_function/score_samples")

def _aggregate_prob_arrays(
    arrays: list[np.ndarray],
    method: str = "mean_logit",
    eps: float = 1e-6,
) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=float)
    stacked_raw = np.vstack([np.asarray(a, dtype=float).reshape(-1) for a in arrays])
    m = (method or "mean_logit").strip().lower()
    if m in ("max", "or", "max_per_model"):
        return np.clip(np.max(stacked_raw, axis=0), 0.0, 1.0)
    if m in ("mean", "avg", "average"):
        return np.clip(np.mean(stacked_raw, axis=0), 0.0, 1.0)
    # default: mean of logits (less saturation than max)
    stacked = np.clip(stacked_raw, eps, 1.0 - eps)
    logits = np.log(stacked / (1.0 - stacked))
    mean_logits = np.mean(logits, axis=0)
    return _sigmoid_stable(mean_logits)

def fit_temperature_scaler(
    scores: np.ndarray,
    labels: np.ndarray,
    kind: str = "",
    t_min: float = 1.0,
    t_max: float = 8.0,
    t_points: int = 41,
) -> tuple[float, dict]:
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    mask = np.isfinite(s) & np.isfinite(y)
    s = s[mask]
    y = y[mask]

    if s.size == 0 or y.size == 0 or s.size != y.size or len(np.unique(y)) < 2:
        return 1.0, {
            "used": False,
            "reason": "invalid_input",
            "temperature": 1.0,
        }

    s = np.clip(s, 1e-6, 1.0 - 1e-6)
    base_nll = _binary_logloss(y, s)
    base_brier = float(np.mean((s - y) ** 2))

    grid = np.geomspace(t_min, t_max, max(3, int(t_points)))
    grid = np.unique(np.concatenate([grid, np.array([1.0])]))
    best_t = 1.0
    best_nll = base_nll
    best_prob = s

    for t in grid:
        p_t = _apply_temperature_scaling(s, float(t))
        nll_t = _binary_logloss(y, p_t)
        if nll_t < best_nll:
            best_nll = nll_t
            best_t = float(t)
            best_prob = p_t

    best_brier = float(np.mean((best_prob - y) ** 2))
    used = bool(best_nll + 1e-6 < base_nll)
    if not used:
        best_t = 1.0
        best_prob = s
        best_nll = base_nll
        best_brier = base_brier

    metrics = {
        "used": used,
        "temperature": float(best_t),
        "nll_before": float(base_nll),
        "nll_after": float(best_nll),
        "brier_before": float(base_brier),
        "brier_after": float(best_brier),
        "rows": int(len(y)),
    }
    logger.info(
        "[retrain][temp] kind=%s used=%s T=%.4f nll_before=%.6f nll_after=%.6f",
        kind,
        used,
        float(best_t),
        float(base_nll),
        float(best_nll),
    )
    return float(best_t), metrics


def want_nlp() -> bool:
    """
    Returns True if TRAIN_TARGET is "all" or "nlp".
    """
    return TRAIN_TARGET in ("all", "nlp")


def want_ml() -> bool:
    """
    Returns True if TRAIN_TARGET is "all" or "ml".
    """
    return TRAIN_TARGET in ("all", "ml")

def ensure_nlp_data(path: str):
    """
    Ensures that NLP training data exists at the given path.
    
    Args:
        path (str): Path to the NLP training data CSV.
    """
    p = Path(path)
    if p.exists():
        logger.info(f"[retrain] NLP data found: {p}")
        return
    logger.info(f"[retrain] NLP data NOT found, creating synthetic: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("SELECT * FROM users WHERE '1'='1'", 1),
        ("<script>alert(1)</script>", 1),
        ("Normal login request user=anna", 0),
        ("Search query: status=active", 0),
        ("UNION SELECT password FROM accounts", 1),
        ("GET /healthz OK", 0),
    ]
    pd.DataFrame(rows, columns=["text","label"]).to_csv(p, index=False)

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing in dataset: {missing}")
    return df[cols].copy()

def write_if_meta(
    model_path: Path,
    feature_order: List[str],
    model_obj,
    calibration: dict | None = None,
    metrics: dict | None = None,
    recommended_threshold: float | None = None,
):
    """
    Meta for IsolationForest:

      - feature_order: feature order (so infer knows how to assemble vector),
      - calibration: {"mu": ..., "sigma": ...} for decision_function,
      - metrics: offline metrics on CSIC label (if any).
      
    Args:
        model_path (Path): Path to the saved model.
        feature_order (List[str]): Order of features used in training.
        model_obj: Trained IsolationForest model object.
        calibration (dict | None): Calibration parameters for decision_function.
        metrics (dict | None): Offline metrics on CSIC label.
    Returns:
        None
    """
    meta_path = model_path.with_suffix(".meta.json")
    meta = {
        "model": "IsolationForest",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_order": feature_order,
        "n_features_in_": int(getattr(model_obj, "n_features_in_", len(feature_order))),
        "params": {
            "n_estimators": int(getattr(model_obj, "n_estimators", 0)),
            "contamination": getattr(model_obj, "contamination", "auto"),
        },
    }
    if calibration:
        meta["calibration"] = calibration
    if metrics:
        meta["metrics"] = metrics
    if recommended_threshold is not None:
        meta["recommended_threshold"] = float(recommended_threshold)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"[retrain] wrote IF meta -> {meta_path}")

def write_ocsvm_meta(
    outp,
    feature_order,
    model,
    calibration: dict | None = None,
    metrics: dict | None = None,
    recommended_threshold: float | None = None,
):
    """
    Meta for OneClassSVM:
      - feature_order: feature order (so infer knows how to assemble vector),
      - calibration: {"mu": ..., "sigma": ...} for decision_function,
      - metrics: offline metrics on CSIC label (if any).
      
    Args:
        outp (Path): Path to the saved model.
        feature_order (list[str]): Order of features used in training.
        model: Trained OneClassSVM model object.
        calibration (dict | None): Calibration parameters for decision_function.
        metrics (dict | None): Offline metrics on CSIC label.
    """
    oc_est = model
    try:
        # Pipeline(scaler -> OneClassSVM) is used by default; extract estimator.
        if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict):
            if "ocsvm" in model.named_steps:
                oc_est = model.named_steps["ocsvm"]
    except Exception:
        oc_est = model

    meta = {
        "model": "OneClassSVM",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_order": feature_order,
        "n_features_in_": int(getattr(model, "n_features_in_", len(feature_order))),
        "params": {
            "kernel": getattr(oc_est, "kernel", None),
            "nu": float(getattr(oc_est, "nu", 0.0)),
            "gamma": getattr(oc_est, "gamma", None),
            "scaler": str(os.getenv("OCSVM_SCALER", "robust")).strip().lower(),
        },
        "calibration": calibration or {},
        "metrics": metrics or {},
    }
    if recommended_threshold is not None:
      meta["recommended_threshold"] = float(recommended_threshold)
    meta_path = outp.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"[retrain] wrote OCSVM meta -> {meta_path}")

def write_ae_meta(
    outp: Path,
    feature_order: list[str],
    input_dim: int,
    model,
    calibration: dict | None = None,
    metrics: dict | None = None,
    recommended_threshold: float | None = None,
):
    """
    Meta for DenseAutoencoder:
      - feature_order: feature order (so infer knows how to assemble vector),
      - input_dim: input dimension,
      - calibration: {"mu": ..., "sigma": ...} for reconstruction error,
      - metrics: offline metrics on CSIC label (if any).
      
    Args:
        outp (Path): Path to the saved model.
        feature_order (list[str]): Order of features used in training.
        input_dim (int): Input dimension of the autoencoder.
        model: Trained autoencoder model object.
        calibration (dict | None): Calibration parameters for reconstruction error.
        metrics (dict | None): Offline metrics on CSIC label.
    """
    meta = {
        "model": "DenseAutoencoder",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_order": feature_order,
        "input_dim": int(input_dim),
        "n_params": int(model.count_params()),
        "layers": [layer.__class__.__name__ for layer in getattr(model, "layers", [])],
    }
    if calibration:
        meta["calibration"] = calibration
    if metrics:
        meta["metrics"] = metrics
    if recommended_threshold is not None:
        meta["recommended_threshold"] = float(recommended_threshold)
    meta_path = outp.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"[retrain] wrote AE meta -> {meta_path}")


def write_lstm_ae_meta(
    outp: Path,
    feature_order: list[str],
    timesteps: int,
    model,
    calibration: dict | None = None,
    metrics: dict | None = None,
    recommended_threshold: float | None = None,
):
    """
    Meta for LSTMAutoencoder:
      - feature_order: feature order (so infer knows how to assemble vector),
      - timesteps: number of timesteps,
      - calibration: {"mu": ..., "sigma": ...} for reconstruction error,
      - metrics: offline metrics on CSIC label (if any).
      
    Args:
        outp (Path): Path to the saved model.
        feature_order (list[str]): Order of features used in training.
        timesteps (int): Number of timesteps used in training.
        model: Trained LSTM autoencoder model object.
        calibration (dict | None): Calibration parameters for reconstruction error.
        metrics (dict | None): Offline metrics on CSIC label.
    """
    meta = {
        "model": "LSTMAutoencoder",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_order": feature_order,
        "timesteps": int(timesteps),
        "n_params": int(model.count_params()),
        "layers": [layer.__class__.__name__ for layer in getattr(model, "layers", [])],
    }
    if calibration:
        meta["calibration"] = calibration
    if metrics:
        meta["metrics"] = metrics
    if recommended_threshold is not None:
        meta["recommended_threshold"] = float(recommended_threshold)
    meta_path = outp.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"[retrain] wrote LSTM AE meta -> {meta_path}")


def write_supervised_meta(
    outp: Path,
    feature_order: list[str],
    model_name: str,
    estimator,
    calibration: dict | None = None,
    metrics: dict | None = None,
    recommended_threshold: float | None = None,
    train_rows: int | None = None,
    calibration_rows: int | None = None,
    calibration_source: str | None = None,
):
    meta = {
        "model": str(model_name),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_order": feature_order,
        "n_features_in_": int(getattr(estimator, "n_features_in_", len(feature_order))),
        "params": {},
    }
    try:
        params = estimator.get_params(deep=False)
        meta["params"] = {str(k): v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}
    except Exception:
        pass
    if calibration:
        meta["calibration"] = calibration
    if metrics:
        meta["metrics"] = metrics
    if recommended_threshold is not None:
        meta["recommended_threshold"] = float(recommended_threshold)
    if train_rows is not None or calibration_rows is not None or calibration_source is not None:
        meta["data_roles"] = {
            "supervised_train_rows": int(train_rows or 0),
            "calibration_rows": int(calibration_rows or 0),
            "calibration_source": str(calibration_source or ""),
        }

    meta_path = outp.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info("[retrain] wrote supervised meta -> %s", meta_path)


def _save_keras_training_history(
    history_obj,
    out_prefix: str,
    title_prefix: str,
) -> dict:
    """
    Save Keras fit() history to:
      - <out_prefix>_history.csv
      - <out_prefix>_loss.png (loss/val_loss curves, when available)
    """
    hist = getattr(history_obj, "history", None)
    if not isinstance(hist, dict) or not hist:
        return {}

    lengths = [len(v) for v in hist.values() if isinstance(v, list)]
    if not lengths:
        return {}
    epochs = int(max(lengths))
    if epochs <= 0:
        return {}

    rows = []
    for i in range(epochs):
        row = {"epoch": i + 1}
        for k, vals in hist.items():
            if not isinstance(vals, list):
                continue
            row[k] = float(vals[i]) if i < len(vals) and vals[i] is not None else np.nan
        rows.append(row)

    out_csv = f"{out_prefix}_history.csv"
    out_png = f"{out_prefix}_loss.png"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_hist = pd.DataFrame(rows)
    df_hist.to_csv(out_csv, index=False)

    has_loss = "loss" in df_hist.columns
    has_val_loss = "val_loss" in df_hist.columns
    if has_loss or has_val_loss:
        try:
            plt.figure()
            if has_loss:
                plt.plot(df_hist["epoch"], df_hist["loss"], label="loss")
            if has_val_loss:
                plt.plot(df_hist["epoch"], df_hist["val_loss"], label="val_loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f"{title_prefix} training loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()
        except Exception as e:
            logger.info("[retrain][%s] loss plot save failed: %s", title_prefix, e)
            out_png = ""

    out = {
        "history_csv": out_csv,
        "loss_plot_png": out_png if out_png else None,
        "epochs": epochs,
    }
    if has_loss:
        s = df_hist["loss"].dropna()
        out["final_loss"] = float(s.iloc[-1]) if not s.empty else None
    if has_val_loss:
        s = df_hist["val_loss"].dropna()
        out["final_val_loss"] = float(s.iloc[-1]) if not s.empty else None
        out["best_val_loss"] = float(s.min()) if not s.empty else None
    return out


def _resolve_threshold_selection_from_meta(meta: dict) -> dict:
    if not isinstance(meta, dict):
        return {}
    metrics = meta.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    candidates = [
        metrics.get("val", {}).get("threshold_selection") if isinstance(metrics.get("val"), dict) else None,
        metrics.get("threshold_selection"),
        metrics.get("primary", {}).get("threshold_selection") if isinstance(metrics.get("primary"), dict) else None,
        metrics.get("aux_novelty", {}).get("threshold_selection") if isinstance(metrics.get("aux_novelty"), dict) else None,
    ]
    for c in candidates:
        if isinstance(c, dict) and c:
            return c
    return {}


def _figure_meta_for_name(fig_name: str) -> tuple[str, str]:
    """
    Map figure filename to (model_key, meta_path).
    """
    name = fig_name.lower()
    mapping = [
        ("sup_hgb", "/models/ml/sup_hgb_model.meta.json"),
        ("sup_rf", "/models/ml/sup_rf_model.meta.json"),
        ("sup_et", "/models/ml/sup_et_model.meta.json"),
        ("sup_lr", "/models/ml/sup_lr_model.meta.json"),
        ("lstm_ae", "/models/ml/lstm_ae_model.meta.json"),
        ("ocsvm", "/models/ml/ocsvm_model.meta.json"),
        ("if", "/models/ml/if_model.meta.json"),
        ("ae", "/models/ml/ae_model.meta.json"),
        ("ml_aux_novelty", "/models/ml/ml_ensemble.meta.json"),
        ("ml_ensemble", "/models/ml/ml_ensemble.meta.json"),
    ]
    for key, meta_path in mapping:
        if name.startswith(f"{key}_") or name == f"{key}.png":
            return key, meta_path
    return "unknown", ""


def build_figures_manifest(
    models_dir: str = "/models/ml",
    out_csv: str = "/models/ml/figures_manifest.csv",
) -> dict:
    """
    Build manifest CSV for all PNG figures in models/ml with:
      - axes,
      - dataset role inferred from filename,
      - threshold policy/target/selected from model metadata.
    """
    pngs = sorted(Path(models_dir).glob("*.png"))
    rows = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _figure_kind(name: str) -> str:
        if "_hist" in name:
            return "histogram"
        if "_roc" in name:
            return "roc"
        if "_pr" in name:
            return "pr"
        if "_loss" in name:
            return "loss_curve"
        return "other"

    def _dataset_role(name: str) -> str:
        if "benign_vs_attack" in name:
            return "validation(benign_vs_attack)"
        if "_mu_sigma_" in name:
            return "validation(mu_sigma)"
        if "_eval_" in name:
            return "validation(runtime)"
        if "_loss" in name:
            return "train(keras_validation_split_0.1)"
        return "unknown"

    for p in pngs:
        fig_name = p.name
        model_key, meta_path = _figure_meta_for_name(fig_name)
        kind = _figure_kind(fig_name)
        role = _dataset_role(fig_name)

        policy = ""
        target_fpr = None
        selected_thr = None
        rec_thr = None
        if meta_path and Path(meta_path).exists():
            try:
                meta = json.loads(Path(meta_path).read_text())
                thr_info = _resolve_threshold_selection_from_meta(meta)
                policy = str(thr_info.get("policy", "") or "")
                if isinstance(thr_info.get("target_fpr"), (int, float)):
                    target_fpr = float(thr_info.get("target_fpr"))
                if isinstance(thr_info.get("selected_threshold"), (int, float)):
                    selected_thr = float(thr_info.get("selected_threshold"))
                if isinstance(meta.get("recommended_threshold"), (int, float)):
                    rec_thr = float(meta.get("recommended_threshold"))
                elif isinstance(meta.get("recommended_aux_novelty_threshold"), (int, float)):
                    rec_thr = float(meta.get("recommended_aux_novelty_threshold"))
            except Exception:
                pass

        if kind == "histogram":
            x_axis, y_axis = "calibrated anomaly risk score", "count"
        elif kind == "roc":
            x_axis, y_axis = "false positive rate", "true positive rate"
        elif kind == "pr":
            x_axis, y_axis = "recall", "precision"
        elif kind == "loss_curve":
            x_axis, y_axis = "epoch", "loss / val_loss"
        else:
            x_axis, y_axis = "", ""

        caption = (
            f"{model_key}: {kind}; X={x_axis}; Y={y_axis}; "
            f"dataset={role}; threshold_policy={policy or 'NA'}"
        )
        if target_fpr is not None:
            caption += f", target_fpr={target_fpr:.4f}"
        if selected_thr is not None:
            caption += f", selected_threshold={selected_thr:.6f}"
        elif rec_thr is not None:
            caption += f", recommended_threshold={rec_thr:.6f}"

        rows.append(
            {
                "generated_at_utc": ts,
                "figure_path": str(p),
                "figure_name": fig_name,
                "model_key": model_key,
                "figure_kind": kind,
                "dataset_role": role,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "threshold_policy": policy,
                "target_fpr": target_fpr,
                "selected_threshold": selected_thr,
                "recommended_threshold": rec_thr,
                "model_meta_path": meta_path,
                "caption_short": caption,
            }
        )

    out_df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    logger.info("[retrain] figures manifest saved -> %s (rows=%s)", out_csv, len(out_df))
    return {"path": out_csv, "rows": int(len(out_df))}


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int).reshape(-1)
    n = int(y.size)
    if n == 0:
        return np.asarray([], dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos <= 0 or n_neg <= 0:
        return np.ones(n, dtype=float)
    w_pos = float(n) / (2.0 * float(n_pos))
    w_neg = float(n) / (2.0 * float(n_neg))
    return np.where(y == 1, w_pos, w_neg).astype(float)


def _supervised_raw_scores(estimator, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        p = np.asarray(estimator.predict_proba(X), dtype=float)
        if p.ndim == 2 and p.shape[1] >= 2:
            return np.clip(p[:, 1], ANOM_PROB_EPS, 1.0 - ANOM_PROB_EPS)
        if p.ndim == 1:
            return np.clip(p, ANOM_PROB_EPS, 1.0 - ANOM_PROB_EPS)
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(X), dtype=float).reshape(-1)
        return np.clip(_sigmoid_stable(raw), ANOM_PROB_EPS, 1.0 - ANOM_PROB_EPS)
    if hasattr(estimator, "predict"):
        pred = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
        return np.clip(pred, ANOM_PROB_EPS, 1.0 - ANOM_PROB_EPS)
    raise AttributeError("estimator has no predict_proba/decision_function/predict")


def _fit_estimator_with_optional_weights(estimator, X: np.ndarray, y: np.ndarray) -> None:
    y = np.asarray(y, dtype=int).reshape(-1)
    sw = _balanced_sample_weight(y)
    fit_kwargs_candidates = [{"sample_weight": sw}]
    if isinstance(estimator, Pipeline) and estimator.steps:
        last_step_name = str(estimator.steps[-1][0])
        fit_kwargs_candidates.insert(0, {f"{last_step_name}__sample_weight": sw})
    for fit_kwargs in fit_kwargs_candidates:
        try:
            estimator.fit(X, y, **fit_kwargs)
            return
        except (TypeError, ValueError):
            continue
    estimator.fit(X, y)


def _split_supervised_train_and_calibration(
    df_mixed: pd.DataFrame,
    train_fraction: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = _labels_from_df(df_mixed)
    if y is None or len(np.unique(y)) < 2:
        raise ValueError("mixed dataset must contain both classes for supervised split")

    idx = np.arange(len(df_mixed))
    strat = y if len(np.unique(y)) > 1 else None
    idx_tr, idx_ca = train_test_split(
        idx,
        train_size=float(train_fraction),
        random_state=int(random_state),
        stratify=strat,
    )
    df_train = df_mixed.iloc[np.sort(idx_tr)].reset_index(drop=True)
    df_calib = df_mixed.iloc[np.sort(idx_ca)].reset_index(drop=True)

    y_tr = _labels_from_df(df_train)
    y_ca = _labels_from_df(df_calib)
    if y_tr is None or y_ca is None or len(np.unique(y_tr)) < 2 or len(np.unique(y_ca)) < 2:
        raise ValueError("supervised split produced a single-class partition; adjust train_fraction")
    return df_train, df_calib


def prepare_supervised_train_and_calibration_paths(
    calib_path: str,
    feature_order: list[str],
) -> tuple[str | None, str, dict]:
    """
    Prepare data roles for supervised numeric baselines:
      - supervised_train: mixed labels, used to fit classifiers
      - calibration_holdout: mixed labels, used only for calibration/threshold
    """
    required_cols = list(feature_order) + ["label"]

    if ML_SUPERVISED_TRAIN_FILE:
        p_sup = Path(ML_SUPERVISED_TRAIN_FILE)
        if p_sup.exists():
            df_head = pd.read_csv(p_sup, nrows=5, low_memory=False)
            have = set(df_head.columns.tolist())
            need = set(required_cols)
            if not need.issubset(have):
                raise SystemExit(
                    f"[retrain][supervised] file {p_sup} missing required columns "
                    f"{sorted(need - have)}"
                )
            rows, pos, neg = _scan_binary_label_distribution(
                str(p_sup),
                label_col="label",
                max_rows=ML_LABEL_CHECK_MAX_ROWS,
            )
            if rows <= 0 or pos <= 0 or neg <= 0:
                raise SystemExit(
                    f"[retrain][supervised] file {p_sup} must contain BOTH classes; "
                    f"got rows={rows}, pos={pos}, neg={neg}"
                )
            info = {
                "mode": "external_supervised_train",
                "supervised_train_path": str(p_sup),
                "calibration_path": str(calib_path),
            }
            logger.info(
                "[retrain][supervised] using external supervised train=%s calibration=%s",
                p_sup,
                calib_path,
            )
            return str(p_sup), str(calib_path), info
        logger.info(
            "[retrain][supervised] ML_SUPERVISED_TRAIN_FILE=%s not found; fallback to calibration split",
            p_sup,
        )

    df_calib_all = pd.read_csv(calib_path, low_memory=False)
    have = set(df_calib_all.columns.tolist())
    need = set(required_cols)
    if not need.issubset(have):
        raise SystemExit(
            f"[retrain][supervised] calibration file {calib_path} missing required columns "
            f"{sorted(need - have)}"
        )
    y_all = _labels_from_df(df_calib_all)
    if y_all is None or len(np.unique(y_all)) < 2:
        raise SystemExit(
            "[retrain][supervised] calibration file must contain both classes for supervised split"
        )

    df_train, df_calib_holdout = _split_supervised_train_and_calibration(
        df_calib_all,
        train_fraction=ML_SUPERVISED_TRAIN_FRACTION,
        random_state=ML_VAL_TEST_RANDOM_STATE,
    )
    y_train = _labels_from_df(df_train)
    df_train, y_train = _maybe_downsample_df(
        df_train,
        y_train,
        max_rows=max(0, int(ML_SUP_MAX_TRAIN_ROWS)),
        tag="supervised/train",
        random_state=ML_VAL_TEST_RANDOM_STATE,
    )
    if "label" in df_train.columns and y_train is not None and len(y_train) == len(df_train):
        df_train["label"] = y_train

    ML_SUPERVISED_TRAIN_SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ML_SUPERVISED_CALIB_SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(ML_SUPERVISED_TRAIN_SPLIT_FILE, index=False)
    df_calib_holdout.to_csv(ML_SUPERVISED_CALIB_SPLIT_FILE, index=False)

    info = {
        "mode": "split_from_calibration",
        "supervised_train_path": str(ML_SUPERVISED_TRAIN_SPLIT_FILE),
        "calibration_path": str(ML_SUPERVISED_CALIB_SPLIT_FILE),
        "supervised_train_rows": int(len(df_train)),
        "calibration_rows": int(len(df_calib_holdout)),
        "supervised_train_fraction": float(ML_SUPERVISED_TRAIN_FRACTION),
    }
    logger.info(
        "[retrain][supervised] split calibration -> train=%s (%s rows), calibration=%s (%s rows)",
        ML_SUPERVISED_TRAIN_SPLIT_FILE,
        len(df_train),
        ML_SUPERVISED_CALIB_SPLIT_FILE,
        len(df_calib_holdout),
    )
    return str(ML_SUPERVISED_TRAIN_SPLIT_FILE), str(ML_SUPERVISED_CALIB_SPLIT_FILE), info


def train_supervised_numeric_baseline(
    model_kind: str,
    model_name: str,
    estimator,
    train_csv_path: str,
    calib_csv_path: str,
    test_csv_path: str | None,
    out_path: str,
    feature_order: list[str],
):
    df_tr = pd.read_csv(train_csv_path, low_memory=False)
    X_tr = ensure_columns(df_tr, feature_order).fillna(0.0).astype(float).to_numpy()
    y_tr = _labels_from_df(df_tr)
    if y_tr is None or len(np.unique(y_tr)) < 2:
        raise ValueError(f"[retrain][{model_kind}] supervised train must contain both classes")

    _fit_estimator_with_optional_weights(estimator, X_tr, y_tr)

    df_cal = pd.read_csv(calib_csv_path, low_memory=False)
    X_cal = ensure_columns(df_cal, feature_order).fillna(0.0).astype(float).to_numpy()
    y_cal = _labels_from_df(df_cal)
    if y_cal is None or len(np.unique(y_cal)) < 2:
        raise ValueError(f"[retrain][{model_kind}] calibration must contain both classes")

    raw_prob = _supervised_raw_scores(estimator, X_cal)
    raw_prob = np.clip(raw_prob, ANOM_PROB_EPS, 1.0 - ANOM_PROB_EPS)

    candidates: list[dict] = []
    raw_calib = {"orientation": "identity"}
    raw_scores = np.asarray(raw_prob, dtype=float).reshape(-1)
    raw_temp_t, raw_temp_metrics = fit_temperature_scaler(raw_scores, y_cal, kind=f"{model_kind}_raw")
    if raw_temp_metrics.get("used"):
        raw_calib["temperature"] = float(raw_temp_t)
        raw_scores = _apply_temperature_scaling(raw_scores, raw_temp_t)
    candidates.append(
        {
            "name": "raw_prob",
            "calibration": dict(raw_calib),
            "scores": raw_scores,
            "eval": _eval_scores_only_with_labels(raw_scores, y_cal),
            "temperature_scaling": raw_temp_metrics,
        }
    )

    logit_params, logit_metrics, _ = fit_attack_logistic_calibrator(
        raw_prob,
        y_cal,
        kind=model_kind,
        orientation="identity",
        standardize_input=True,
    )
    if logit_params is not None:
        platt_calib = {"orientation": "identity", "attack_logistic": dict(logit_params)}
        platt_scores = score_from_stat_with_calib(raw_prob, platt_calib, kind=model_kind)
        platt_temp_t, platt_temp_metrics = fit_temperature_scaler(platt_scores, y_cal, kind=f"{model_kind}_platt")
        if platt_temp_metrics.get("used"):
            platt_calib["temperature"] = float(platt_temp_t)
            platt_scores = _apply_temperature_scaling(platt_scores, platt_temp_t)
        candidates.append(
            {
                "name": "platt",
                "calibration": dict(platt_calib),
                "scores": platt_scores,
                "eval": _eval_scores_only_with_labels(platt_scores, y_cal),
                "temperature_scaling": platt_temp_metrics,
                "attack_logistic": logit_metrics,
            }
        )

    best_candidate, comparison, selection_reason = _select_best_runtime_calibration_candidate(candidates)
    calib = dict(best_candidate.get("calibration") or {})
    scores_cal = np.asarray(best_candidate.get("scores"), dtype=float).reshape(-1)
    temp_metrics = dict(best_candidate.get("temperature_scaling") or {"used": False, "temperature": 1.0})

    calib, prior_info = _attach_prior_correction_to_calibration(calib, y_cal, kind=model_kind)
    scores_cal = score_from_stat_with_calib(raw_prob, calib, kind=model_kind)

    eval_runtime = _eval_and_plot_scores_with_labels(
        scores_cal,
        y_cal,
        out_prefix=str(Path(out_path).with_suffix("")) + "_eval",
        title_prefix=f"{model_name}(runtime)",
    )
    recommended_threshold, thr_info = choose_operating_threshold(y_cal, scores_cal)
    val_metrics = {
        **metrics_on_threshold(y_cal, scores_cal, recommended_threshold),
        "best_f1_on_val": float((eval_runtime or {}).get("best_f1", 0.0)),
        "recommended_threshold": float(recommended_threshold),
        "threshold_selection": thr_info,
        "calibrated_eval": eval_runtime,
        "calibration_selection": {
            "selected": best_candidate.get("name"),
            "reason": selection_reason,
            "candidates": comparison,
        },
        "temperature_scaling": temp_metrics,
        "prior_correction": prior_info,
        "attack_logistic": logit_metrics,
    }

    metrics = {
        "train_rows": int(len(df_tr)),
        "calibration_rows": int(len(df_cal)),
        "val": val_metrics,
    }

    if (
        RETRAIN_ENABLE_TEST_EVAL
        and test_csv_path
        and Path(test_csv_path).exists()
        and recommended_threshold is not None
    ):
        try:
            df_te = pd.read_csv(test_csv_path, low_memory=False)
            y_te = _labels_from_df(df_te)
            if y_te is not None and len(np.unique(y_te)) >= 2:
                df_te, y_te = _maybe_downsample_df(
                    df_te,
                    y_te,
                    ML_TEST_MAX_ROWS,
                    tag=f"{model_kind}/test",
                    random_state=43,
                )
                X_te = ensure_columns(df_te, feature_order).fillna(0.0).astype(float).to_numpy()
                raw_te = _supervised_raw_scores(estimator, X_te)
                score_te = score_from_stat_with_calib(raw_te, calib, kind=model_kind)
                metrics["test"] = metrics_on_threshold(y_te, score_te, recommended_threshold)
            else:
                logger.info("[retrain][%s] test missing both classes; skip test metrics", model_kind)
        except Exception as e:
            logger.info("[retrain][%s] test eval failed: %s", model_kind, e)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, outp)
    logger.info("[retrain] saved %s -> %s", model_name, outp)

    write_supervised_meta(
        outp=outp,
        feature_order=feature_order,
        model_name=model_name,
        estimator=estimator,
        calibration=calib,
        metrics=metrics,
        recommended_threshold=recommended_threshold,
        train_rows=len(df_tr),
        calibration_rows=len(df_cal),
        calibration_source=str(calib_csv_path),
    )
    return estimator, calib, recommended_threshold, metrics


def supervised_baseline_specs() -> list[tuple[str, str, object, str]]:
    return [
        (
            "sup_lr",
            "LogisticRegression(StandardScaler)",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=8000, solver="liblinear", random_state=42)),
                ]
            ),
            SUP_LR_OUT,
        ),
        (
            "sup_hgb",
            "HistGradientBoostingClassifier",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=None,
                max_iter=max(50, int(ML_SUP_HGB_MAX_ITER)),
                random_state=42,
            ),
            SUP_HGB_OUT,
        ),
        (
            "sup_rf",
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=max(50, int(ML_SUP_RF_N_ESTIMATORS)),
                random_state=42,
                n_jobs=-1,
            ),
            SUP_RF_OUT,
        ),
        (
            "sup_et",
            "ExtraTreesClassifier",
            ExtraTreesClassifier(
                n_estimators=max(50, int(ML_SUP_ET_N_ESTIMATORS)),
                random_state=42,
                n_jobs=-1,
            ),
            SUP_ET_OUT,
        ),
    ]

def _scan_binary_label_distribution(
    path: str,
    label_col: str = "label",
    max_rows: int = 0,
    chunksize: int = 200_000,
) -> tuple[int, int, int]:
    rows = 0
    pos = 0
    neg = 0

    for chunk in pd.read_csv(path, usecols=[label_col], chunksize=chunksize, low_memory=False):
        y = pd.to_numeric(chunk[label_col], errors="coerce").fillna(0.0).to_numpy()
        y = (y > 0.0).astype(int)
        if max_rows > 0 and rows + len(y) > max_rows:
            y = y[: max_rows - rows]
        rows += int(len(y))
        pos += int(y.sum())
        neg += int(len(y) - y.sum())
        if max_rows > 0 and rows >= max_rows:
            break

    return rows, pos, neg

def ensure_ml_num_data(path: str, required_cols: List[str]):
    """
    Ensures that ML numeric data exists at the given path and has required columns.
    
    Args:
        path (str): Path to the ML numeric data CSV.
        required_cols (List[str]): List of required column names.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"[retrain] ML numeric file not found: {p}. "
            f"Firstly generate it using build_ml_numeric_*.py (attacklab, csic, ...)"
        )

    try:
        df = pd.read_csv(p, nrows=5, low_memory=False)
    except Exception as e:
        raise SystemExit(f"[retrain] ML numeric exists but unreadable ({p}): {e}")

    have = set(df.columns.tolist())
    need = set(required_cols)
    if not need.issubset(have):
        raise SystemExit(
            f"[retrain] ML numeric {p} has wrong columns. "
            f"Need at least: {sorted(need)}, have: {sorted(have)}"
        )

    logger.info(f"[retrain] ML numeric OK: {p}, columns={sorted(have)}")
    if "label" in have:
        rows, pos, neg = _scan_binary_label_distribution(
            str(p),
            label_col="label",
            max_rows=ML_LABEL_CHECK_MAX_ROWS,
        )
        if rows > 0:
            pos_frac = float(pos) / float(rows)
            logger.info(
                "[retrain] ML numeric label distribution: rows=%s pos=%s neg=%s pos_frac=%.6f",
                rows, pos, neg, pos_frac,
            )
            if pos_frac > ML_NUM_MAX_ATTACK_FRAC:
                raise SystemExit(
                    f"[retrain] ML numeric expected benign-only, but pos_frac={pos_frac:.6f} "
                    f"> ML_NUM_MAX_ATTACK_FRAC={ML_NUM_MAX_ATTACK_FRAC:.6f}"
                )

def ensure_ml_validation_data(path: str, required_cols: List[str]):
    """
    Ensures that ML validation data exists at the given path and has required columns.

    Args:
        path (str): Path to the ML validation data CSV.
        required_cols (List[str]): List of required column names.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"[retrain] ML validation file not found: {p}. "
            f"Firstly generate it using build_ml_numeric_*.py (attacklab, csic, ...)"
        )

    try:
        df = pd.read_csv(p, nrows=5, low_memory=False)
    except Exception as e:
        raise SystemExit(f"[retrain] ML validation exists but unreadable ({p}): {e}")

    have = set(df.columns.tolist())
    need = set(required_cols)
    if not need.issubset(have):
        raise SystemExit(
            f"[retrain] ML validation {p} has wrong columns. "
            f"Need at least: {sorted(need)}, have: {sorted(have)}"
        )
    if ML_SOURCE_HOLDOUT_ENABLED and ML_SOURCE_HOLDOUT_COLUMN and ML_SOURCE_HOLDOUT_COLUMN not in have:
        raise SystemExit(
            f"[retrain] source holdout is enabled, but ML validation {p} has no "
            f"'{ML_SOURCE_HOLDOUT_COLUMN}' column. Rebuild ml_validation via build_ml_numeric_*.py."
        )

    logger.info(f"[retrain] ML validation OK: {p}, columns={sorted(have)}")
    if "label" in have:
        rows, pos, neg = _scan_binary_label_distribution(
            str(p),
            label_col="label",
            max_rows=ML_LABEL_CHECK_MAX_ROWS,
        )
        logger.info(
            "[retrain] ML validation label distribution: rows=%s pos=%s neg=%s",
            rows, pos, neg,
        )
        if rows <= 0 or pos <= 0 or neg <= 0:
            raise SystemExit(
                f"[retrain] ML validation must contain BOTH classes for calibration; "
                f"got rows={rows}, pos={pos}, neg={neg}"
            )

def _norm_text(s: str) -> str:
    """
    Normalizes text by URL-decoding, HTML-unescaping, and cleaning.
    
    Args:
        s (str): Input string.
    Returns:
        str: Normalized string.
    """
    if not s:
        return ""
    try:
        s = urllib.parse.unquote_plus(s)
    except Exception:
        pass
    s = html.unescape(s)
    s = s.replace("\\u0027", "'").replace("\\u003d", "=").replace("\\'", "'").replace('\\"', '"')
    return " ".join(s.split()).lower()

def _compile_rules(rules_json: dict):
    """
    Compiles rules from JSON into regex patterns and tokens.
    
    Args:
        rules_json (dict): Rules in JSON format.
    Returns:
        dict: Compiled rules.
    """
    cfg = rules_json.get("config", {}) or {}
    attacks = rules_json.get("attack_patterns", {}) or {}

    compiled = {}
    for atk, grp in attacks.items():
        strong = grp.get("strong", []) or []
        partial = grp.get("partial_tokens", []) or []
        min_hits = int(grp.get("min_hits", 1))
        div = max(1, int(grp.get("partial_norm_divisor", 3)))

        match_fields = grp.get("match_fields") or []
        requires_rule_hit = bool(grp.get("requires_rule_hit", False))

        strong_res = []
        for pat in strong:
            try:
                strong_res.append(re.compile(pat))
            except Exception:
                # ignore bad regex
                continue

        compiled[atk] = {
            "strong_res": strong_res,
            "partial_tokens": [t.lower() for t in partial],
            "min_hits": min_hits,
            "divisor": div,
            "match_fields": match_fields,
            "requires_rule_hit": requires_rule_hit,
        }

    return compiled

def _scan_blob(blob: str, compiled_rules: dict) -> dict:
    """
    Scans the input blob against compiled rules to generate signals.
    
    Args:
        blob (str): Input text blob.
        compiled_rules (dict): Compiled rules.
    Returns:
        dict: Signals for each attack type.
    """
    if isinstance(blob, dict):
        fields_map = {}
        for k, v in blob.items():
            if v is None:
                continue
            fields_map[k] = _norm_text(str(v))
        full_lb = " ".join(fields_map.values())
    else:
        full_lb = _norm_text(str(blob or ""))
        fields_map = {"__all__": full_lb}

    sig = {}
    for atk, grp in compiled_rules.items():
        match_fields = grp.get("match_fields") or []

        if match_fields:
            parts = []
            for name in match_fields:
                v = fields_map.get(name)
                if v:
                    parts.append(v)
            lb = " ".join(parts) if parts else full_lb
        else:
            lb = full_lb

        strong_hit = 0.0
        for rx in grp["strong_res"]:
            try:
                if rx.search(lb):
                    strong_hit = 1.0
                    break
            except Exception:
                continue

        hits = 0
        for tok in grp["partial_tokens"]:
            if tok in lb:
                hits += 1

        base = hits / max(1, grp["divisor"])
        sig[f"{atk}_strong"] = float(1.0 if strong_hit >= 1.0 else 0.0)
        sig[f"{atk}_partial"] = float(min(1.0, base))

    return sig

def _label_from_signals(sig: dict, min_partial: float, compiled_rules: dict) -> dict:
    """
    Generates labels from signals based on a minimum partial threshold.
    
    Args:
        sig (dict): Signals dictionary.
        min_partial (float): Minimum partial threshold.
    Returns:
        dict: Labels dictionary.
    """
    labels = {}
    any_pos = False

    for k in list(sig.keys()):
        if not k.endswith("_strong"):
            continue

        atk = k[:-7]
        strong = float(sig.get(f"{atk}_strong", 0.0))
        partial = float(sig.get(f"{atk}_partial", 0.0))

        cfg = compiled_rules.get(atk) or {}
        requires_rule = bool(cfg.get("requires_rule_hit", False))

        if requires_rule:
            y = 1 if strong >= 1.0 else 0
        else:
            y = 1 if (strong >= 1.0 or partial >= min_partial) else 0

        labels[f"label_{atk}"] = int(y)
        any_pos = any_pos or bool(y)

    labels["label_any"] = 1 if any_pos else 0
    return labels

# ML silver rules helpers
def _char_entropy(s: str) -> float:
    """
    Calculates the character entropy of a string.
    
    Args:
        s (str): Input string.
    Returns:
        float: Character entropy.
    """
    if not s:
        return 0.0
    cnt = collections.Counter(s)
    n = len(s)
    return float(-sum((c/n) * math.log2(c/n) for c in cnt.values()))

def _suspicious_char_ratio(s: str) -> float:
    """
    Calculates the ratio of suspicious characters in a string.
    
    Args:
        s (str): Input string.
    Returns:
        float: Ratio of suspicious characters.
    """
    if not s:
        return 0.0
    sus = sum(ch in "<>\"'`;$(){}[]\\|&" for ch in s)
    return float(sus / max(1, len(s)))

def _digit_ratio(s: str) -> float:
    """
    Calculates the ratio of digit characters in a string.
    
    Args:
        s (str): Input string.
    Returns:
        float: Ratio of digit characters.
    """
    if not s:
        return 0.0
    digs = sum(ch.isdigit() for ch in s)
    return float(digs / max(1, len(s)))

def _punct_ratio(s: str) -> float:
    """
    Calculates the ratio of punctuation characters in a string.
    
    Args:
        s (str): Input string.
    Returns:
        float: Ratio of punctuation characters.
    """
    if not s:
        return 0.0
    p = sum(ch in string.punctuation for ch in s)
    return float(p / max(1, len(s)))

def _token_count(s: str) -> int:
    """
    Counts the number of tokens in a string.
    Args:
        s (str): Input string.
    Returns:
        int: Number of tokens.
    """
    return int(len(s.split())) if s else 0

def _featurize_for_ml_from_text(text: str) -> dict:
    """
    Extracts basic numeric features from text for ML.
    
    Args:
        text (str): Input text.
    Returns:
        dict: Extracted features.
    """
    t = _norm_text(text)
    return {
        "text_len": len(t),
        "char_entropy": _char_entropy(t),
        "suspicious_char_ratio": _suspicious_char_ratio(t),
        "digit_ratio": _digit_ratio(t),
        "punct_ratio": _punct_ratio(t),
        "token_count": _token_count(t),
    }

def build_ml_rules_dataset_from_silver(silver_df: pd.DataFrame, rules_path: Path, out_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Builds ML rules dataset from silver-labeled DataFrame and saves to CSV.
    
    Args:
        silver_df (pd.DataFrame): Silver-labeled DataFrame.
        rules_path (Path): Path to the rules JSON file.
        out_csv (Path): Output CSV path.
    Returns:
        tuple[pd.DataFrame, list[str]]: DataFrame and feature columns list.
    """
    rules_json = json.loads(rules_path.read_text(encoding="utf-8"))
    attacks = list((rules_json.get("attack_patterns") or {}).keys())

    rows = []
    for _, r in silver_df.iterrows():
        t = str(r.get("text", ""))
        feats = _featurize_for_ml_from_text(t)
        # adding rule features in the same format as expected by preprocess
        for atk in attacks:
            feats[f"rule_{atk}_strong"]  = float(r.get(f"{atk}_strong", 0.0))
            feats[f"rule_{atk}_partial"] = float(r.get(f"{atk}_partial", 0.0))

        feats["label_any"] = int(r.get("label_any", 0))
        rows.append(feats)

    out = pd.DataFrame(rows)
    # feature list
    feature_cols = [c for c in out.columns if c != "label_any"]
    out.to_csv(out_csv, index=False)
    return out, feature_cols

def build_silver_csv(csv_in: str, rules_path: Path, csv_out: Path) -> pd.DataFrame:
    """
    Builds silver-labeled CSV from input CSV using rules and saves to output CSV.
    
    Args:
        csv_in (str): Input CSV path.
        rules_path (Path): Path to the rules JSON file.
        csv_out (Path): Output CSV path.
    Returns:
        pd.DataFrame: Silver-labeled DataFrame.
    """
    df = pd.read_csv(csv_in, low_memory=False)
    if "text" not in df.columns:
        raise ValueError("NLP CSV must contain 'text' column for silver labeling")
    rules_json = json.loads(rules_path.read_text(encoding="utf-8"))
    compiled = _compile_rules(rules_json)

    rows = []
    for _, r in df.iterrows():
        row_dict = {}
        for col in df.columns:
            if col.lower() in ("label", "y", "target"):
                continue
            row_dict[col] = r.get(col)
        sig = _scan_blob(row_dict, compiled)
        labs = _label_from_signals(sig, SILVER_MIN_PARTIAL, compiled)
        rows.append({**r.to_dict(), **sig, **labs})
    out_df = pd.DataFrame(rows)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(csv_out, index=False)
    return out_df

def list_attacks_from_rules(rules_path: Path) -> list[str]:
    """
    Lists attack patterns from the rules JSON file.
    
    Args:
        rules_path (Path): Path to the rules JSON file.
    Returns:
        list[str]: List of attack pattern names.
    """
    rules_json = json.loads(rules_path.read_text(encoding="utf-8"))
    return list((rules_json.get("attack_patterns") or {}).keys())
  
# NLP training functions
def train_tfidf_lr(csv_path: str, out_path: str, label_col: str = "label"):
    df = pd.read_csv(csv_path, low_memory=False)
    if not {"text", label_col}.issubset(df.columns):
        raise ValueError(f"NLP CSV must contain columns: text,{label_col}")

    X = df["text"].astype(str)
    y = df[label_col].astype(int).to_numpy()

    if RETRAIN_ENABLE_TEST_EVAL:
        X_tr, y_tr, X_va, y_va, X_te, y_te = split_train_val_test(
            X, y, val_size=0.2, test_size=0.1, random_state=42
        )
    else:
        strat = y if len(np.unique(y)) > 1 else None
        X_tr, X_va, y_tr, y_va = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=strat,
        )
        X_te, y_te = None, None

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=norm_text,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            max_features=50000,
            strip_accents="unicode",
            lowercase=False,
            token_pattern=r"(?u)\b[\w@./-]*'?[\w@./-]*\b",
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            C=4.0,
            fit_intercept=True
        )),
    ])

    pipe.fit(X_tr, y_tr)

    # choose threshold on VAL only
    p_va = pipe.predict_proba(X_va)[:, 1]
    best_thr, best_f1 = best_threshold_f1(y_va, p_va)

    # report metrics on VAL (selection set)
    m_val = metrics_on_threshold(y_va, p_va, best_thr)

    m_test = None
    # final metrics on TEST (optional, untouched)
    if RETRAIN_ENABLE_TEST_EVAL and X_te is not None and y_te is not None:
        p_te = pipe.predict_proba(X_te)[:, 1]
        m_test = metrics_on_threshold(y_te, p_te, best_thr)

    joblib.dump({"pipeline": pipe}, out_path)

    meta = {
        "model": "TFIDF+LogReg (pipeline)",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": {"train": int(len(y_tr)), "val": int(len(y_va)), "test": int(len(y_te)) if y_te is not None else 0},
        "threshold_selected_on": "val",
        "recommended_threshold": float(best_thr),
        "metrics": {"val": {**m_val, "best_f1_on_val": float(best_f1)}},
        "test_eval_enabled": bool(RETRAIN_ENABLE_TEST_EVAL),
        "data_rows": int(df.shape[0]),
        "label_col": label_col,
    }
    if m_test is not None:
        meta["metrics"]["test"] = m_test
    Path(out_path).with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def train_char_sgd(csv_path: str, out_path: str, label_col: str = "label"):
    df = pd.read_csv(csv_path, low_memory=False)
    if not {"text", label_col}.issubset(df.columns):
        raise ValueError(f"NLP CSV must contain columns: text,{label_col}")

    X = df["text"].astype(str)
    y = df[label_col].astype(int).to_numpy()

    if RETRAIN_ENABLE_TEST_EVAL:
        X_tr, y_tr, X_va, y_va, X_te, y_te = split_train_val_test(
            X, y, val_size=0.2, test_size=0.1, random_state=42
        )
    else:
        strat = y if len(np.unique(y)) > 1 else None
        X_tr, X_va, y_tr, y_va = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=strat,
        )
        X_te, y_te = None, None

    # oversample positives on TRAIN only
    X_tr = pd.Series(X_tr).reset_index(drop=True)
    y_tr = pd.Series(y_tr).reset_index(drop=True)

    pos_mask = (y_tr.values == 1)
    n_pos = int(pos_mask.sum()); n_neg = int((~pos_mask).sum())
    if 0 < n_pos < n_neg:
        target_pos = int(0.6 * n_neg)
        reps = max(1, math.ceil((target_pos - n_pos) / max(1, n_pos)))
        reps = min(reps, 8)
        if reps > 1:
            X_pos = X_tr[pos_mask]; y_pos = y_tr[pos_mask]
            X_tr = pd.concat([X_tr] + [X_pos] * (reps - 1), ignore_index=True)
            y_tr = pd.concat([y_tr] + [y_pos] * (reps - 1), ignore_index=True)

    vec = TfidfVectorizer(
        preprocessor=norm_text,
        analyzer="char",
        ngram_range=(2, 6),
        min_df=1,
        max_df=1.0,
        max_features=200_000,
        lowercase=False
    )

    clf = SGDClassifier(
        loss="log_loss",
        alpha=5e-6,
        max_iter=5000,
        random_state=42,
        class_weight="balanced",
        tol=1e-4
    )

    pipe = Pipeline([("tfidf_char", vec), ("clf", clf)])
    pipe.fit(X_tr, y_tr)

    p_va = pipe.predict_proba(pd.Series(X_va))[:, 1]
    best_thr, best_f1 = best_threshold_f1(y_va, p_va)
    m_val = metrics_on_threshold(y_va, p_va, best_thr)

    m_test = None
    if RETRAIN_ENABLE_TEST_EVAL and X_te is not None and y_te is not None:
        p_te = pipe.predict_proba(pd.Series(X_te))[:, 1]
        m_test = metrics_on_threshold(y_te, p_te, best_thr)

    joblib.dump({"pipeline": pipe}, out_path)

    meta = {
        "model": "CharTFIDF+SGD(log_loss)",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": {"train": int(len(y_tr)), "val": int(len(y_va)), "test": int(len(y_te)) if y_te is not None else 0},
        "threshold_selected_on": "val",
        "recommended_threshold": float(best_thr),
        "metrics": {"val": {**m_val, "best_f1_on_val": float(best_f1)}},
        "test_eval_enabled": bool(RETRAIN_ENABLE_TEST_EVAL),
        "data_rows": int(df.shape[0]),
        "label_col": label_col,
    }
    if m_test is not None:
        meta["metrics"]["test"] = m_test
    Path(out_path).with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

def ensure_ml_test_data(path: str, required_cols: List[str]):
    p = Path(path)
    if not p.exists():
        logger.info(f"[retrain] ML test file not found (OK if you don't use test): {p}")
        return
    df = pd.read_csv(p, nrows=5, low_memory=False)
    have = set(df.columns.tolist())
    need = set(required_cols)
    if not need.issubset(have):
        raise SystemExit(
            f"[retrain] ML test {p} has wrong columns. "
            f"Need at least: {sorted(need)}, have: {sorted(have)}"
        )
    logger.info(f"[retrain] ML test OK: {p}")
    if "label" in have:
        rows, pos, neg = _scan_binary_label_distribution(
            str(p),
            label_col="label",
            max_rows=ML_LABEL_CHECK_MAX_ROWS,
        )
        logger.info(
            "[retrain] ML test label distribution: rows=%s pos=%s neg=%s",
            rows, pos, neg,
        )
        if rows > 0 and (pos <= 0 or neg <= 0):
            logger.info(
                "[retrain] ML test has a single class; test ROC/PR will be skipped (rows=%s pos=%s neg=%s)",
                rows,
                pos,
                neg,
            )


def train_ml_rules_classifier(csv_path: str, out_path: str, meta_path: str, feature_cols: list[str], label_col: str = "label_any"):
    """
    Trains a HistGradientBoostingClassifier on ML rules dataset.
    
    Args:
        csv_path (str): Path to the input CSV file containing features and labels.
        out_path (str): Path to save the trained model.
        meta_path (str): Path to save the model metadata.
        feature_cols (list[str]): List of feature column names.
        label_col (str, optional): Name of the label column. Defaults to "label_any".
    """
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in (feature_cols + [label_col]) if c not in df.columns]
    if missing:
        raise ValueError(f"[ml-rules] missing columns: {missing}")

    X = df[feature_cols].fillna(0.0).astype(float).values
    y = df[label_col].astype(int).values

    # Simple validation split with stratification
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.sum() and y.sum()!=len(y) else None)

    clf = HistGradientBoostingClassifier(
        max_depth=None, learning_rate=0.1, max_iter=300, random_state=42
    )
    pos_idx = (y_tr == 1)
    n_pos = int(pos_idx.sum())
    n_neg = int((~pos_idx).sum())
    if 0 < n_pos < n_neg:
        reps = max(1, min(5, int((0.4 * n_neg) // max(1, n_pos))))
        if reps > 1:
            X_pos = X_tr[pos_idx]
            y_pos = y_tr[pos_idx]
            X_tr = np.vstack([X_tr] + [X_pos] * (reps - 1))
            y_tr = np.concatenate([y_tr] + [y_pos] * (reps - 1))

    clf.fit(X_tr, y_tr)

    proba_te = clf.predict_proba(X_te)[:,1]
    f1 = f1_score(y_te, (proba_te >= 0.5).astype(int), zero_division=0)
    auc = roc_auc_score(y_te, proba_te) if len(set(y_te)) > 1 else None

    proba_all = clf.predict_proba(X)[:, 1]
    prec, rec, thr = precision_recall_curve(y, proba_all)
    denom = prec + rec
    f1s = np.zeros_like(denom)
    valid = denom > 0
    f1s[valid] = 2.0 * prec[valid] * rec[valid] / denom[valid]
    thr_for_f1 = np.append(thr, [thr[-1] if len(thr) > 0 else 0.5])
    best_idx = int(np.argmax(f1s)) if len(f1s) else 0
    recommended_threshold = float(thr_for_f1[best_idx]) if len(thr_for_f1) else 0.6
    auc_all = float(roc_auc_score(y, proba_all)) if len(set(y)) > 1 else None

    joblib.dump({"clf": clf, "features": feature_cols}, out_path)

    meta = {
        "model": "HistGBDT (rules_supervised)",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_order": feature_cols,
        "recommended_threshold": recommended_threshold,
        "metrics": {
            "val_f1": float(f1),
            "val_auc": float(auc) if auc is not None else None,
            "all_auc": auc_all,
            "positives": int(y.sum()),
            "rows": int(len(y)),
        },
        "label_col": label_col
    }
    Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"[ml-rules] saved model -> {out_path}; meta -> {meta_path}")

def fit_attack_logistic_calibrator(
    stats: np.ndarray,
    labels: np.ndarray,
    kind: str = "",
    orientation: str | None = None,
    standardize_input: bool = True,
) -> tuple[dict | None, dict | None, float | None]:
    """
    Fit a 1D logistic regression mapping oriented anomaly statistic -> P(attack).
    ...
    """
    stats = np.asarray(stats, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    kind_norm = _normalize_anomaly_kind(kind)
    resolved_orientation = _resolve_score_orientation(
        kind_norm,
        override=orientation,
    )
    stats_oriented = _oriented_stats(stats, resolved_orientation)

    mask = np.isfinite(stats_oriented) & np.isfinite(labels)
    stats_oriented = stats_oriented[mask]
    labels = labels[mask]

    if stats_oriented.size == 0 or labels.size == 0 or stats_oriented.size != labels.size:
        logger.info(
            "[retrain][logit_calib] empty or mismatched stats/labels for kind=%s "
            "(n_stats=%s, n_labels=%s)",
            kind,
            stats_oriented.size,
            labels.size,
        )
        return None, None, None

    if len(np.unique(labels)) < 2:
        logger.info(
            "[retrain][logit_calib] only one class present for kind=%s, "
            "skip logistic calibration",
            kind,
        )
        return None, None, None

    z_mu = 0.0
    z_sigma = 1.0
    if standardize_input:
        z_mu = float(np.mean(stats_oriented))
        z_sigma = float(np.std(stats_oriented))
        if (not np.isfinite(z_sigma)) or z_sigma <= 1e-12:
            z_sigma = 1.0
        stats_fit = (stats_oriented - z_mu) / z_sigma
    else:
        stats_fit = stats_oriented

    try:
        # 1D feature: reshape to (n, 1)
        X = stats_fit.reshape(-1, 1)
        class_weight = "balanced" if ANOM_LOGIT_CLASS_WEIGHT in ("balanced", "1", "true", "yes", "on") else None
        default_c = float(max(1e-6, float(ANOM_LOGIT_C)))
        c_grid_raw = os.getenv("ANOM_LOGIT_C_GRID", "").strip()
        c_grid = []
        if c_grid_raw:
            for tok in c_grid_raw.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    c_grid.append(float(max(1e-6, float(tok))))
                except Exception:
                    continue
        if not c_grid:
            c_grid = [default_c]
        if all(abs(c - default_c) > 1e-12 for c in c_grid):
            c_grid.append(default_c)
        c_grid = sorted(set(c_grid))

        best_c = default_c
        best_cv_nll = None
        holdout = float(os.getenv("ANOM_LOGIT_CV_HOLDOUT", "0.2"))
        if (not np.isfinite(holdout)) or holdout <= 0.0 or holdout >= 0.5:
            holdout = 0.2

        # Select C on a stratified holdout by proper scoring (NLL).
        if len(c_grid) > 1 and labels.size >= 1000 and len(np.unique(labels)) >= 2:
            try:
                X_tr, X_va, y_tr, y_va = train_test_split(
                    X,
                    labels,
                    test_size=holdout,
                    random_state=42,
                    stratify=labels,
                )
                for c_try in c_grid:
                    clf_try = LogisticRegression(
                        solver="lbfgs",
                        max_iter=1000,
                        class_weight=class_weight,
                        C=float(c_try),
                    )
                    clf_try.fit(X_tr, y_tr)
                    p_va = clf_try.predict_proba(X_va)[:, 1]
                    nll_va = _binary_logloss(y_va, p_va)
                    if (best_cv_nll is None) or (nll_va < best_cv_nll):
                        best_cv_nll = float(nll_va)
                        best_c = float(c_try)
            except Exception as e:
                logger.info("[retrain][logit_calib] C-grid selection failed for kind=%s: %s", kind, e)
                best_c = default_c
                best_cv_nll = None

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight=class_weight,
            C=float(best_c),
        )
        clf.fit(X, labels)

        w = float(clf.coef_[0, 0])
        b0 = float(clf.intercept_[0])

        prob = clf.predict_proba(X)[:, 1]

        try:
            roc_probe = roc_auc_score(labels, prob)
        except Exception:
            roc_probe = None
        if roc_probe is not None and roc_probe < 0.5:
            # Keep calibration monotonic with "higher stat => higher attack risk".
            w = -w
            b0 = -b0
            prob = 1.0 - prob

        # Metrics on this calibration dataset
        try:
            roc = roc_auc_score(labels, prob)
        except Exception as e:
            logger.info("[retrain][logit_calib] roc_auc_score failed for kind=%s: %s", kind, e)
            roc = None

        try:
            ap = average_precision_score(labels, prob)
        except Exception as e:
            logger.info("[retrain][logit_calib] average_precision_score failed for kind=%s: %s", kind, e)
            ap = None

        prec, rec, thr = precision_recall_curve(labels, prob)
        denom = prec + rec
        f1s = np.zeros_like(denom)
        valid = denom > 0
        f1s[valid] = 2.0 * prec[valid] * rec[valid] / denom[valid]
        thr_for_f1 = np.append(thr, [thr[-1] if len(thr) else 0.5])
        best_idx = int(np.argmax(f1s)) if len(f1s) else 0
        recommended_threshold = float(thr_for_f1[best_idx]) if len(thr_for_f1) else 0.6

        metrics = {
            "roc_auc_attack": float(roc) if roc is not None else None,
            "pr_auc_attack": float(ap) if ap is not None else None,
            "best_f1_attack": float(f1s[best_idx]) if len(f1s) else None,
            "best_thr_attack": recommended_threshold,
            "rows": int(labels.size),
            "positives": int(labels.sum()),
            "orientation": resolved_orientation,
            "class_weight": "balanced" if class_weight == "balanced" else "none",
            "C": float(best_c),
            "C_default": float(default_c),
            "C_grid": [float(c) for c in c_grid],
            "C_cv_holdout": float(holdout),
            "C_cv_nll": float(best_cv_nll) if best_cv_nll is not None else None,
        }

        logger.info(
            "[retrain][logit_calib] kind=%s orientation=%s standardized=%s class_weight=%s C=%.4f C_default=%.4f cv_nll=%s z_mu=%.6f z_sigma=%.6f w=%.6f b=%.6f roc=%.4f ap=%.4f best_f1=%s thr=%.3f",
            kind,
            resolved_orientation,
            bool(standardize_input),
            "balanced" if class_weight == "balanced" else "none",
            float(best_c),
            float(default_c),
            f"{best_cv_nll:.6f}" if best_cv_nll is not None else None,
            z_mu,
            z_sigma,
            w,
            b0,
            roc if roc is not None else -1.0,
            ap if ap is not None else -1.0,
            metrics["best_f1_attack"],
            recommended_threshold,
        )

        return {
            "w": w,
            "b": b0,
            "orientation": resolved_orientation,
            "z_mu": z_mu,
            "z_sigma": z_sigma,
            "standardized_input": bool(standardize_input),
        }, metrics, recommended_threshold

    except Exception as e:
        logger.info("[retrain][logit_calib] failed for kind=%s: %s", kind, e)
        return None, None, None

def compute_anomaly_calibration(
    model,
    X: np.ndarray,
    kind: str = "",
) -> dict:
    """
    Compute calibration statistics (mu, sigma) for raw anomaly statistics
    where X is assumed to be BENIGN data only.

    These parameters are later used by decision_to_anomaly_score to
    map decision_function outputs to [0, 1].

    Args:
        model: Trained anomaly detection model exposing decision_function.
        X: 2D numpy array of features used for calibration (benign only).
        kind: Optional string used only for logging.
    Returns:
        A dict {"mu": float, "sigma": float, "orientation": str}
        or empty dict if calibration failed.
    """
    try:
        df_vals = _raw_anomaly_stat(model, X, kind=kind)
    except Exception as e:
        logger.info(
            f"[retrain] anomaly_calib: raw_stat failed for "
            f"{kind or type(model).__name__}: {e}"
        )
        return {}

    if df_vals.size == 0:
        logger.info(
            f"[retrain] anomaly_calib: empty raw_stat for "
            f"{kind or type(model).__name__}"
        )
        return {}

    kind_norm = _normalize_anomaly_kind(kind)
    orientation = _resolve_score_orientation(kind_norm)
    oriented_vals = _oriented_stats(df_vals, orientation)

    mu = float(oriented_vals.mean())
    sigma = float(oriented_vals.std())
    if not np.isfinite(sigma) or sigma <= 1e-9:
        sigma = 1.0

    logger.info(
        f"[retrain] anomaly_calib[{kind or type(model).__name__}]: "
        f"mu={mu:.6f} sigma={sigma:.6f} orientation={orientation}"
    )
    return {"mu": mu, "sigma": sigma, "orientation": orientation}

def decision_to_anomaly_score(
    df_vals: np.ndarray,
    calibration: dict | None,
    kind: str = "IF",
) -> np.ndarray:
    """
    Backwards-compatible враппер, щоб старий код не дублював логіку.

    Просто делегує в score_from_stat_with_calib з правильним kind.
    """
    k = _normalize_anomaly_kind(kind)
    return score_from_stat_with_calib(df_vals, calibration, kind=k)

def score_from_stat_with_calib(
    stats: np.ndarray,
    calibration: dict | None,
    kind: str,
) -> np.ndarray:
    stats = np.asarray(stats, dtype=float).reshape(-1)
    if stats.size == 0:
        return np.zeros_like(stats)

    calib = calibration or {}
    kind_norm = _normalize_anomaly_kind(kind)
    orientation = _resolve_score_orientation(kind_norm, calibration=calib)
    oriented = _oriented_stats(stats, orientation)

    attack_log = calib.get("attack_logistic")
    if isinstance(attack_log, dict):
        try:
            w = float(attack_log["w"])
            b0 = float(attack_log["b"])
            logit_override = attack_log.get("orientation") if "orientation" in attack_log else "identity"
            logit_orientation = _resolve_score_orientation(
                kind_norm,
                calibration=calib,
                override=logit_override,
            )
            oriented_for_logit = _oriented_stats(stats, logit_orientation)
            z_mu = float(attack_log.get("z_mu", 0.0))
            z_sigma = float(attack_log.get("z_sigma", 1.0))
            if (not np.isfinite(z_sigma)) or z_sigma <= 1e-12:
                z_sigma = 1.0
            oriented_for_logit = (oriented_for_logit - z_mu) / z_sigma
            z = w * oriented_for_logit + b0
            p = _sigmoid_from_logit_with_temperature(z, calib.get("temperature", 1.0))
            return _apply_prior_correction_probs(p, calib.get("prior_correction"))
        except Exception:
            pass

    ecdf_map = calib.get("ecdf_risk")
    p_ecdf = _apply_ecdf_risk_mapping(oriented, ecdf_map)
    if p_ecdf is not None:
        p = _apply_temperature_scaling(p_ecdf, calib.get("temperature", 1.0))
        return _apply_prior_correction_probs(p, calib.get("prior_correction"))

    # 2) mu / sigma
    mu = calib.get("mu")
    sigma = calib.get("sigma")

    try:
        mu = float(mu)
    except Exception:
        mu = float(oriented.mean())

    try:
        sigma = float(sigma)
    except Exception:
        sigma = float(oriented.std() or 1.0)

    if (not np.isfinite(sigma)) or sigma <= 1e-9:
        sigma = 1.0

    z = (oriented - mu) / sigma
    p = _sigmoid_from_logit_with_temperature(z, calib.get("temperature", 1.0))
    return _apply_prior_correction_probs(p, calib.get("prior_correction"))
  
def evaluate_if_benign_vs_attack(
    if_model,
    df_benign: pd.DataFrame,
    df_attack: pd.DataFrame,
    feature_order: list[str],
    calibration: dict | None,
    out_prefix: str = "/models/ml/if_eval_benign_vs_attack",
) -> dict:
    """
    Evaluate IsolationForest on a clean benign-vs-attack split.

    Assumptions:
      - df_benign contains only benign examples (label 0 by construction).
      - df_attack contains only attack examples (label 1 by construction).
      - Both come from the same feature space (feature_order).

    We:
      - compute decision_function on both,
      - map to anomaly scores via decision_to_anomaly_score (same as runtime),
      - combine into a single dataset with labels (0 for benign, 1 for attack),
      - compute ROC AUC, PR AUC, best F1, and save plots + CSV.

    Args:
        if_model: Trained IsolationForest model.
        df_benign: DataFrame with benign rows (e.g. validation label=0).
        df_attack: DataFrame with attack rows (e.g. validation label=1).
        feature_order: List of feature column names in the correct order.
        calibration: Dict {"mu", "sigma"} used by decision_to_anomaly_score.
        out_prefix: Prefix for output CSV/PNG files.
    Returns:
        metrics dict with ROC/PR/F1 and basic counts.
    """
    if df_benign.empty or df_attack.empty:
        logger.info(
            "[retrain][IF eval] benign or attack dataframe is empty, "
            "skip evaluation (benign=%s, attack=%s)",
            df_benign.shape[0],
            df_attack.shape[0],
        )
        return {}

    X_benign = ensure_columns(df_benign, feature_order).fillna(0.0).astype(float).values
    X_attack = ensure_columns(df_attack, feature_order).fillna(0.0).astype(float).values

    df_raw_benign = _raw_anomaly_stat(if_model, X_benign, kind="if")
    df_raw_attack = _raw_anomaly_stat(if_model, X_attack, kind="if")

    scores_benign = decision_to_anomaly_score(
        df_raw_benign,
        calibration,
        kind="IF_eval_benign",
    )
    scores_attack = decision_to_anomaly_score(
        df_raw_attack,
        calibration,
        kind="IF_eval_attack",
    )

    y_benign = np.zeros_like(scores_benign, dtype=int)
    y_attack = np.ones_like(scores_attack, dtype=int)

    scores = np.concatenate([scores_benign, scores_attack])
    y = np.concatenate([y_benign, y_attack])

    # Metrics
    try:
        roc = roc_auc_score(y, scores)
    except Exception as e:
        logger.info(f"[retrain][IF eval] roc_auc_score failed: {e}")
        roc = None

    try:
        ap = average_precision_score(y, scores)
    except Exception as e:
        logger.info(f"[retrain][IF eval] average_precision_score failed: {e}")
        ap = None

    prec, rec, thr = precision_recall_curve(y, scores)
    denom = prec + rec
    f1s = np.zeros_like(denom)
    valid = denom > 0
    f1s[valid] = 2.0 * prec[valid] * rec[valid] / denom[valid]
    best_f1 = float(f1s.max()) if len(f1s) else None

    logger.info(
        "[retrain][IF eval] benign=%s attack=%s ROC_AUC=%s PR_AUC=%s best_F1=%s",
        int((y == 0).sum()),
        int((y == 1).sum()),
        f"{roc:.4f}" if roc is not None else None,
        f"{ap:.4f}" if ap is not None else None,
        best_f1,
    )

    # Save scores CSV
    out_scores_csv = f"{out_prefix}_scores.csv"
    eval_df = pd.DataFrame({"score": scores, "label": y})
    Path(out_scores_csv).parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(out_scores_csv, index=False)
    logger.info(f"[retrain][IF eval] scores saved -> {out_scores_csv}")

    # Plots
    try:
        # ROC
        if roc is not None:
            fpr, tpr, _ = roc_curve(y, scores)
            plt.figure()
            plt.plot(fpr, tpr, label=f"IF (AUC={roc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("IsolationForest ROC (val benign vs val attack)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_roc.png")
            plt.close()

        # PR
        if ap is not None:
            plt.figure()
            plt.step(rec, prec, where="post")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"IsolationForest PR (val split, AP={ap:.3f})")
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_pr.png")
            plt.close()

        # Histograms for benign vs attack
        plt.figure()
        plt.hist(
            scores_benign,
            bins=100,
            alpha=0.7,
            label="benign (val)",
        )
        plt.hist(
            scores_attack,
            bins=100,
            alpha=0.7,
            label="attack (val)",
        )
        plt.xlabel("IF calibrated anomaly risk score")
        plt.ylabel("Count")
        plt.title("IsolationForest score distribution (val benign vs val attack)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_hist.png")
        plt.close()

        logger.info(f"[retrain][IF eval] plots saved with prefix {out_prefix}_*.png")
    except Exception as e:
        logger.info(f"[retrain][IF eval] plotting failed: {e}")

    metrics = {
        "roc_auc_benign_vs_attack": float(roc) if roc is not None else None,
        "pr_auc_benign_vs_attack": float(ap) if ap is not None else None,
        "best_f1_benign_vs_attack": best_f1,
        "benign_rows": int((y == 0).sum()),
        "attack_rows": int((y == 1).sum()),
    }
    return metrics

def evaluate_if_with_labels(
    if_model,
    ml_df: pd.DataFrame,
    feature_order: list[str],
    out_prefix: str = "/models/ml/if_csic_eval",
    calibration: dict | None = None,
):
    """
    Evaluate IsolationForest on labeled ML numeric data and compute metrics.

    Uses the same decision_function → score mapping as runtime
    (decision_to_anomaly_score + calibration).

    Args:
        if_model: Trained IsolationForest model.
        ml_df: DataFrame with features + 'label' column (0/1).
        feature_order: List of feature column names in the correct order.
        out_prefix: Prefix for output CSV/PNG files.
        calibration: Dict with {"mu","sigma"} used by decision_to_anomaly_score.
    """
    if "label" not in ml_df.columns:
        logger.info("[retrain] IF eval: 'label' column not found in ml_numeric – skip")
        return

    df = ml_df.dropna(subset=["label"]).copy()
    if df.empty:
        logger.info("[retrain] IF eval: no rows with non-null label – skip")
        return

    y = df["label"].astype(int).values
    X = df[feature_order].values

    df_raw = _raw_anomaly_stat(if_model, X, kind="if")

    # Use the same calibration/mapping as in train/runtime
    scores = decision_to_anomaly_score(df_raw, calibration, kind="IF_eval")

    # Metrics
    try:
        roc = roc_auc_score(y, scores)
    except Exception as e:
        logger.info(f"[retrain] IF eval: roc_auc_score failed: {e}")
        roc = None

    try:
        ap = average_precision_score(y, scores)
    except Exception as e:
        logger.info(f"[retrain] IF eval: average_precision failed: {e}")
        ap = None

    logger.info(
        f"[retrain] IF eval: n={len(y)}, pos={int(y.sum())}, neg={int((y == 0).sum())}"
    )
    logger.info(
        f"[retrain] IF eval: ROC AUC={roc:.4f} AP(PR AUC)={ap:.4f}"
        if roc is not None and ap is not None
        else f"[retrain] IF eval: ROC={roc} AP={ap}"
    )

    # Per-class statistics
    stats = {}
    for lab in (0, 1):
        mask = (y == lab)
        if not mask.any():
            continue
        s = scores[mask]
        stats[lab] = {
            "n": int(mask.sum()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "p5": float(np.percentile(s, 5)),
            "p25": float(np.percentile(s, 25)),
            "p50": float(np.percentile(s, 50)),
            "p75": float(np.percentile(s, 75)),
            "p95": float(np.percentile(s, 95)),
        }
    logger.info(f"[retrain] IF eval: per-class stats = {json.dumps(stats, indent=2)}")

    # Save scores to CSV
    out_scores_csv = f"{out_prefix}_scores.csv"
    eval_df = pd.DataFrame({"score": scores, "label": y})
    eval_df.to_csv(out_scores_csv, index=False)
    logger.info(f"[retrain] IF eval: scores saved -> {out_scores_csv}")

    # Plots
    try:
        # ROC
        if roc is not None:
            fpr, tpr, _ = roc_curve(y, scores)
            plt.figure()
            plt.plot(fpr, tpr, label=f"IF (AUC={roc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("IsolationForest ROC (CSIC)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_roc.png")
            plt.close()

        # PR
        prec, rec, _ = precision_recall_curve(y, scores)
        if ap is not None:
            plt.figure()
            plt.step(rec, prec, where="post")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"IsolationForest PR (AP={ap:.3f})")
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_pr.png")
            plt.close()

        # Histograms
        plt.figure()
        plt.hist(scores[y == 0], bins=100, alpha=0.7, label="label=0 (normal)")
        if (y == 1).any():
            plt.hist(scores[y == 1], bins=100, alpha=0.7, label="label=1 (attack)")
        plt.xlabel("IF calibrated anomaly risk score")
        plt.ylabel("Count")
        plt.title("IsolationForest score distribution (CSIC)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_hist.png")
        plt.close()

        logger.info(f"[retrain] IF eval: plots saved with prefix {out_prefix}_*.png")
    except Exception as e:
        logger.info(f"[retrain] IF eval: plotting failed: {e}")

# ML-model training functions
def train_isolation_forest(
    train_csv_path: str,
    val_csv_path: str | None,
    test_csv_path: str | None,
    out_path: str,
    feature_order: List[str],
):
    logger.info("[retrain][IF] train=%s val=%s test=%s", train_csv_path, val_csv_path, test_csv_path)

    df_tr = pd.read_csv(train_csv_path, low_memory=False)
    X_tr = ensure_columns(df_tr, feature_order).fillna(0.0).astype(float).values
    if X_tr.shape[0] == 0:
        raise SystemExit(f"[retrain][IF] training CSV {train_csv_path} has no rows")

    def _parse_bool_token(tok: str) -> bool | None:
        s = str(tok).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        return None

    def _parse_int_grid(raw: str, fallback: list[int]) -> list[int]:
        vals: list[int] = []
        if raw:
            for tok in raw.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    v = int(float(tok))
                    if v > 0:
                        vals.append(v)
                except Exception:
                    continue
        if not vals:
            vals = list(fallback)
        out: list[int] = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out

    def _parse_max_samples_grid(raw: str, fallback: list[object]) -> list[object]:
        vals: list[object] = []
        if raw:
            for tok in raw.split(","):
                s = tok.strip().lower()
                if not s:
                    continue
                if s == "auto":
                    vals.append("auto")
                    continue
                try:
                    v = float(s)
                    if v <= 0.0:
                        continue
                    if v >= 1.0 and abs(v - round(v)) < 1e-9:
                        vals.append(int(round(v)))
                    elif v <= 1.0:
                        vals.append(float(v))
                except Exception:
                    continue
        if not vals:
            vals = list(fallback)
        out: list[object] = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out

    def _parse_max_features_grid(raw: str, fallback: list[object]) -> list[object]:
        vals: list[object] = []
        if raw:
            for tok in raw.split(","):
                s = tok.strip()
                if not s:
                    continue
                try:
                    v = float(s)
                    if v <= 0.0:
                        continue
                    if v >= 1.0 and abs(v - round(v)) < 1e-9:
                        vals.append(int(round(v)))
                    elif v <= 1.0:
                        vals.append(float(v))
                except Exception:
                    continue
        if not vals:
            vals = list(fallback)
        out: list[object] = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out

    def _parse_bootstrap_grid(raw: str, fallback: list[bool]) -> list[bool]:
        vals: list[bool] = []
        if raw:
            for tok in raw.split(","):
                b = _parse_bool_token(tok)
                if b is not None:
                    vals.append(bool(b))
        if not vals:
            vals = list(fallback)
        out: list[bool] = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out

    if_n_estimators = int(os.getenv("IF_N_ESTIMATORS", "200"))
    if_contamination = os.getenv("IF_CONTAMINATION", "auto")
    if_random_state = int(os.getenv("IF_RANDOM_STATE", "42"))
    if_max_samples_default: object = os.getenv("IF_MAX_SAMPLES", "auto").strip().lower()
    if if_max_samples_default != "auto":
        try:
            _ms = float(if_max_samples_default)
            if _ms >= 1.0 and abs(_ms - round(_ms)) < 1e-9:
                if_max_samples_default = int(round(_ms))
            elif _ms > 0.0 and _ms <= 1.0:
                if_max_samples_default = float(_ms)
            else:
                if_max_samples_default = "auto"
        except Exception:
            if_max_samples_default = "auto"

    if_max_features_default: object = 1.0
    try:
        _mf = float(os.getenv("IF_MAX_FEATURES", "1.0"))
        if _mf >= 1.0 and abs(_mf - round(_mf)) < 1e-9:
            if_max_features_default = int(round(_mf))
        elif _mf > 0.0 and _mf <= 1.0:
            if_max_features_default = float(_mf)
    except Exception:
        if_max_features_default = 1.0

    if_bootstrap_default = _parse_bool_token(os.getenv("IF_BOOTSTRAP", "false"))
    if_bootstrap_default = bool(if_bootstrap_default) if if_bootstrap_default is not None else False

    if_tune_enabled = os.getenv("IF_TUNE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
    if_tune_topk = max(1, int(os.getenv("IF_TUNE_TOPK_LOG", "10")))

    if_selection_info: dict | None = None
    model = None
    selected_params = {
        "n_estimators": int(if_n_estimators),
        "max_samples": if_max_samples_default,
        "max_features": if_max_features_default,
        "bootstrap": bool(if_bootstrap_default),
        "contamination": if_contamination,
        "random_state": int(if_random_state),
    }

    X_val_tune = None
    y_val_tune = None
    if if_tune_enabled and val_csv_path and Path(val_csv_path).exists():
        try:
            df_val_tune = pd.read_csv(val_csv_path, low_memory=False)
            y_val_tune = _labels_from_df(df_val_tune)
            if y_val_tune is not None and len(np.unique(y_val_tune)) >= 2:
                df_val_tune, y_val_tune = _maybe_downsample_df(
                    df_val_tune,
                    y_val_tune,
                    ML_VAL_MAX_ROWS,
                    tag="IF/val_tune",
                )
                X_val_tune = ensure_columns(df_val_tune, feature_order).fillna(0.0).astype(float).values
        except Exception as e:
            logger.info("[retrain][IF] tune-val load failed, fallback to default params: %s", e)
            X_val_tune = None
            y_val_tune = None

    if if_tune_enabled and X_val_tune is not None and y_val_tune is not None:
        n_grid = _parse_int_grid(
            os.getenv("IF_N_ESTIMATORS_GRID", "").strip(),
            fallback=[int(if_n_estimators)],
        )
        ms_grid = _parse_max_samples_grid(
            os.getenv("IF_MAX_SAMPLES_GRID", "").strip(),
            fallback=[if_max_samples_default],
        )
        mf_grid = _parse_max_features_grid(
            os.getenv("IF_MAX_FEATURES_GRID", "").strip(),
            fallback=[if_max_features_default],
        )
        bs_grid = _parse_bootstrap_grid(
            os.getenv("IF_BOOTSTRAP_GRID", "").strip(),
            fallback=[bool(if_bootstrap_default)],
        )

        candidates: list[dict] = []
        best_obj = -np.inf
        best_rank = None

        for n_est in n_grid:
            for max_samples in ms_grid:
                for max_features in mf_grid:
                    for bootstrap in bs_grid:
                        try:
                            cand_model = IsolationForest(
                                n_estimators=int(n_est),
                                max_samples=max_samples,
                                max_features=max_features,
                                bootstrap=bool(bootstrap),
                                contamination=if_contamination,
                                random_state=if_random_state,
                                n_jobs=-1,
                            )
                            cand_model.fit(X_tr)
                            stat_val_rank = _raw_anomaly_stat(cand_model, X_val_tune, kind="if")
                            score_rank = _oriented_stats(stat_val_rank, _resolve_score_orientation("if"))
                            roc_val = float(roc_auc_score(y_val_tune, score_rank))
                            pr_val = float(average_precision_score(y_val_tune, score_rank))
                            obj = float(roc_val + pr_val)
                            item = {
                                "n_estimators": int(n_est),
                                "max_samples": max_samples,
                                "max_features": max_features,
                                "bootstrap": bool(bootstrap),
                                "roc_auc": roc_val,
                                "pr_auc": pr_val,
                                "objective": obj,
                            }
                            candidates.append(item)
                            rank = (obj, roc_val, pr_val)
                            if rank > (best_rank if best_rank is not None else (-np.inf, -np.inf, -np.inf)):
                                best_rank = rank
                                best_obj = obj
                                model = cand_model
                                selected_params = {
                                    "n_estimators": int(n_est),
                                    "max_samples": max_samples,
                                    "max_features": max_features,
                                    "bootstrap": bool(bootstrap),
                                    "contamination": if_contamination,
                                    "random_state": if_random_state,
                                }
                        except Exception as e:
                            logger.info(
                                "[retrain][IF] candidate failed n=%s max_samples=%s max_features=%s bootstrap=%s: %s",
                                n_est,
                                max_samples,
                                max_features,
                                bootstrap,
                                e,
                            )

        if candidates and model is not None:
            cand_sorted = sorted(candidates, key=lambda z: (z["objective"], z["roc_auc"], z["pr_auc"]), reverse=True)
            if_selection_info = {
                "enabled": True,
                "selection_metric": "max(roc_auc + pr_auc) on validation raw ranking",
                "selected": {
                    **selected_params,
                    "objective": float(best_obj),
                },
                "grid_sizes": {
                    "n_estimators": len(n_grid),
                    "max_samples": len(ms_grid),
                    "max_features": len(mf_grid),
                    "bootstrap": len(bs_grid),
                    "candidates_total": len(candidates),
                },
                "top_candidates": cand_sorted[:if_tune_topk],
            }
            logger.info(
                "[retrain][IF] selected candidate n=%s max_samples=%s max_features=%s bootstrap=%s roc=%.4f pr=%.4f obj=%.4f",
                selected_params["n_estimators"],
                selected_params["max_samples"],
                selected_params["max_features"],
                selected_params["bootstrap"],
                cand_sorted[0]["roc_auc"],
                cand_sorted[0]["pr_auc"],
                cand_sorted[0]["objective"],
            )
        else:
            logger.info("[retrain][IF] no valid tuning candidates; fallback to default IF params")
            model = None

    if model is None:
        model = IsolationForest(
            n_estimators=int(selected_params["n_estimators"]),
            max_samples=selected_params["max_samples"],
            max_features=selected_params["max_features"],
            bootstrap=bool(selected_params["bootstrap"]),
            contamination=selected_params["contamination"],
            random_state=int(selected_params["random_state"]),
            n_jobs=-1,
        )
        model.fit(X_tr)

    logger.info(
        "[retrain][IF] trained on %s rows with params n_estimators=%s max_samples=%s max_features=%s bootstrap=%s",
        X_tr.shape[0],
        selected_params["n_estimators"],
        selected_params["max_samples"],
        selected_params["max_features"],
        selected_params["bootstrap"],
    )

    # base calibration on benign train
    calib = compute_anomaly_calibration(model, X_tr, kind="if")

    metrics = {"val": None, "test": None}
    recommended_threshold = None

    # --- VAL: fit optional logistic calibrator + choose threshold on VAL ---
    if val_csv_path and Path(val_csv_path).exists():
        df_val = pd.read_csv(val_csv_path, low_memory=False)
        y_val = _labels_from_df(df_val)
        if y_val is not None and len(np.unique(y_val)) >= 2:
            df_val, y_val = _maybe_downsample_df(
                df_val,
                y_val,
                ML_VAL_MAX_ROWS,
                tag="IF/val",
            )
            X_val = ensure_columns(df_val, feature_order).fillna(0.0).astype(float).values
            stat_val = _raw_anomaly_stat(model, X_val, kind="if")

            # 1) baseline eval on mu/sigma only
            calib_mu_sigma = dict(calib or {})
            calib_mu_sigma.pop("attack_logistic", None)
            score_val_mu_sigma = score_from_stat_with_calib(stat_val, calib_mu_sigma, kind="if")
            if_eval_mu_sigma = _eval_and_plot_scores_with_labels(
                score_val_mu_sigma,
                y_val,
                out_prefix="/models/ml/if_eval_mu_sigma",
                title_prefix="IF(mu_sigma)",
            )

            # 2) fit logistic on oriented statistic and update calibration
            logit_params, logit_metrics, logit_thr = fit_attack_logistic_calibrator(
                stat_val,
                y_val,
                kind="if",
                orientation=_resolve_score_orientation("if", calibration=calib_mu_sigma),
            )
            if logit_params is not None:
                calib = calib or {}
                calib["attack_logistic"] = logit_params

            # 3) runtime scores with final calibration (possibly with attack_logistic)
            score_val = score_from_stat_with_calib(stat_val, calib, kind="if")
            if_eval_runtime = _eval_and_plot_scores_with_labels(
                score_val,
                y_val,
                out_prefix="/models/ml/if_eval",
                title_prefix="IF(runtime)",
            )
            use_logit, logit_reason = _should_use_attack_logistic(
                logit_metrics,
                baseline_eval_metrics=if_eval_mu_sigma,
                runtime_eval_metrics=if_eval_runtime,
            )
            if (logit_params is not None) and (not use_logit):
                logger.info("[retrain][IF] skip attack_logistic: %s", logit_reason)
                calib = dict(calib or {})
                calib.pop("attack_logistic", None)
                score_val = score_val_mu_sigma
                if_eval_runtime = _eval_and_plot_scores_with_labels(
                    score_val,
                    y_val,
                    out_prefix="/models/ml/if_eval",
                    title_prefix="IF(runtime)",
                )

            # 4) optional temperature scaling on final runtime score
            temp_t, temp_metrics = fit_temperature_scaler(score_val, y_val, kind="if")
            if temp_metrics.get("used"):
                calib = calib or {}
                calib["temperature"] = float(temp_t)
                score_val = score_from_stat_with_calib(stat_val, calib, kind="if")
                if_eval_runtime = _eval_and_plot_scores_with_labels(
                    score_val,
                    y_val,
                    out_prefix="/models/ml/if_eval",
                    title_prefix="IF(runtime)",
                )
            elif isinstance(calib, dict):
                calib.pop("temperature", None)

            # Optional prior correction (train/calibration prior -> deployment prior).
            calib, prior_info = _attach_prior_correction_to_calibration(calib, y_val, kind="if")
            score_val = score_from_stat_with_calib(stat_val, calib, kind="if")
            if_eval_runtime = _eval_and_plot_scores_with_labels(
                score_val,
                y_val,
                out_prefix="/models/ml/if_eval",
                title_prefix="IF(runtime)",
            )

            # Choose operating threshold by configured policy.
            recommended_threshold, thr_info = choose_operating_threshold(y_val, score_val)
            _, bestf1_ref = best_threshold_f1(y_val, score_val)

            metrics["val"] = {
                **metrics_on_threshold(y_val, score_val, recommended_threshold),
                "best_f1_on_val": float(bestf1_ref),
                "calibrated_eval": if_eval_runtime,
                "eval_mu_sigma": if_eval_mu_sigma,
                "runtime_score_type": "attack_logistic" if ((logit_params is not None) and use_logit) else "mu_sigma",
                "threshold_selection": thr_info,
                "prior_correction": prior_info,
            }
            if if_selection_info is not None:
                metrics["val"]["if_model_selection"] = if_selection_info
            if logit_metrics is not None:
                metrics["val"]["attack_logistic"] = logit_metrics
                metrics["val"]["attack_logistic_used"] = bool((logit_params is not None) and use_logit)
                metrics["val"]["attack_logistic_reason"] = logit_reason
            metrics["val"]["temperature_scaling"] = temp_metrics

            # Comparable IF eval on the same validation split as the other models.
            df_val_benign = df_val[y_val == 0]
            df_val_attack = df_val[y_val == 1]
            if (not df_val_benign.empty) and (not df_val_attack.empty):
                metrics["val_benign_vs_attack"] = evaluate_if_benign_vs_attack(
                    if_model=model,
                    df_benign=df_val_benign,
                    df_attack=df_val_attack,
                    feature_order=feature_order,
                    calibration=calib,
                    out_prefix="/models/ml/if_benign_vs_attack",
                )
        else:
            logger.info("[retrain][IF] VAL has no usable labels; skipping val threshold selection")

    # --- TEST: optional final metrics on untouched TEST (disabled by default) ---
    if RETRAIN_ENABLE_TEST_EVAL and test_csv_path and Path(test_csv_path).exists() and recommended_threshold is not None:
        df_te = pd.read_csv(test_csv_path, low_memory=False)
        y_te = _labels_from_df(df_te)
        if y_te is not None and len(np.unique(y_te)) >= 2:
            df_te, y_te = _maybe_downsample_df(
                df_te,
                y_te,
                ML_TEST_MAX_ROWS,
                tag="IF/test",
            )
            X_te = ensure_columns(df_te, feature_order).fillna(0.0).astype(float).values
            stat_te = _raw_anomaly_stat(model, X_te, kind="if")
            score_te = score_from_stat_with_calib(stat_te, calib, kind="if")
            metrics["test"] = metrics_on_threshold(y_te, score_te, recommended_threshold)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, outp)
    logger.info(f"[retrain][IF] saved -> {outp}")

    write_if_meta(
        outp, feature_order, model,
        calibration=calib,
        metrics=metrics,
        recommended_threshold=recommended_threshold,
    )
    return model, calib, recommended_threshold, metrics


def train_ocsvm(
    train_csv_path: str,
    val_csv_path: str | None,
    test_csv_path: str | None,
    out_path: str,
    feature_order: List[str],
):
    """
    Train OCSVM on benign-only train data.
    Calibration/selection is performed only on mixed validation set:
      - mu_sigma
      - attack_logistic (Platt-style 1D)
      - ecdf_risk mapping
    Runtime calibration is selected by proper scoring rules (log-loss/brier/ece).
    """
    logger.info("[retrain][OCSVM] train_csv=%s val_csv=%s", train_csv_path, val_csv_path)

    # === TRAIN (benign only) ===
    df_tr = pd.read_csv(train_csv_path, low_memory=False)

    # Optional benign augmentation with disjoint CSIC slice for OCSVM only.
    if OCSVM_EXTRA_BENIGN_FILE:
        extra_path = Path(OCSVM_EXTRA_BENIGN_FILE)
        if extra_path.exists():
            try:
                df_extra = pd.read_csv(extra_path, low_memory=False)
                if "label" in df_extra.columns:
                    y_extra = pd.to_numeric(df_extra["label"], errors="coerce").fillna(0.0)
                    df_extra = df_extra.loc[y_extra <= 0.0].copy()
                missing_extra = [c for c in feature_order if c not in df_extra.columns]
                if missing_extra:
                    logger.info(
                        "[retrain][OCSVM] skip extra benign file %s: missing feature columns %s",
                        extra_path,
                        missing_extra,
                    )
                elif not df_extra.empty:
                    if OCSVM_EXTRA_BENIGN_MAX_ROWS > 0 and df_extra.shape[0] > OCSVM_EXTRA_BENIGN_MAX_ROWS:
                        df_extra = df_extra.sample(n=OCSVM_EXTRA_BENIGN_MAX_ROWS, random_state=42)
                    df_tr = pd.concat([df_tr, df_extra], ignore_index=True)
                    logger.info(
                        "[retrain][OCSVM] appended extra benign rows from %s: +%s (train_total=%s)",
                        extra_path,
                        df_extra.shape[0],
                        df_tr.shape[0],
                    )
                else:
                    logger.info("[retrain][OCSVM] extra benign file is empty after benign filter: %s", extra_path)
            except Exception as e:
                logger.info("[retrain][OCSVM] failed to load extra benign file %s: %s", extra_path, e)
        else:
            logger.info("[retrain][OCSVM] extra benign file not found (skip): %s", extra_path)

    max_rows = int(os.getenv("OCSVM_MAX_TRAIN_ROWS", "150000"))
    if df_tr.shape[0] > max_rows:
        logger.info(
            "[retrain][OCSVM] train has %s rows, sampling %s for training",
            df_tr.shape[0],
            max_rows,
        )
        df_tr = df_tr.sample(n=max_rows, random_state=42)

    X_tr = ensure_columns(df_tr, feature_order).fillna(0.0).astype(float).values

    nu = float(os.getenv("OCSVM_NU", "0.05"))
    gamma = os.getenv("OCSVM_GAMMA", "scale")
    scaler_mode = os.getenv("OCSVM_SCALER", "robust").strip().lower()

    if scaler_mode == "standard":
        oc_scaler = StandardScaler()
    elif scaler_mode in ("none", "off", "disabled"):
        oc_scaler = None
    else:
        # Robust scaling is generally safer for heavy-tailed web-log features.
        q_low = float(os.getenv("OCSVM_ROBUST_Q_LOW", "5.0"))
        q_high = float(os.getenv("OCSVM_ROBUST_Q_HIGH", "95.0"))
        if q_low < 0.0:
            q_low = 0.0
        if q_high > 100.0:
            q_high = 100.0
        if q_low >= q_high:
            q_low, q_high = 5.0, 95.0
        oc_scaler = RobustScaler(quantile_range=(q_low, q_high))
        scaler_mode = "robust"

    oc_est = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    if oc_scaler is None:
        model = oc_est
    else:
        model = Pipeline(
            steps=[
                ("scaler", oc_scaler),
                ("ocsvm", oc_est),
            ]
        )
    model.fit(X_tr)

    logger.info(
        "[retrain][OCSVM] trained model on %s rows, scaler=%s n_features=%s",
        X_tr.shape[0],
        scaler_mode,
        getattr(model, "n_features_in_", None),
    )

    # Robust calibration baseline from benign train raw statistic.
    stat_tr = _raw_anomaly_stat(model, X_tr, kind="ocsvm")
    orientation = _resolve_score_orientation("ocsvm")
    stat_tr_oriented = _oriented_stats(stat_tr, orientation)
    calib_mu = float(np.median(stat_tr_oriented))
    q25, q75 = np.quantile(stat_tr_oriented, [0.25, 0.75])
    calib_sigma = float((q75 - q25) / 1.349)
    if (not np.isfinite(calib_sigma)) or calib_sigma <= 1e-9:
        calib_sigma = float(np.std(stat_tr_oriented) or 1.0)
    calib = {
        "mu": calib_mu,
        "sigma": calib_sigma,
        "orientation": orientation,
        "scale_method": "median_iqr",
    }
    ocsvm_ecdf_risk = fit_ecdf_risk_mapping(stat_tr_oriented)

    metrics = None
    recommended_threshold = None

    if val_csv_path is not None and Path(val_csv_path).exists():
        try:
            df_val = pd.read_csv(val_csv_path, low_memory=False)
            if "label" in df_val.columns:
                y_val = _labels_from_df(df_val)
                df_val, y_val = _maybe_downsample_df(
                    df_val,
                    y_val,
                    ML_VAL_MAX_ROWS,
                    tag="OCSVM/val",
                )
                X_val = ensure_columns(df_val, feature_order).fillna(0.0).astype(float).values

                if y_val is not None and len(np.unique(y_val)) >= 2:
                    stat_val = _raw_anomaly_stat(model, X_val, kind="ocsvm")

                    # 1) Baseline eval on mu/sigma only
                    calib_mu_sigma = dict(calib or {})
                    calib_mu_sigma.pop("attack_logistic", None)
                    calib_mu_sigma.pop("ecdf_risk", None)
                    calib_mu_sigma.pop("temperature", None)
                    scores_val_mu_sigma = score_from_stat_with_calib(
                        stat_val,
                        calib_mu_sigma,
                        kind="ocsvm",
                    )
                    oc_metrics_mu_sigma = _eval_and_plot_scores_with_labels(
                        scores_val_mu_sigma,
                        y_val,
                        out_prefix="/models/ml/ocsvm_eval_mu_sigma",
                        title_prefix="OCSVM(mu_sigma)",
                    )

                    # 2) Fit Platt-style 1D logistic on oriented raw statistic.
                    logit_params, logit_metrics, logit_thr = fit_attack_logistic_calibrator(
                        stat_val,
                        y_val,
                        kind="ocsvm",
                        orientation=_resolve_score_orientation("ocsvm", calibration=calib_mu_sigma),
                    )

                    # 3) Select runtime calibration path by proper scoring rules.
                    candidate_runs: list[dict] = []
                    for cand_name in ("mu_sigma", "attack_logistic", "ecdf_risk"):
                        if cand_name == "attack_logistic" and logit_params is None:
                            continue
                        if cand_name == "ecdf_risk" and ocsvm_ecdf_risk is None:
                            continue

                        cand_calib = dict(calib_mu_sigma or {})
                        cand_calib.pop("temperature", None)
                        if cand_name == "attack_logistic":
                            cand_calib["attack_logistic"] = dict(logit_params)
                            cand_calib.pop("ecdf_risk", None)
                        elif cand_name == "ecdf_risk":
                            cand_calib.pop("attack_logistic", None)
                            cand_calib["ecdf_risk"] = dict(ocsvm_ecdf_risk)
                        else:
                            cand_calib.pop("attack_logistic", None)
                            cand_calib.pop("ecdf_risk", None)

                        cand_scores = score_from_stat_with_calib(
                            stat_val,
                            cand_calib,
                            kind="ocsvm",
                        )
                        cand_temp_t, cand_temp_metrics = fit_temperature_scaler(
                            cand_scores,
                            y_val,
                            kind=f"ocsvm/{cand_name}",
                        )
                        if cand_temp_metrics.get("used"):
                            cand_calib["temperature"] = float(cand_temp_t)
                            cand_scores = score_from_stat_with_calib(
                                stat_val,
                                cand_calib,
                                kind="ocsvm",
                            )
                        else:
                            cand_calib.pop("temperature", None)

                        cand_eval = _eval_scores_only_with_labels(cand_scores, y_val)
                        candidate_runs.append(
                            {
                                "name": cand_name,
                                "calibration": cand_calib,
                                "scores": cand_scores,
                                "eval": cand_eval,
                                "temperature_scaling": cand_temp_metrics,
                            }
                        )

                    best_candidate, candidate_comparison, selection_reason = _select_best_runtime_calibration_candidate(
                        candidate_runs
                    )
                    runtime_score_type = str(best_candidate.get("name", "mu_sigma"))
                    use_logit = runtime_score_type == "attack_logistic"
                    logit_reason = selection_reason
                    calib = dict(best_candidate.get("calibration") or dict(calib_mu_sigma or {}))
                    temp_metrics = best_candidate.get("temperature_scaling") or {
                        "used": False,
                        "temperature": 1.0,
                    }
                    scores_val_runtime = np.asarray(
                        best_candidate.get("scores", scores_val_mu_sigma),
                        dtype=float,
                    ).reshape(-1)

                    logger.info(
                        "[retrain][OCSVM] runtime calibration selected: %s (%s)",
                        runtime_score_type,
                        logit_reason,
                    )

                    # Optional prior correction (calibration prior -> deployment prior).
                    calib, prior_info = _attach_prior_correction_to_calibration(
                        calib,
                        y_val,
                        kind="ocsvm",
                    )
                    scores_val_runtime = score_from_stat_with_calib(
                        stat_val,
                        calib,
                        kind="ocsvm",
                    )
                    oc_metrics_runtime = _eval_and_plot_scores_with_labels(
                        scores_val_runtime,
                        y_val,
                        out_prefix="/models/ml/ocsvm_eval",
                        title_prefix="OCSVM(runtime)",
                    )

                    # Choose operating threshold by configured policy.
                    recommended_threshold, thr_info = choose_operating_threshold(y_val, scores_val_runtime)
                    _, bestf1_ref = best_threshold_f1(y_val, scores_val_runtime)

                    metrics = {
                        **metrics_on_threshold(y_val, scores_val_runtime, recommended_threshold),
                        "best_f1_on_val": float(bestf1_ref),
                        "calibrated_eval": oc_metrics_runtime,
                        "eval_mu_sigma": oc_metrics_mu_sigma,
                        "runtime_score_type": runtime_score_type,
                        "label_metrics_from": str(val_csv_path),
                        "label_positives": int(y_val.sum()),
                        "label_rows": int(len(y_val)),
                        "calibration_selection": {
                            "method": "min_log_loss_then_brier",
                            "reason": logit_reason,
                            "candidates": candidate_comparison,
                        },
                        "threshold_selection": thr_info,
                        "prior_correction": prior_info,
                    }
                    if ocsvm_ecdf_risk is not None:
                        metrics["ecdf_risk"] = {
                            "method": "ecdf_quantile",
                            "n_ref": int(ocsvm_ecdf_risk.get("n_ref", 0)),
                            "n_knots": int(ocsvm_ecdf_risk.get("n_knots", 0)),
                        }
                    if logit_metrics is not None:
                        metrics["attack_logistic"] = logit_metrics
                        metrics["attack_logistic_used"] = bool((logit_params is not None) and use_logit)
                        metrics["attack_logistic_reason"] = logit_reason
                    metrics["temperature_scaling"] = temp_metrics

                    logger.info(
                        "[retrain][OCSVM] val runtime metrics: ROC_AUC=%s PR_AUC=%s f1@thr=%s thr=%.3f pos=%s n=%s",
                        f"{oc_metrics_runtime.get('roc_auc'):.4f}" if oc_metrics_runtime.get("roc_auc") is not None else None,
                        f"{oc_metrics_runtime.get('pr_auc'):.4f}" if oc_metrics_runtime.get("pr_auc") is not None else None,
                        metrics.get("f1"),
                        recommended_threshold if recommended_threshold is not None else -1.0,
                        int(y_val.sum()),
                        len(y_val),
                    )
                else:
                    logger.info(
                        "[retrain][OCSVM] val has only one class in label, skipping ROC/PR/F1 metrics (path=%s)",
                        val_csv_path,
                    )
            else:
                logger.info(
                    "[retrain][OCSVM] no label column in val CSV, skip label-based metrics (path=%s)",
                    val_csv_path,
                )
        except Exception as e:
            logger.info(
                "[retrain][OCSVM] failed to compute label-based metrics from %s: %s",
                val_csv_path,
                e,
            )

    # =========================
    # TEST EVAL (untouched, optional)
    # =========================
    if (
        RETRAIN_ENABLE_TEST_EVAL
        and
        test_csv_path is not None
        and Path(test_csv_path).exists()
        and recommended_threshold is not None
    ):
        try:
            df_te = pd.read_csv(test_csv_path, low_memory=False)
            y_te = _labels_from_df(df_te)
            if y_te is not None and len(np.unique(y_te)) >= 2:
                df_te, y_te = _maybe_downsample_df(
                    df_te,
                    y_te,
                    ML_TEST_MAX_ROWS,
                    tag="OCSVM/test",
                )
                X_te = ensure_columns(df_te, feature_order).fillna(0.0).astype(float).values
                stat_te = _raw_anomaly_stat(model, X_te, kind="ocsvm")
                scores_te = score_from_stat_with_calib(
                    stat_te,
                    calib,
                    kind="ocsvm",
                )

                test_score_eval = _eval_scores_only_with_labels(scores_te, y_te)
                test_metrics = metrics_on_threshold(y_te, scores_te, recommended_threshold)
                if metrics is None:
                    metrics = {}
                metrics["test"] = {
                    **(test_score_eval or {}),
                    **test_metrics,
                    "test_metrics_from": str(test_csv_path),
                    "threshold_used": float(recommended_threshold),
                }

                logger.info(
                    "[retrain][OCSVM][TEST] ROC_AUC=%s PR_AUC=%s F1@thr=%s thr=%.3f pos=%s n=%s",
                    f"{test_metrics.get('roc_auc'):.4f}" if test_metrics.get("roc_auc") is not None else None,
                    f"{test_metrics.get('pr_auc'):.4f}" if test_metrics.get("pr_auc") is not None else None,
                    f"{test_metrics.get('f1'):.4f}" if test_metrics.get("f1") is not None else None,
                    float(recommended_threshold),
                    int(y_te.sum()),
                    len(y_te),
                )
            else:
                logger.info("[retrain][OCSVM][TEST] no usable labels or only one class; skip")
        except Exception as e:
            logger.info("[retrain][OCSVM][TEST] failed to compute test metrics: %s", e)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, outp)
    logger.info(f"[retrain] saved OCSVM -> {outp}")

    write_ocsvm_meta(
        outp,
        feature_order,
        model,
        calibration=calib,
        metrics=metrics,
        recommended_threshold=recommended_threshold,
    )
    return model, calib, recommended_threshold, metrics

def _labels_from_df(df: pd.DataFrame) -> np.ndarray | None:
    """
    Extracts binary labels from DataFrame 'label' column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        np.ndarray | None: Binary labels array or None if 'label' column not present.
    """
    if "label" not in df.columns:
        return None
    y_raw = pd.to_numeric(df["label"], errors="coerce").fillna(0.0)
    return (y_raw > 0.0).astype(int).to_numpy()

def _eval_and_plot_scores_with_labels(
    scores: np.ndarray,
    labels: np.ndarray,
    out_prefix: str,
    title_prefix: str,
) -> dict:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)

    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask]
    labels = labels[mask]

    if scores.size == 0 or labels.size == 0:
        logger.info("[retrain][%s eval] empty scores/labels after cleaning", title_prefix)
        return {}

    scores_prob = np.clip(scores, 1e-9, 1.0 - 1e-9)
    log_loss = _binary_logloss(labels, scores_prob)
    brier = float(np.mean((scores_prob - labels) ** 2))
    ece15 = _expected_calibration_error(labels, scores_prob, n_bins=15)

    try:
        roc = roc_auc_score(labels, scores)
    except Exception as e:
        logger.info(f"[retrain][{title_prefix} eval] roc_auc_score failed: {e}")
        roc = None

    try:
        ap = average_precision_score(labels, scores)
    except Exception as e:
        logger.info(f"[retrain][{title_prefix} eval] average_precision_score failed: {e}")
        ap = None

    prec, rec, thr = precision_recall_curve(labels, scores)
    denom = prec + rec
    f1s = np.zeros_like(denom)
    valid = denom > 0
    f1s[valid] = 2.0 * prec[valid] * rec[valid] / denom[valid]
    best_f1 = float(f1s.max()) if len(f1s) else None
    thr_for_f1 = np.append(thr, [thr[-1] if len(thr) else 0.5])
    best_idx = int(np.argmax(f1s)) if len(f1s) else 0
    best_thr = float(thr_for_f1[best_idx]) if len(thr_for_f1) else 0.5

    logger.info(
        "[retrain][%s eval] n=%s pos=%s ROC_AUC=%s PR_AUC=%s logloss=%.6f brier=%.6f ece15=%.6f best_F1=%s best_thr=%.3f",
        title_prefix,
        len(labels),
        int(labels.sum()),
        f"{roc:.4f}" if roc is not None else None,
        f"{ap:.4f}" if ap is not None else None,
        float(log_loss),
        float(brier),
        float(ece15) if np.isfinite(ece15) else float("nan"),
        best_f1,
        best_thr,
    )

    # CSV
    out_scores_csv = f"{out_prefix}_scores.csv"
    Path(out_scores_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"score": scores, "label": labels}).to_csv(out_scores_csv, index=False)
    logger.info(f"[retrain][{title_prefix} eval] scores saved -> {out_scores_csv}")

    try:
        # ROC
        if roc is not None:
            fpr, tpr, _ = roc_curve(labels, scores)
            plt.figure()
            plt.plot(fpr, tpr, label=f"{title_prefix} (AUC={roc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{title_prefix} ROC")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_roc.png")
            plt.close()

        # PR
        if ap is not None:
            plt.figure()
            plt.step(rec, prec, where="post")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"{title_prefix} PR (AP={ap:.3f})")
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_pr.png")
            plt.close()

        # Hist
        plt.figure()
        plt.hist(scores[labels == 0], bins=100, alpha=0.7, label="label=0 (benign)")
        if (labels == 1).any():
            plt.hist(scores[labels == 1], bins=100, alpha=0.7, label="label=1 (attack)")
        plt.xlabel("calibrated anomaly risk score")
        plt.ylabel("Count")
        plt.title(f"{title_prefix} score distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_hist.png")
        plt.close()
    except Exception as e:
        logger.info(f"[retrain][{title_prefix} eval] plotting failed: {e}")

    return {
        "roc_auc": float(roc) if roc is not None else None,
        "pr_auc": float(ap) if ap is not None else None,
        "log_loss": float(log_loss),
        "brier": float(brier),
        "ece_15": float(ece15) if np.isfinite(ece15) else None,
        "best_f1": best_f1,
        "best_thr": best_thr,
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_p01": float(np.quantile(scores, 0.01)),
        "score_p99": float(np.quantile(scores, 0.99)),
        "pct_near_0_5_eps1e3": float(np.mean(np.abs(scores - 0.5) < 1e-3)),
        "pct_near_1_0_eps1e3": float(np.mean(scores > (1.0 - 1e-3))),
        "pct_near_1_0_benign_eps1e3": float(np.mean(scores[labels == 0] > (1.0 - 1e-3))) if np.any(labels == 0) else None,
        "pct_near_1_0_attack_eps1e3": float(np.mean(scores[labels == 1] > (1.0 - 1e-3))) if np.any(labels == 1) else None,
        "uniq_1e4": int(np.unique(np.round(scores, 4)).size),
        "rows": int(len(labels)),
        "positives": int(labels.sum()),
    }

def metrics_on_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    out = _binary_metrics_at_threshold(y_true, scores, thr)
    if ML_BOOTSTRAP_CI_ENABLED:
        ci = _bootstrap_ci_for_threshold_metrics(
            y_true,
            scores,
            thr=thr,
            n_boot=ML_BOOTSTRAP_CI_SAMPLES,
            alpha=ML_BOOTSTRAP_CI_ALPHA,
            random_state=ML_BOOTSTRAP_CI_RANDOM_STATE,
            max_rows=ML_BOOTSTRAP_CI_MAX_ROWS,
        )
        if ci is not None:
            out["bootstrap_ci"] = ci
    return out


def train_autoencoder(
    train_csv_path: str,
    val_csv_path: str | None,
    test_csv_path: str | None,
    out_path: str,
    feature_order: list[str],
):
    """
    Dense autoencoder:
      - train on benign rows from train CSV (ml_numeric),
      - calibrate reconstruction error on benign train,
      - optionally compute metrics on separate validation CSV (ml_validation),
      - fit attack_logistic calibrator recon_err -> P(attack),
    """
    # === TRAIN (benign) ===
    df_tr = pd.read_csv(train_csv_path, low_memory=False)
    X_tr_full = ensure_columns(df_tr, feature_order).fillna(0.0).to_numpy(dtype=float)
    y_tr = _labels_from_df(df_tr)

    if y_tr is not None and (y_tr == 0).any():
        X_train = X_tr_full[y_tr == 0]
        logger.info(f"[retrain][AE] using {X_train.shape[0]} normal rows (label=0) for training")
    else:
        X_train = X_tr_full
        logger.info(f"[retrain][AE] label not present or all=1, training on all rows: n={X_train.shape[0]}")

    max_train = int(os.getenv("AE_MAX_TRAIN_ROWS", "300000"))
    if X_train.shape[0] > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_train.shape[0], size=max_train, replace=False)
        X_train = X_train[idx]
        logger.info(f"[retrain][AE] sampling {max_train} rows for training (from {X_train.shape[0]})")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    input_dim = X_train_sc.shape[1]

    # Hyperparameters
    ae_hidden1 = int(os.getenv("AE_HIDDEN1", "64"))
    ae_hidden2 = int(os.getenv("AE_HIDDEN2", "32"))
    ae_code    = int(os.getenv("AE_CODE_DIM", "16"))
    ae_epochs  = int(os.getenv("AE_EPOCHS", "20"))
    ae_batch   = int(os.getenv("AE_BATCH", "64"))

    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(ae_hidden1, activation="relu")(inp)
    x = keras.layers.Dense(ae_hidden2, activation="relu")(x)
    code = keras.layers.Dense(ae_code, activation="relu")(x)
    x = keras.layers.Dense(ae_hidden2, activation="relu")(code)
    x = keras.layers.Dense(ae_hidden1, activation="relu")(x)
    out = keras.layers.Dense(input_dim, activation="linear")(x)

    ae_model = keras.Model(inp, out, name="dense_autoencoder")
    ae_model.compile(optimizer="adam", loss="mse")

    logger.info(f"[retrain][AE] training Dense AE: input_dim={input_dim}, code_dim={ae_code}, epochs={ae_epochs}, batch={ae_batch}")
    ae_history = ae_model.fit(
        X_train_sc,
        X_train_sc,
        epochs=ae_epochs,
        batch_size=ae_batch,
        shuffle=True,
        validation_split=0.1,
        verbose=1,
    )
    ae_history_meta = _save_keras_training_history(
        ae_history,
        out_prefix="/models/ml/ae_train",
        title_prefix="DenseAE",
    )

    # === CALIBRATION on benign train ===
    X_rec_train = ae_model.predict(X_train_sc, batch_size=ae_batch, verbose=0)
    recon_err_train = np.mean((X_train_sc - X_rec_train) ** 2, axis=1)

    # Robust calibration statistics (median + IQR-based sigma) are much more
    # stable for heavy-tailed reconstruction errors than mean/std.
    calib_mu = float(np.median(recon_err_train))
    q25, q75 = np.quantile(recon_err_train, [0.25, 0.75])
    calib_sigma = float((q75 - q25) / 1.349)  # IQR-to-sigma conversion
    if (not np.isfinite(calib_sigma)) or calib_sigma <= 1e-9:
        calib_sigma = float(np.std(recon_err_train) or 1e-6)
    calib = {
        "mu": calib_mu,
        "sigma": calib_sigma,
        "orientation": "identity",
        "scale_method": "median_iqr",
    }
    ae_ecdf_risk = fit_ecdf_risk_mapping(recon_err_train)

    metrics = {}
    if ae_history_meta:
        metrics["training_history"] = ae_history_meta
    recommended_threshold = None

    if val_csv_path is not None and Path(val_csv_path).exists():
        try:
            df_val = pd.read_csv(val_csv_path, low_memory=False)
            y_val = _labels_from_df(df_val)
            df_val, y_val = _maybe_downsample_df(
                df_val,
                y_val,
                ML_VAL_MAX_ROWS,
                tag="AE/val",
            )
            X_val = ensure_columns(df_val, feature_order).fillna(0.0).to_numpy(dtype=float)

            if y_val is not None:
                X_val_sc = scaler.transform(X_val)
                X_val_rec = ae_model.predict(X_val_sc, batch_size=ae_batch, verbose=0)
                recon_err_val = np.mean((X_val_sc - X_val_rec) ** 2, axis=1)

                # 1) baseline eval on mu/sigma only
                calib_mu_sigma = dict(calib or {})
                calib_mu_sigma.pop("attack_logistic", None)
                scores_val_mu_sigma = score_from_stat_with_calib(
                    recon_err_val,
                    calib_mu_sigma,
                    kind="ae",
                )
                ae_eval_mu_sigma = _eval_and_plot_scores_with_labels(
                    scores_val_mu_sigma,
                    y_val,
                    out_prefix="/models/ml/ae_eval_mu_sigma",
                    title_prefix="DenseAE(mu_sigma)",
                )

                # 2) fit logistic on oriented reconstruction error (Platt-style, 1D)
                logit_params, logit_metrics, logit_thr = fit_attack_logistic_calibrator(
                    recon_err_val,
                    y_val,
                    kind="ae",
                    orientation=_resolve_score_orientation("ae", calibration=calib_mu_sigma),
                )

                # 3) select runtime calibration path by proper scoring rules on calibration set.
                # Candidates:
                #   - mu_sigma (+ optional temperature)
                #   - attack_logistic (+ optional temperature), if available
                #   - ecdf_risk (+ optional temperature), benign-train quantile map
                candidate_runs: list[dict] = []
                for cand_name in ("mu_sigma", "attack_logistic", "ecdf_risk"):
                    if cand_name == "attack_logistic" and logit_params is None:
                        continue
                    if cand_name == "ecdf_risk" and ae_ecdf_risk is None:
                        continue
                    cand_calib = dict(calib_mu_sigma or {})
                    cand_calib.pop("temperature", None)
                    if cand_name == "attack_logistic":
                        cand_calib["attack_logistic"] = dict(logit_params)
                        cand_calib.pop("ecdf_risk", None)
                    elif cand_name == "ecdf_risk":
                        cand_calib.pop("attack_logistic", None)
                        cand_calib["ecdf_risk"] = dict(ae_ecdf_risk)
                    else:
                        cand_calib.pop("attack_logistic", None)
                        cand_calib.pop("ecdf_risk", None)

                    cand_scores = score_from_stat_with_calib(
                        recon_err_val,
                        cand_calib,
                        kind="ae",
                    )
                    cand_temp_t, cand_temp_metrics = fit_temperature_scaler(
                        cand_scores,
                        y_val,
                        kind=f"ae/{cand_name}",
                    )
                    if cand_temp_metrics.get("used"):
                        cand_calib["temperature"] = float(cand_temp_t)
                        cand_scores = score_from_stat_with_calib(
                            recon_err_val,
                            cand_calib,
                            kind="ae",
                        )
                    else:
                        cand_calib.pop("temperature", None)

                    cand_eval = _eval_scores_only_with_labels(cand_scores, y_val)
                    candidate_runs.append(
                        {
                            "name": cand_name,
                            "calibration": cand_calib,
                            "scores": cand_scores,
                            "eval": cand_eval,
                            "temperature_scaling": cand_temp_metrics,
                        }
                    )

                best_candidate, candidate_comparison, selection_reason = _select_best_runtime_calibration_candidate(
                    candidate_runs
                )
                runtime_score_type = str(best_candidate.get("name", "mu_sigma"))
                use_logit = runtime_score_type == "attack_logistic"
                logit_reason = selection_reason
                calib = dict(best_candidate.get("calibration") or dict(calib_mu_sigma or {}))
                temp_metrics = best_candidate.get("temperature_scaling") or {
                    "used": False,
                    "temperature": 1.0,
                }
                scores_val_runtime = np.asarray(
                    best_candidate.get("scores", scores_val_mu_sigma),
                    dtype=float,
                ).reshape(-1)

                logger.info(
                    "[retrain][AE] runtime calibration selected: %s (%s)",
                    runtime_score_type,
                    logit_reason,
                )

                # Optional prior correction (calibration prior -> deployment prior).
                calib, prior_info = _attach_prior_correction_to_calibration(
                    calib,
                    y_val,
                    kind="ae",
                )
                scores_val_runtime = score_from_stat_with_calib(
                    recon_err_val,
                    calib,
                    kind="ae",
                )

                # 4) publish runtime artifacts for the selected path
                ae_eval_runtime = _eval_and_plot_scores_with_labels(
                    scores_val_runtime,
                    y_val,
                    out_prefix="/models/ml/ae_eval",
                    title_prefix="DenseAE(runtime)",
                )

                # Choose operating threshold by configured policy.
                recommended_threshold, thr_info = choose_operating_threshold(y_val, scores_val_runtime)
                _, bestf1_ref = best_threshold_f1(y_val, scores_val_runtime)

                val_metrics = {
                    **metrics_on_threshold(y_val, scores_val_runtime, recommended_threshold),
                    "best_f1_on_val": float(bestf1_ref),
                    "calibrated_eval": ae_eval_runtime,
                    "eval_mu_sigma": ae_eval_mu_sigma,
                    "runtime_score_type": runtime_score_type,
                    "label_positives": int(y_val.sum()),
                    "label_rows": int(len(y_val)),
                    "label_metrics_from": str(val_csv_path),
                    "calibration_selection": {
                        "method": "min_log_loss_then_brier",
                        "reason": logit_reason,
                        "candidates": candidate_comparison,
                    },
                    "threshold_selection": thr_info,
                    "prior_correction": prior_info,
                }
                if ae_ecdf_risk is not None:
                    val_metrics["ecdf_risk"] = {
                        "method": "ecdf_quantile",
                        "n_ref": int(ae_ecdf_risk.get("n_ref", 0)),
                        "n_knots": int(ae_ecdf_risk.get("n_knots", 0)),
                    }
                if logit_metrics is not None:
                    val_metrics["attack_logistic"] = logit_metrics
                    val_metrics["attack_logistic_used"] = bool((logit_params is not None) and use_logit)
                    val_metrics["attack_logistic_reason"] = logit_reason
                val_metrics["temperature_scaling"] = temp_metrics
                metrics.update(val_metrics)

            else:
                logger.info(
                    "[retrain][AE] no usable label in val CSV, skipping label/logistic metrics (path=%s)",
                    val_csv_path,
                )
        except Exception as e:
            logger.info(f"[retrain][AE] failed to compute label/logistic metrics: {e}")
    # =========================
    # TEST EVAL (untouched, optional)
    # =========================
    if (
        RETRAIN_ENABLE_TEST_EVAL
        and
        test_csv_path is not None
        and Path(test_csv_path).exists()
        and recommended_threshold is not None
    ):
        try:
            df_te = pd.read_csv(test_csv_path, low_memory=False)
            y_te = _labels_from_df(df_te)
            df_te, y_te = _maybe_downsample_df(
                df_te,
                y_te,
                ML_TEST_MAX_ROWS,
                tag="AE/test",
            )
            X_te = ensure_columns(df_te, feature_order).fillna(0.0).to_numpy(dtype=float)

            if y_te is not None and len(np.unique(y_te)) >= 2:
                X_te_sc = scaler.transform(X_te)
                X_te_rec = ae_model.predict(X_te_sc, batch_size=ae_batch, verbose=0)
                recon_err_te = np.mean((X_te_sc - X_te_rec) ** 2, axis=1)

                scores_te = score_from_stat_with_calib(
                    recon_err_te,
                    calib,          # IMPORTANT: final calib (may include attack_logistic)
                    kind="ae",
                )

                test_metrics = metrics_on_threshold(y_te, scores_te, recommended_threshold)

                metrics["test"] = {
                    **test_metrics,
                    "test_metrics_from": str(test_csv_path),
                    "threshold_used": float(recommended_threshold),
                }

                logger.info(
                    "[retrain][AE][TEST] ROC_AUC=%s PR_AUC=%s F1@thr=%s thr=%.3f pos=%s n=%s",
                    f"{test_metrics.get('roc_auc'):.4f}" if test_metrics.get("roc_auc") is not None else None,
                    f"{test_metrics.get('pr_auc'):.4f}" if test_metrics.get("pr_auc") is not None else None,
                    f"{test_metrics.get('f1'):.4f}" if test_metrics.get("f1") is not None else None,
                    float(recommended_threshold),
                    int(y_te.sum()),
                    len(y_te),
                )
            else:
                logger.info("[retrain][AE][TEST] no usable labels or only one class; skip")
        except Exception as e:
            logger.info("[retrain][AE][TEST] failed to compute test metrics: %s", e)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "type": "dense_autoencoder",
        "feature_order": feature_order,
        "scaler": scaler,
        "model": ae_model,
    }
    joblib.dump(bundle, outp)
    logger.info(f"[retrain][AE] saved Dense AE -> {outp}")

    write_ae_meta(
        outp,
        feature_order,
        input_dim=input_dim,
        model=ae_model,
        calibration=calib,
        metrics=metrics,
        recommended_threshold=recommended_threshold,
    )
    return bundle, calib, recommended_threshold, metrics

def train_lstm_autoencoder(
    train_csv_path: str,
    val_csv_path: str | None,
    test_csv_path: str | None,
    out_path: str,
    feature_order: list[str],
):
    """
    LSTM autoencoder:
      - train on benign rows from train CSV (ml_numeric),
      - calibrate reconstruction error on benign train,
      - optionally compute metrics on separate validation CSV (ml_validation),
      - fit attack_logistic calibrator recon_err -> P(attack),
    """
    df_tr = pd.read_csv(train_csv_path, low_memory=False)
    X_tr_full = ensure_columns(df_tr, feature_order).fillna(0.0).to_numpy(dtype=float)
    y_tr = _labels_from_df(df_tr)

    if y_tr is not None and (y_tr == 0).any():
        X_train = X_tr_full[y_tr == 0]
        logger.info(f"[retrain][LSTM-AE] using {X_train.shape[0]} normal rows (label=0) for training")
    else:
        X_train = X_tr_full
        logger.info(f"[retrain][LSTM-AE] label not present or all=1, training on all rows: n={X_train.shape[0]}")

    max_train = int(os.getenv("LSTM_AE_MAX_TRAIN_ROWS", "200000"))
    if X_train.shape[0] > max_train:
        rng = np.random.RandomState(44)
        idx = rng.choice(X_train.shape[0], size=max_train, replace=False)
        X_train = X_train[idx]
        logger.info(f"[retrain][LSTM-AE] sampling {max_train} rows for training (from {X_train.shape[0]})")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    timesteps = X_train_sc.shape[1]
    X_train_seq = X_train_sc.reshape((-1, timesteps, 1))

    lstm_latent = int(os.getenv("LSTM_AE_LATENT", "16"))
    lstm_epochs = int(os.getenv("LSTM_AE_EPOCHS", "15"))
    lstm_batch  = int(os.getenv("LSTM_AE_BATCH", "64"))

    inp = keras.Input(shape=(timesteps, 1))
    x = keras.layers.LSTM(lstm_latent * 2, return_sequences=True)(inp)
    x = keras.layers.LSTM(lstm_latent, return_sequences=False)(x)
    x = keras.layers.RepeatVector(timesteps)(x)
    x = keras.layers.LSTM(lstm_latent, return_sequences=True)(x)
    out = keras.layers.TimeDistributed(keras.layers.Dense(1))(x)

    lstm_ae = keras.Model(inp, out, name="lstm_autoencoder")
    lstm_ae.compile(optimizer="adam", loss="mse")

    logger.info(f"[retrain][LSTM-AE] training LSTM AE: timesteps={timesteps}, latent={lstm_latent}, epochs={lstm_epochs}, batch={lstm_batch}")
    lstm_history = lstm_ae.fit(
        X_train_seq,
        X_train_seq,
        epochs=lstm_epochs,
        batch_size=lstm_batch,
        shuffle=True,
        validation_split=0.1,
        verbose=1,
    )
    lstm_history_meta = _save_keras_training_history(
        lstm_history,
        out_prefix="/models/ml/lstm_ae_train",
        title_prefix="LSTM-AE",
    )

    X_rec_train = lstm_ae.predict(X_train_seq, batch_size=lstm_batch, verbose=0)
    recon_err_train = np.mean((X_train_seq - X_rec_train) ** 2, axis=(1, 2))

    # Robust calibration statistics (median + IQR-based sigma) are much more
    # stable for heavy-tailed reconstruction errors than mean/std.
    calib_mu = float(np.median(recon_err_train))
    q25, q75 = np.quantile(recon_err_train, [0.25, 0.75])
    calib_sigma = float((q75 - q25) / 1.349)  # IQR-to-sigma conversion
    if (not np.isfinite(calib_sigma)) or calib_sigma <= 1e-9:
        calib_sigma = float(np.std(recon_err_train) or 1e-6)
    calib = {
        "mu": calib_mu,
        "sigma": calib_sigma,
        "orientation": "identity",
        "scale_method": "median_iqr",
    }
    lstm_ecdf_risk = fit_ecdf_risk_mapping(recon_err_train)

    metrics = {}
    if lstm_history_meta:
        metrics["training_history"] = lstm_history_meta
    recommended_threshold = None

    if val_csv_path is not None and Path(val_csv_path).exists():
        try:
            df_val = pd.read_csv(val_csv_path, low_memory=False)
            y_val = _labels_from_df(df_val)
            df_val, y_val = _maybe_downsample_df(
                df_val,
                y_val,
                ML_VAL_MAX_ROWS,
                tag="LSTM-AE/val",
            )
            X_val = ensure_columns(df_val, feature_order).fillna(0.0).to_numpy(dtype=float)

            if y_val is not None:
                X_val_sc  = scaler.transform(X_val)
                X_val_seq = X_val_sc.reshape((-1, timesteps, 1))

                X_rec_val = lstm_ae.predict(X_val_seq, batch_size=lstm_batch, verbose=0)
                recon_err_val = np.mean((X_val_seq - X_rec_val) ** 2, axis=(1, 2))

                # 1) baseline eval on mu/sigma only
                calib_mu_sigma = dict(calib or {})
                calib_mu_sigma.pop("attack_logistic", None)
                scores_val_mu_sigma = score_from_stat_with_calib(
                    recon_err_val,
                    calib_mu_sigma,
                    kind="lstm_ae",
                )
                lstm_eval_mu_sigma = _eval_and_plot_scores_with_labels(
                    scores_val_mu_sigma,
                    y_val,
                    out_prefix="/models/ml/lstm_ae_eval_mu_sigma",
                    title_prefix="LSTM-AE(mu_sigma)",
                )

                # 2) fit logistic on oriented reconstruction error (Platt-style, 1D)
                logit_params, logit_metrics, logit_thr = fit_attack_logistic_calibrator(
                    recon_err_val,
                    y_val,
                    kind="lstm_ae",
                    orientation=_resolve_score_orientation("lstm_ae", calibration=calib_mu_sigma),
                )

                # 3) select runtime calibration path by proper scoring rules on calibration set.
                candidate_runs: list[dict] = []
                for cand_name in ("mu_sigma", "attack_logistic", "ecdf_risk"):
                    if cand_name == "attack_logistic" and logit_params is None:
                        continue
                    if cand_name == "ecdf_risk" and lstm_ecdf_risk is None:
                        continue
                    cand_calib = dict(calib_mu_sigma or {})
                    cand_calib.pop("temperature", None)
                    if cand_name == "attack_logistic":
                        cand_calib["attack_logistic"] = dict(logit_params)
                        cand_calib.pop("ecdf_risk", None)
                    elif cand_name == "ecdf_risk":
                        cand_calib.pop("attack_logistic", None)
                        cand_calib["ecdf_risk"] = dict(lstm_ecdf_risk)
                    else:
                        cand_calib.pop("attack_logistic", None)
                        cand_calib.pop("ecdf_risk", None)

                    cand_scores = score_from_stat_with_calib(
                        recon_err_val,
                        cand_calib,
                        kind="lstm_ae",
                    )
                    cand_temp_t, cand_temp_metrics = fit_temperature_scaler(
                        cand_scores,
                        y_val,
                        kind=f"lstm_ae/{cand_name}",
                    )
                    if cand_temp_metrics.get("used"):
                        cand_calib["temperature"] = float(cand_temp_t)
                        cand_scores = score_from_stat_with_calib(
                            recon_err_val,
                            cand_calib,
                            kind="lstm_ae",
                        )
                    else:
                        cand_calib.pop("temperature", None)

                    cand_eval = _eval_scores_only_with_labels(cand_scores, y_val)
                    candidate_runs.append(
                        {
                            "name": cand_name,
                            "calibration": cand_calib,
                            "scores": cand_scores,
                            "eval": cand_eval,
                            "temperature_scaling": cand_temp_metrics,
                        }
                    )

                best_candidate, candidate_comparison, selection_reason = _select_best_runtime_calibration_candidate(
                    candidate_runs
                )
                runtime_score_type = str(best_candidate.get("name", "mu_sigma"))
                use_logit = runtime_score_type == "attack_logistic"
                logit_reason = selection_reason
                calib = dict(best_candidate.get("calibration") or dict(calib_mu_sigma or {}))
                temp_metrics = best_candidate.get("temperature_scaling") or {
                    "used": False,
                    "temperature": 1.0,
                }
                scores_val_runtime = np.asarray(
                    best_candidate.get("scores", scores_val_mu_sigma),
                    dtype=float,
                ).reshape(-1)

                logger.info(
                    "[retrain][LSTM-AE] runtime calibration selected: %s (%s)",
                    runtime_score_type,
                    logit_reason,
                )

                # Optional prior correction (calibration prior -> deployment prior).
                calib, prior_info = _attach_prior_correction_to_calibration(
                    calib,
                    y_val,
                    kind="lstm_ae",
                )
                scores_val_runtime = score_from_stat_with_calib(
                    recon_err_val,
                    calib,
                    kind="lstm_ae",
                )

                # 4) publish runtime artifacts for the selected path
                lstm_eval_runtime = _eval_and_plot_scores_with_labels(
                    scores_val_runtime,
                    y_val,
                    out_prefix="/models/ml/lstm_ae_eval",
                    title_prefix="LSTM-AE(runtime)",
                )

                # Choose operating threshold by configured policy.
                recommended_threshold, thr_info = choose_operating_threshold(y_val, scores_val_runtime)
                _, bestf1_ref = best_threshold_f1(y_val, scores_val_runtime)

                val_metrics = {
                    **metrics_on_threshold(y_val, scores_val_runtime, recommended_threshold),
                    "best_f1_on_val": float(bestf1_ref),
                    "calibrated_eval": lstm_eval_runtime,
                    "eval_mu_sigma": lstm_eval_mu_sigma,
                    "runtime_score_type": runtime_score_type,
                    "label_positives": int(y_val.sum()),
                    "label_rows": int(len(y_val)),
                    "label_metrics_from": str(val_csv_path),
                    "calibration_selection": {
                        "method": "min_log_loss_then_brier",
                        "reason": logit_reason,
                        "candidates": candidate_comparison,
                    },
                    "threshold_selection": thr_info,
                    "prior_correction": prior_info,
                }
                if lstm_ecdf_risk is not None:
                    val_metrics["ecdf_risk"] = {
                        "method": "ecdf_quantile",
                        "n_ref": int(lstm_ecdf_risk.get("n_ref", 0)),
                        "n_knots": int(lstm_ecdf_risk.get("n_knots", 0)),
                    }
                if logit_metrics is not None:
                    val_metrics["attack_logistic"] = logit_metrics
                    val_metrics["attack_logistic_used"] = bool((logit_params is not None) and use_logit)
                    val_metrics["attack_logistic_reason"] = logit_reason
                val_metrics["temperature_scaling"] = temp_metrics
                metrics.update(val_metrics)

            else:
                logger.info(
                    "[retrain][LSTM-AE] no usable label in val CSV, skipping label/logistic metrics (path=%s)",
                    val_csv_path,
                )
        except Exception as e:
            logger.info(f"[retrain][LSTM-AE] failed to compute label/logistic metrics: {e}")
    # =========================
    # TEST EVAL (untouched, optional)
    # =========================
    if (
        RETRAIN_ENABLE_TEST_EVAL
        and
        test_csv_path is not None
        and Path(test_csv_path).exists()
        and recommended_threshold is not None
    ):
        try:
            df_te = pd.read_csv(test_csv_path, low_memory=False)
            y_te = _labels_from_df(df_te)
            df_te, y_te = _maybe_downsample_df(
                df_te,
                y_te,
                ML_TEST_MAX_ROWS,
                tag="LSTM-AE/test",
            )
            X_te = ensure_columns(df_te, feature_order).fillna(0.0).to_numpy(dtype=float)

            if y_te is not None and len(np.unique(y_te)) >= 2:
                X_te_sc  = scaler.transform(X_te)
                X_te_seq = X_te_sc.reshape((-1, timesteps, 1))

                X_rec_te = lstm_ae.predict(X_te_seq, batch_size=lstm_batch, verbose=0)
                recon_err_te = np.mean((X_te_seq - X_rec_te) ** 2, axis=(1, 2))

                scores_te = score_from_stat_with_calib(
                    recon_err_te,
                    calib,          # IMPORTANT: final calib (may include attack_logistic)
                    kind="lstm_ae",
                )

                test_metrics = metrics_on_threshold(y_te, scores_te, recommended_threshold)

                metrics["test"] = {
                    **test_metrics,
                    "test_metrics_from": str(test_csv_path),
                    "threshold_used": float(recommended_threshold),
                }

                logger.info(
                    "[retrain][LSTM-AE][TEST] ROC_AUC=%s PR_AUC=%s F1@thr=%s thr=%.3f pos=%s n=%s",
                    f"{test_metrics.get('roc_auc'):.4f}" if test_metrics.get("roc_auc") is not None else None,
                    f"{test_metrics.get('pr_auc'):.4f}" if test_metrics.get("pr_auc") is not None else None,
                    f"{test_metrics.get('f1'):.4f}" if test_metrics.get("f1") is not None else None,
                    float(recommended_threshold),
                    int(y_te.sum()),
                    len(y_te),
                )
            else:
                logger.info("[retrain][LSTM-AE][TEST] no usable labels or only one class; skip")
        except Exception as e:
            logger.info("[retrain][LSTM-AE][TEST] failed to compute test metrics: %s", e)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "type": "lstm_autoencoder",
        "feature_order": feature_order,
        "scaler": scaler,
        "timesteps": timesteps,
        "model": lstm_ae,
    }
    joblib.dump(bundle, outp)
    logger.info(f"[retrain][LSTM-AE] saved LSTM AE -> {outp}")

    write_lstm_ae_meta(
        outp,
        feature_order,
        timesteps=timesteps,
        model=lstm_ae,
        calibration=calib,
        metrics=metrics,
        recommended_threshold=recommended_threshold,
    )
    return bundle, calib, recommended_threshold, metrics
  
def _collect_ensemble_model_scores(
    df_eval: pd.DataFrame,
    feature_order: list[str],
    if_model=None,
    if_calib: dict | None = None,
    ocsvm_model=None,
    ocsvm_calib: dict | None = None,
    ae_bundle=None,
    ae_calib: dict | None = None,
    lstm_bundle=None,
    lstm_calib: dict | None = None,
    supervised_models: dict | None = None,
    supervised_calibs: dict | None = None,
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    y = _labels_from_df(df_eval)
    if y is None:
        return None, {}

    X = ensure_columns(df_eval, feature_order).fillna(0.0).astype(float).values
    per_model: dict[str, np.ndarray] = {}

    if if_model is not None:
        try:
            stats_if = _raw_anomaly_stat(if_model, X, kind="if")
            per_model["if"] = score_from_stat_with_calib(stats_if, if_calib, kind="if")
        except Exception as e:
            logger.info("[retrain][ML-ensemble] IF scoring failed: %s", e)

    if ocsvm_model is not None:
        try:
            stats_oc = _raw_anomaly_stat(ocsvm_model, X, kind="ocsvm")
            per_model["ocsvm"] = score_from_stat_with_calib(stats_oc, ocsvm_calib, kind="ocsvm")
        except Exception as e:
            logger.info("[retrain][ML-ensemble] OCSVM scoring failed: %s", e)

    if ae_bundle is not None:
        try:
            scaler_ae = ae_bundle["scaler"]
            model_ae = ae_bundle["model"]
            ae_batch = int(os.getenv("AE_BATCH", "64"))
            X_sc = scaler_ae.transform(X)
            X_rec = model_ae.predict(X_sc, batch_size=ae_batch, verbose=0)
            recon_err = np.mean((X_sc - X_rec) ** 2, axis=1)
            per_model["ae"] = score_from_stat_with_calib(recon_err, ae_calib, kind="ae")
        except Exception as e:
            logger.info("[retrain][ML-ensemble] AE scoring failed: %s", e)

    if lstm_bundle is not None:
        try:
            scaler_l = lstm_bundle["scaler"]
            model_l = lstm_bundle["model"]
            timesteps = int(lstm_bundle["timesteps"])
            lstm_batch = int(os.getenv("LSTM_AE_BATCH", "64"))
            X_sc = scaler_l.transform(X)
            X_seq = X_sc.reshape((-1, timesteps, 1))
            X_rec = model_l.predict(X_seq, batch_size=lstm_batch, verbose=0)
            recon_err = np.mean((X_seq - X_rec) ** 2, axis=(1, 2))
            per_model["lstm_ae"] = score_from_stat_with_calib(recon_err, lstm_calib, kind="lstm_ae")
        except Exception as e:
            logger.info("[retrain][ML-ensemble] LSTM-AE scoring failed: %s", e)

    if isinstance(supervised_models, dict):
        sup_calibs = supervised_calibs or {}
        for kind, model in supervised_models.items():
            if model is None:
                continue
            try:
                p_raw = _supervised_raw_scores(model, X)
                per_model[str(kind)] = score_from_stat_with_calib(
                    p_raw,
                    sup_calibs.get(kind),
                    kind=str(kind),
                )
            except Exception as e:
                logger.info("[retrain][ML-ensemble] %s scoring failed: %s", kind, e)

    return y, per_model


def evaluate_ml_ensemble(
    df_val: pd.DataFrame,
    feature_order: list[str],
    if_model=None,
    if_calib: dict | None = None,
    ocsvm_model=None,
    ocsvm_calib: dict | None = None,
    ae_bundle=None,
    ae_calib: dict | None = None,
    lstm_bundle=None,
    lstm_calib: dict | None = None,
    supervised_models: dict | None = None,
    supervised_calibs: dict | None = None,
    df_test: pd.DataFrame | None = None,
    out_prefix: str = "/models/ml/ml_ensemble",
) -> dict:
    y_val, per_model_val = _collect_ensemble_model_scores(
        df_eval=df_val,
        feature_order=feature_order,
        if_model=if_model,
        if_calib=if_calib,
        ocsvm_model=ocsvm_model,
        ocsvm_calib=ocsvm_calib,
        ae_bundle=ae_bundle,
        ae_calib=ae_calib,
        lstm_bundle=lstm_bundle,
        lstm_calib=lstm_calib,
        supervised_models=supervised_models,
        supervised_calibs=supervised_calibs,
    )
    if y_val is None:
        logger.info("[retrain][ML-ensemble] no 'label' col in calibration, skip ensemble eval")
        return {}
    if not per_model_val:
        logger.info("[retrain][ML-ensemble] no usable per-model scores, skip ensemble eval")
        return {}

    y_test: np.ndarray | None = None
    per_model_test: dict[str, np.ndarray] = {}
    if df_test is not None:
        y_test, per_model_test = _collect_ensemble_model_scores(
            df_eval=df_test,
            feature_order=feature_order,
            if_model=if_model,
            if_calib=if_calib,
            ocsvm_model=ocsvm_model,
            ocsvm_calib=ocsvm_calib,
            ae_bundle=ae_bundle,
            ae_calib=ae_calib,
            lstm_bundle=lstm_bundle,
            lstm_calib=lstm_calib,
            supervised_models=supervised_models,
            supervised_calibs=supervised_calibs,
        )
        if y_test is None:
            per_model_test = {}

    available_all = list(per_model_val.keys())
    available_supervised = [k for k in available_all if k.startswith("sup_")]
    available_novelty = [k for k in ("if", "ocsvm", "ae", "lstm_ae") if k in per_model_val]

    def _resolve_group_kinds(requested: list[str], fallback: list[str]) -> list[str]:
        out: list[str] = []
        for k in requested:
            kk = str(k).strip().lower()
            if kk in per_model_val and kk not in out:
                out.append(kk)
        if not out:
            for kk in fallback:
                if kk in per_model_val and kk not in out:
                    out.append(kk)
        return out

    primary_kinds = _resolve_group_kinds(ML_PRIMARY_CORE_KINDS, available_supervised)
    aux_kinds = _resolve_group_kinds(ML_AUX_NOVELTY_KINDS, available_novelty)
    aux_kinds = [k for k in aux_kinds if k not in primary_kinds]

    def _eval_group(
        group_name: str,
        selected_kinds: list[str],
        agg_method: str,
        group_out_prefix: str,
        title_prefix: str,
    ) -> tuple[dict, dict]:
        selected = {k: per_model_val[k] for k in selected_kinds if k in per_model_val}
        if not selected:
            return {}, {}

        score_val = _aggregate_prob_arrays(list(selected.values()), method=agg_method)
        metrics = _eval_and_plot_scores_with_labels(
            score_val,
            y_val,
            out_prefix=group_out_prefix,
            title_prefix=title_prefix,
        )

        temp_t, temp_metrics = fit_temperature_scaler(score_val, y_val, kind=f"ml_ensemble_{group_name}")
        if temp_metrics.get("used"):
            score_val = _apply_temperature_scaling(score_val, temp_t)
            metrics = _eval_and_plot_scores_with_labels(
                score_val,
                y_val,
                out_prefix=group_out_prefix,
                title_prefix=f"{title_prefix} (temp)",
            )

        calib_meta: dict = {}
        if temp_metrics.get("used"):
            calib_meta["temperature"] = float(temp_t)

        group_calib, prior_info = _attach_prior_correction_to_calibration(
            {},
            y_val,
            kind=f"ml_ensemble_{group_name}",
        )
        prior_cfg = (group_calib or {}).get("prior_correction")
        if isinstance(prior_cfg, dict) and bool(prior_cfg.get("enabled", False)):
            score_val = _apply_prior_correction_probs(score_val, prior_cfg)
            metrics = _eval_and_plot_scores_with_labels(
                score_val,
                y_val,
                out_prefix=group_out_prefix,
                title_prefix=f"{title_prefix} (temp, prior)",
            )
            try:
                calib_meta["prior_correction"] = {
                    "enabled": True,
                    "calibration_prior": float(prior_cfg.get("calibration_prior")),
                    "deployment_prior": float(prior_cfg.get("deployment_prior")),
                    "method": str(prior_cfg.get("method", "odds_correction")),
                }
            except Exception:
                pass

        out_scores_csv = f"{group_out_prefix}_per_model_scores.csv"
        df_out = pd.DataFrame({"ml_score": score_val, "label": y_val})
        for k, v in selected.items():
            df_out[f"score_{k}"] = v
        Path(out_scores_csv).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_scores_csv, index=False)
        logger.info("[retrain][ML-ensemble:%s] per-model scores saved -> %s", group_name, out_scores_csv)

        metrics = metrics or {}
        recommended_thr, thr_info = choose_operating_threshold(y_val, score_val)
        _, bestf1_ref = best_threshold_f1(y_val, score_val)
        metrics.update(metrics_on_threshold(y_val, score_val, recommended_thr))
        metrics["best_f1_on_val"] = float(bestf1_ref)
        metrics["recommended_ml_threshold"] = float(recommended_thr)
        metrics["threshold_selection"] = thr_info
        metrics["aggregation"] = str(agg_method)
        metrics["temperature_scaling"] = temp_metrics
        metrics["prior_correction"] = prior_info
        metrics["selected_models"] = list(selected.keys())

        if (
            RETRAIN_ENABLE_TEST_EVAL
            and y_test is not None
            and per_model_test
            and selected
        ):
            selected_test = {k: per_model_test[k] for k in selected.keys() if k in per_model_test}
            if selected_test:
                score_test = _aggregate_prob_arrays(list(selected_test.values()), method=agg_method)
                if temp_metrics.get("used"):
                    score_test = _apply_temperature_scaling(score_test, temp_t)
                if isinstance(prior_cfg, dict) and bool(prior_cfg.get("enabled", False)):
                    score_test = _apply_prior_correction_probs(score_test, prior_cfg)
                metrics["test"] = metrics_on_threshold(y_test, score_test, recommended_thr)
                metrics["test"]["threshold_used"] = float(recommended_thr)
                metrics["test_eval_enabled"] = True
                metrics["test_rows"] = int(len(y_test))
            else:
                metrics["test_eval_enabled"] = False

        return metrics, calib_meta

    primary_metrics, primary_calib_meta = _eval_group(
        group_name="primary",
        selected_kinds=primary_kinds,
        agg_method=(ML_PRIMARY_ENSEMBLE_AGG or ML_ENSEMBLE_AGG or "mean_logit").strip().lower(),
        group_out_prefix=out_prefix,
        title_prefix=f"ML-ensemble primary ({(ML_PRIMARY_ENSEMBLE_AGG or ML_ENSEMBLE_AGG or 'mean_logit').strip().lower()})",
    )
    aux_metrics, aux_calib_meta = _eval_group(
        group_name="aux_novelty",
        selected_kinds=aux_kinds,
        agg_method=(ML_AUX_NOVELTY_AGG or "mean_logit").strip().lower(),
        group_out_prefix="/models/ml/ml_aux_novelty",
        title_prefix=f"ML-aux-novelty ({(ML_AUX_NOVELTY_AGG or 'mean_logit').strip().lower()})",
    )

    out: dict = {
        "primary": primary_metrics,
        "aux_novelty": aux_metrics,
        "primary_models": primary_kinds,
        "aux_novelty_models": aux_kinds,
    }
    if primary_metrics:
        out.update(primary_metrics)
        out["recommended_ml_threshold"] = float(primary_metrics.get("recommended_ml_threshold", out.get("recommended_ml_threshold", 0.6)))
        out["aggregation"] = str(primary_metrics.get("aggregation", ML_PRIMARY_ENSEMBLE_AGG))
    elif aux_metrics:
        out["recommended_ml_threshold"] = float(aux_metrics.get("recommended_ml_threshold", 0.6))
        out["aggregation"] = str(aux_metrics.get("aggregation", ML_AUX_NOVELTY_AGG))

    if aux_metrics:
        out["recommended_aux_novelty_threshold"] = float(aux_metrics.get("recommended_ml_threshold", 0.8))

    if primary_calib_meta:
        out["calibration"] = primary_calib_meta
    if aux_calib_meta:
        out["aux_novelty_calibration"] = aux_calib_meta

    return out

def main():
    """
    Main retraining function that orchestrates the training of NLP and ML models.
    It checks configuration flags, prepares data, and calls individual training functions.
    1. Trains NLP models (TFIDF+LR and CharTFIDF+SGD) for general and per-attack classification.
    2. Trains ML models (IsolationForest, OneClassSVM, Autoencoder, LSTM Autoencoder).
    3. Trains a rules-supervised classifier if needed.
    4. Saves model metadata and feature order signature for inference.
    """
    logger.info(f"[retrain] starting... target={TRAIN_TARGET}")
    if not FEATURE_ORDER:
        raise SystemExit("FEATURE_ORDER is empty, cannot proceed with ML/NLP model training.")

    do_nlp = want_nlp()
    do_ml  = want_ml()

    need_rules_clf = do_ml


    silver_df = None
    if do_nlp or need_rules_clf:
        ensure_nlp_data(NLP_TRAIN_FILE)
        silver_df = build_silver_csv(NLP_TRAIN_FILE, RULES_FILE, SILVER_OUT_CSV)

    # NLP (TFIDF+LR + CharTFIDF+SGD)
    if do_nlp:
        train_tfidf_lr(str(SILVER_OUT_CSV), NLP_TFIDF_LR_OUT, label_col="label_any")
        train_char_sgd(str(SILVER_OUT_CSV), NLP_CHAR_SGD_OUT, label_col="label_any")

        attacks = list_attacks_from_rules(RULES_FILE)
        per_attack = {}
        for atk in attacks:
            col = f"label_{atk}"
            pos = int(silver_df[col].sum()) if (silver_df is not None and col in silver_df.columns) else 0
            if pos >= MIN_POS_PER_ATTACK:
                out_tfidf = f"/models/nlp/nlp_tfidf_lr__{atk}.joblib"
                out_char  = f"/models/nlp/nlp_char_sgd__{atk}.joblib"
                train_tfidf_lr(str(SILVER_OUT_CSV), out_tfidf, label_col=col)
                train_char_sgd(str(SILVER_OUT_CSV), out_char,  label_col=col)
                per_attack[atk] = {"tfidf_lr": out_tfidf, "char_sgd": out_char}
            else:
                logger.info(f"[retrain][skip] not enough positives for {atk} (pos={pos}, need={MIN_POS_PER_ATTACK})")

        meta_idx = {
            "tfidf_lr": NLP_TFIDF_LR_OUT,
            "char_sgd": NLP_CHAR_SGD_OUT,
            "per_attack": per_attack
        }
        Path(NLP_META_PATH).write_text(json.dumps(meta_idx, ensure_ascii=False, indent=2))
        logger.info(f"[retrain] wrote nlp meta -> {NLP_META_PATH}")

    # ML (IF/OCSVM/AE/LSTM AE + rules-supervised clf)
    if do_ml:
        ensure_ml_num_data(str(ML_NUM_FILE), FEATURE_ORDER)
        ensure_ml_validation_data(str(ML_VALIDATION_FILE), FEATURE_ORDER + ["label"])
        if RETRAIN_ENABLE_TEST_EVAL:
            ensure_ml_test_data(str(ML_TEST_FILE), FEATURE_ORDER + ["label"])

        ml_calib_path = str(ML_VALIDATION_FILE)
        ml_test_path = str(ML_TEST_FILE) if (RETRAIN_ENABLE_TEST_EVAL and ML_TEST_FILE.exists()) else None
        if RETRAIN_ENABLE_TEST_EVAL and ml_test_path is None and ML_AUTO_SPLIT_VAL_FOR_TEST:
            ml_calib_path, ml_test_path = build_calibration_and_test_from_validation(
                str(ML_VALIDATION_FILE),
                FEATURE_ORDER + ["label"],
                out_calib_path=ML_CALIB_SPLIT_FILE,
                out_test_path=ML_TEST_SPLIT_FILE,
                test_size=ML_VAL_TEST_SPLIT,
                random_state=ML_VAL_TEST_RANDOM_STATE,
            )
        logger.info(
            "[retrain] ML data roles: train(benign)=%s calibration(mixed)=%s test(mixed)=%s auto_split=%s",
            ML_NUM_FILE,
            ml_calib_path,
            ml_test_path,
            ML_AUTO_SPLIT_VAL_FOR_TEST,
        )
        if not RETRAIN_ENABLE_TEST_EVAL:
            logger.info("[retrain] RETRAIN_ENABLE_TEST_EVAL=false -> skipping test metrics during retrain")

        sup_train_path = None
        sup_calib_path = str(ml_calib_path)
        sup_split_info: dict = {
            "enabled": bool(ML_ENABLE_SUPERVISED_BASELINES),
            "mode": "disabled",
        }
        if ML_ENABLE_SUPERVISED_BASELINES:
            try:
                sup_train_path, sup_calib_path, sup_split_info = prepare_supervised_train_and_calibration_paths(
                    str(ml_calib_path),
                    FEATURE_ORDER,
                )
            except Exception as e:
                logger.info(
                    "[retrain][supervised] prepare split failed -> disable supervised baselines for this run: %s",
                    e,
                )
                sup_train_path = None
                sup_calib_path = str(ml_calib_path)
                sup_split_info = {
                    "enabled": False,
                    "mode": "error",
                    "error": str(e),
                }

        if_model, if_calib, if_thr, if_metrics = train_isolation_forest(
            str(ML_NUM_FILE),
            ml_calib_path,
            ml_test_path,
            IF_OUT,
            FEATURE_ORDER,
        )

        ocsvm_model, ocsvm_calib, ocsvm_thr, ocsvm_metrics = train_ocsvm(
            str(ML_NUM_FILE),
            ml_calib_path,
            ml_test_path,
            OCSVM_OUT,
            FEATURE_ORDER,
        )

        ae_bundle = ae_calib = ae_thr = ae_metrics = None
        try:
            ae_bundle, ae_calib, ae_thr, ae_metrics = train_autoencoder(
                str(ML_NUM_FILE),
                ml_calib_path,
                ml_test_path,
                AE_OUT,
                FEATURE_ORDER,
            )
        except Exception as e:
            logger.info(f"[retrain][AE] training failed: {e}")

        lstm_bundle = lstm_calib = lstm_thr = lstm_metrics = None
        try:
            lstm_bundle, lstm_calib, lstm_thr, lstm_metrics = train_lstm_autoencoder(
                str(ML_NUM_FILE),
                ml_calib_path,
                ml_test_path,
                LSTM_AE_OUT,
                FEATURE_ORDER,
            )
        except Exception as e:
            logger.info(f"[retrain][LSTM-AE] training failed: {e}")

        supervised_models: dict[str, object] = {}
        supervised_calibs: dict[str, dict] = {}
        supervised_metrics: dict[str, dict] = {}
        if ML_ENABLE_SUPERVISED_BASELINES and sup_train_path:
            for kind, model_name, estimator, out_path in supervised_baseline_specs():
                try:
                    sup_model, sup_calib, _, sup_metrics = train_supervised_numeric_baseline(
                        model_kind=kind,
                        model_name=model_name,
                        estimator=estimator,
                        train_csv_path=str(sup_train_path),
                        calib_csv_path=str(sup_calib_path),
                        test_csv_path=ml_test_path,
                        out_path=out_path,
                        feature_order=FEATURE_ORDER,
                    )
                    supervised_models[kind] = sup_model
                    supervised_calibs[kind] = dict(sup_calib or {})
                    supervised_metrics[kind] = sup_metrics or {}
                except Exception as e:
                    logger.info("[retrain][%s] supervised baseline failed: %s", kind, e)
        elif ML_ENABLE_SUPERVISED_BASELINES:
            logger.info("[retrain][supervised] no supervised train path available, skip baselines")

        try:
            ensemble_eval_path = str(sup_calib_path or ml_calib_path)
            df_val_ml = pd.read_csv(ensemble_eval_path, low_memory=False)
            y_val_ml = _labels_from_df(df_val_ml)
            df_val_ml, _ = _maybe_downsample_df(
                df_val_ml,
                y_val_ml,
                ML_VAL_MAX_ROWS,
                tag="ML-ensemble/calibration",
            )
            df_test_ml = None
            if RETRAIN_ENABLE_TEST_EVAL and ml_test_path and Path(ml_test_path).exists():
                try:
                    df_test_ml = pd.read_csv(ml_test_path, low_memory=False)
                    y_test_ml = _labels_from_df(df_test_ml)
                    df_test_ml, _ = _maybe_downsample_df(
                        df_test_ml,
                        y_test_ml,
                        ML_TEST_MAX_ROWS,
                        tag="ML-ensemble/test",
                    )
                except Exception as e:
                    logger.info("[retrain][ML-ensemble] failed to load test set: %s", e)
                    df_test_ml = None
            ensemble_metrics = evaluate_ml_ensemble(
                df_val_ml,
                FEATURE_ORDER,
                if_model=if_model,
                if_calib=if_calib,
                ocsvm_model=ocsvm_model,
                ocsvm_calib=ocsvm_calib,
                ae_bundle=ae_bundle,
                ae_calib=ae_calib,
                lstm_bundle=lstm_bundle,
                lstm_calib=lstm_calib,
                supervised_models=supervised_models,
                supervised_calibs=supervised_calibs,
                df_test=df_test_ml,
                out_prefix="/models/ml/ml_ensemble",
            )
            if ensemble_metrics:
                ml_meta_path = Path("/models/ml/ml_ensemble.meta.json")
                ml_meta = {
                    "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "feature_order": FEATURE_ORDER,
                    "models": {
                        "if": if_model is not None,
                        "ocsvm": ocsvm_model is not None,
                        "ae": ae_bundle is not None,
                        "lstm_ae": lstm_bundle is not None,
                        "sup_lr": supervised_models.get("sup_lr") is not None,
                        "sup_hgb": supervised_models.get("sup_hgb") is not None,
                        "sup_rf": supervised_models.get("sup_rf") is not None,
                        "sup_et": supervised_models.get("sup_et") is not None,
                    },
                    "supervised_baselines": {
                        "enabled": bool(ML_ENABLE_SUPERVISED_BASELINES),
                        "split": sup_split_info,
                        "metrics": supervised_metrics,
                    },
                    "ensemble_roles": {
                        "primary_core_kinds": list(ML_PRIMARY_CORE_KINDS),
                        "aux_novelty_kinds": list(ML_AUX_NOVELTY_KINDS),
                        "primary_core_used": list((ensemble_metrics or {}).get("primary_models") or []),
                        "aux_novelty_used": list((ensemble_metrics or {}).get("aux_novelty_models") or []),
                    },
                    "recommended_threshold": float(
                        ensemble_metrics.get("recommended_ml_threshold", 0.7)
                    ),
                    "metrics": ensemble_metrics,
                }
                if isinstance((ensemble_metrics or {}).get("recommended_aux_novelty_threshold"), (int, float)):
                    ml_meta["recommended_aux_novelty_threshold"] = float(
                        ensemble_metrics.get("recommended_aux_novelty_threshold")
                    )
                primary_calib = (ensemble_metrics or {}).get("calibration")
                if isinstance(primary_calib, dict) and primary_calib:
                    ml_meta["calibration"] = primary_calib
                aux_calib = (ensemble_metrics or {}).get("aux_novelty_calibration")
                if isinstance(aux_calib, dict) and aux_calib:
                    ml_meta["aux_novelty_calibration"] = aux_calib
                ml_meta_path.parent.mkdir(parents=True, exist_ok=True)
                ml_meta_path.write_text(json.dumps(ml_meta, ensure_ascii=False, indent=2))
                logger.info(
                    "[retrain][ML-ensemble] wrote ensemble meta -> %s (recommended ML_THRESHOLD ≈ %.3f)",
                    ml_meta_path,
                    ml_meta["recommended_threshold"],
                )
        except Exception as e:
            logger.info(f"[retrain][ML-ensemble] ensemble evaluation failed: {e}")

        # rules-supervised classifier - silver_df
        if need_rules_clf and silver_df is not None:
            _, ml_rules_feats = build_ml_rules_dataset_from_silver(
                silver_df,
                RULES_FILE,
                ML_RULES_CSV,
            )
            train_ml_rules_classifier(
                str(ML_RULES_CSV),
                ML_RULES_CLF_OUT,
                ML_RULES_META_PATH,
                ml_rules_feats,
                label_col="label_any",
            )
            
    # feature_order signature for inference
    sig = hashlib.sha1(",".join(FEATURE_ORDER).encode()).hexdigest()[:8]
    Path("/models/feature_order.sig").write_text(sig)
    logger.info(f"[retrain] feature_order_sig={sig}")

    try:
        build_figures_manifest(
            models_dir="/models/ml",
            out_csv="/models/ml/figures_manifest.csv",
        )
    except Exception as e:
        logger.info("[retrain] figures manifest generation failed: %s", e)

    logger.info("[retrain] done.")


if __name__ == "__main__": 
    main()
