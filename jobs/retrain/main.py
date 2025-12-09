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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
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
    "query_param_count,path_depth,suspicious_char_ratio,is_json_payload,http_method_code"
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

def write_if_meta(model_path: Path, feature_order: List[str], model_obj, calibration: dict | None = None, metrics: dict | None = None):
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
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"[retrain] wrote IF meta -> {meta_path}")

def evaluate_if_with_labels(
    if_model,
    ml_df: pd.DataFrame,
    feature_order: list[str],
    out_prefix: str = "/models/ml/if_csic_eval"
):
    """
    Evaluates IsolationForest model using labeled ML numeric data (CSIC).
    
    Args:
        if_model: Trained IsolationForest model.
        ml_df (pd.DataFrame): ML numeric DataFrame with 'label' column.
        feature_order (list[str]): Order of features used in training.
        out_prefix (str): Prefix for output files (CSV, PNG).
    Returns:
        None
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

    df_raw = if_model.decision_function(X)
    df_raw = df_raw.reshape(-1)

    mu = df_raw.mean()
    sigma = df_raw.std() if df_raw.std() > 1e-9 else 1.0

    z = -(df_raw - mu) / sigma
    
    scores = 1.0 / (1.0 + np.exp(-z))

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

    logger.info(f"[retrain] IF eval: n={len(y)}, pos={int(y.sum())}, neg={int((y==0).sum())}")
    logger.info(f"[retrain] IF eval: ROC AUC={roc:.4f} AP(PR AUC)={ap:.4f}" if roc is not None and ap is not None else
        f"[retrain] IF eval: ROC={roc} AP={ap}")

    # Statistics per class
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

    # Store scores in CSV
    out_scores_csv = f"{out_prefix}_scores.csv"
    eval_df = pd.DataFrame({"score": scores, "label": y})
    eval_df.to_csv(out_scores_csv, index=False)
    logger.info(f"[retrain] IF eval: scores saved -> {out_scores_csv}")
    # Build ROC/PR curves and histograms in PNG
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

        # Histograms for label=0 vs 1
        plt.figure()
        plt.hist(scores[y == 0], bins=50, alpha=0.7, label="label=0 (normal)")
        if (y == 1).any():
            plt.hist(scores[y == 1], bins=50, alpha=0.7, label="label=1 (attack)")
        plt.xlabel("IF calibrated score (anomaly)")
        plt.ylabel("Count")
        plt.title("IsolationForest score distribution (CSIC)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_hist.png")
        plt.close()

        logger.info(f"[retrain] IF eval: plots saved with prefix {out_prefix}_*.png")
    except Exception as e:
        logger.info(f"[retrain] IF eval: plotting failed: {e}")

def write_ocsvm_meta(outp, feature_order, model, calibration: dict | None = None, metrics: dict | None = None):
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
    meta = {
        "model": "OneClassSVM",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_order": feature_order,
        "n_features_in_": int(getattr(model, "n_features_in_", len(feature_order))),
        "params": {
            "kernel": getattr(model, "kernel", None),
            "nu": float(getattr(model, "nu", 0.0)),
            "gamma": getattr(model, "gamma", None),
        },
        "calibration": calibration or {},
        "metrics": metrics or {},
    }
    meta_path = outp.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"[retrain] wrote OCSVM meta -> {meta_path}")

def write_ae_meta(outp: Path, feature_order: list[str], input_dim: int, model, calibration: dict | None = None, metrics: dict | None = None):
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
    meta_path = outp.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"[retrain] wrote AE meta -> {meta_path}")


def write_lstm_ae_meta(outp: Path, feature_order: list[str], timesteps: int, model, calibration: dict | None = None, metrics: dict | None = None):
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
    meta_path = outp.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"[retrain] wrote LSTM AE meta -> {meta_path}")

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
        df = pd.read_csv(p, nrows=5)
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
    sig = {}
    for atk, grp in compiled_rules.items():
        strong_hit = 0.0
        for rx in grp["strong_res"]:
            try:
                if rx.search(blob):
                    strong_hit = 1.0
                    break
            except Exception:
                continue
        hits = 0
        for tok in grp["partial_tokens"]:
            if tok in blob:
                hits += 1
        # normalization
        base = hits / max(1, grp["divisor"])
        sig[f"{atk}_strong"] = float(1.0 if strong_hit >= 1.0 else 0.0)
        sig[f"{atk}_partial"] = float(min(1.0, base))
    return sig

def _label_from_signals(sig: dict, min_partial: float) -> dict:
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
        if k.endswith("_strong"):
            atk = k[:-7]
            strong = sig.get(f"{atk}_strong", 0.0)
            partial = sig.get(f"{atk}_partial", 0.0)
            y = 1 if (strong >= 1.0 or partial >= min_partial) else 0
            labels[f"label_{atk}"] = y
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
    df = pd.read_csv(csv_in)
    if "text" not in df.columns:
        raise ValueError("NLP CSV must contain 'text' column for silver labeling")
    rules_json = json.loads(rules_path.read_text(encoding="utf-8"))
    compiled = _compile_rules(rules_json)

    rows = []
    for _, r in df.iterrows():
        t = _norm_text(str(r["text"]))
        sig = _scan_blob(t, compiled)
        labs = _label_from_signals(sig, SILVER_MIN_PARTIAL)
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
    """
    Trains a TF-IDF + Logistic Regression model on the given CSV data.
    
    Args:
        csv_path (str): Path to the input CSV file containing text and labels.
        out_path (str): Path to save the trained model pipeline.
        label_col (str, optional): Name of the label column. Defaults to "label".
    """
    df = pd.read_csv(csv_path)
    if not {"text", label_col}.issubset(df.columns):
        raise ValueError(f"NLP CSV must contain columns: text,{label_col}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df[label_col].astype(int),
        test_size=0.2, random_state=42, stratify=df[label_col].astype(int)
    )

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

    pipe.fit(X_train, y_train)

    prob_te = pipe.predict_proba(X_test)[:, 1]
    pred = (prob_te >= 0.5).astype(int)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, prob_te) if len(set(y_test)) > 1 else None

    # Evaluate on all data for recommended threshold
    y_all = df[label_col].astype(int).to_numpy()
    proba_all = pipe.predict_proba(df["text"])[:, 1]
    prec, rec, thr = precision_recall_curve(y_all, proba_all)
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    thr_for_f1 = np.append(thr, [thr[-1] if len(thr) else 0.6])
    best_idx = int(np.argmax(f1s)) if len(f1s) else 0
    recommended_threshold = float(thr_for_f1[best_idx]) if len(thr_for_f1) else 0.6
    auc_all = float(roc_auc_score(y_all, proba_all)) if len(set(y_all)) > 1 else None

    # Save the trained pipeline
    joblib.dump({"pipeline": pipe}, out_path)

    meta = {
        "model": "TFIDF+LogReg (pipeline)",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": {
            "roc_auc_all": auc_all,
            "best_f1_all": float(f1s[best_idx]) if len(f1s) else None,
            "val_f1": float(f1),
            "val_auc": float(auc) if auc is not None else None,
        },
        "recommended_threshold": recommended_threshold,
        "data_rows": int(df.shape[0]),
        "label_col": label_col,
        "vectorizer": {
            "ngram_range": (1, 2),
            "max_features": 50000,
            "min_df": 1,
            "max_df": 0.98,
        }
    }
    Path(out_path).with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

def train_char_sgd(csv_path: str, out_path: str, label_col: str = "label"):
    """
    Trains a character-level TF-IDF + SGD Classifier model on the given CSV data.
    
    Args:
        csv_path (str): Path to the input CSV file containing text and labels.
        out_path (str): Path to save the trained model pipeline.
        label_col (str, optional): Name of the label column. Defaults to "label".
    """
    df = pd.read_csv(csv_path)
    if not {"text", label_col}.issubset(df.columns):
        raise ValueError(f"NLP CSV must contain columns: text,{label_col}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df[label_col].astype(int),
        test_size=0.2, random_state=42, stratify=df[label_col].astype(int)
    )

    # Simple oversampling of positive class
    pos_mask = y_train.values == 1
    n_pos = int(pos_mask.sum()); n_neg = int((~pos_mask).sum())
    if 0 < n_pos < n_neg:
        target_pos = int(0.6 * n_neg)
        reps = max(1, math.ceil((target_pos - n_pos) / max(1, n_pos)))
        reps = min(reps, 8)
        if reps > 1:
            X_pos = X_train[pos_mask]; y_pos = y_train[pos_mask]
            X_train = pd.concat([X_train] + [X_pos] * (reps - 1), ignore_index=True)
            y_train = pd.concat([y_train] + [y_pos] * (reps - 1), ignore_index=True)

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

    pipe = Pipeline([
        ("tfidf_char", vec),
        ("clf", clf),
    ])

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    f1 = f1_score(y_test, pred)

    y_all = df[label_col].astype(int).to_numpy()
    proba_all = pipe.predict_proba(df["text"])[:, 1]
    prec, rec, thr = precision_recall_curve(y_all, proba_all)
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    thr_for_f1 = np.append(thr, [thr[-1] if len(thr) else 0.6])
    best_idx = int(np.argmax(f1s)) if len(f1s) else 0
    recommended_threshold = float(thr_for_f1[best_idx]) if len(thr_for_f1) else 0.6
    auc_all = float(roc_auc_score(y_all, proba_all)) if len(set(y_all)) > 1 else None

    joblib.dump({"pipeline": pipe}, out_path)

    meta = {
        "model": "CharTFIDF+SGD(log_loss)",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": {
            "roc_auc_all": auc_all,
            "best_f1_all": float(f1s[best_idx]) if len(f1s) else None,
            "val_f1": float(f1),
        },
        "recommended_threshold": recommended_threshold,
        "data_rows": int(df.shape[0]),
        "label_col": label_col,
        "vectorizer": {
            "analyzer": "char",
            "ngram_range": (2, 6),
            "max_features": 200_000,
            "min_df": 1,
            "max_df": 1.0,
        }
    }
    Path(out_path).with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

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
    df = pd.read_csv(csv_path)
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
    f1 = f1_score(y_te, (proba_te >= 0.5).astype(int))
    auc = roc_auc_score(y_te, proba_te) if len(set(y_te)) > 1 else None

    proba_all = clf.predict_proba(X)[:,1]
    prec, rec, thr = precision_recall_curve(y, proba_all)
    f1s = np.where((prec + rec) > 0, 2*prec*rec/(prec+rec), 0.0)
    thr_for_f1 = np.append(thr, [thr[-1] if len(thr)>0 else 0.5])
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

# ML-model training functions
def train_isolation_forest(csv_path: str, out_path: str, feature_order: List[str]):
    """
    Trains an IsolationForest model on the given CSV data.
    
    Args:
        csv_path (str): Path to the input CSV file containing features.
        out_path (str): Path to save the trained IsolationForest model.
        feature_order (List[str]): List of feature column names in the correct order.
    """
    df = pd.read_csv(csv_path)
    X = ensure_columns(df, feature_order).fillna(0.0)

    model = IsolationForest(
        n_estimators=int(os.getenv("IF_N_ESTIMATORS", "200")),
        contamination=os.getenv("IF_CONTAMINATION", "auto"),
        random_state=int(os.getenv("IF_RANDOM_STATE", "42")),
        n_jobs=-1,
    )
    model.fit(X.values)

    scores = None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X.values)

    # calibration
    if_mu = float(np.mean(scores)) if scores is not None else 0.0
    if_sigma = float(np.std(scores)) if (scores is not None and np.std(scores) > 1e-9) else 0.1

    # metrics on CSIC label (if present)
    metrics = None
    if "label" in df.columns and scores is not None:
        try:
            y_raw = df["label"]
            # convert to binary 0/1
            y = (pd.to_numeric(y_raw, errors="coerce").fillna(0.0) > 0.0).astype(int).to_numpy()

            if len(np.unique(y)) >= 2:
                # IsolationForest decision_function
                anom_score = -scores

                roc = roc_auc_score(y, anom_score)
                ap = average_precision_score(y, anom_score)

                prec, rec, thr = precision_recall_curve(y, anom_score)
                f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
                best_f1 = float(f1s.max()) if len(f1s) else None

                metrics = {
                    "csic_label_present": True,
                    "roc_auc_label": float(roc),
                    "pr_auc_label": float(ap),
                    "best_f1_label": best_f1,
                    "label_positives": int(y.sum()),
                    "label_rows": int(len(y)),
                }
                logger.info(f"[retrain][IF] CSIC label metrics: ROC_AUC={roc:.4f} PR_AUC={ap:.4f} best_F1={best_f1}")
            else:
                logger.info("[retrain][IF] label column present but only one class — skip ROC/PR metrics")
        except Exception as e:
            logger.info(f"[retrain][IF] failed to compute label-based metrics: {e}")
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, outp)
    logger.info(f"[retrain] saved IsolationForest -> {outp} (n_features_in_={getattr(model,'n_features_in_',None)})")

    write_if_meta(
        outp,
        feature_order,
        model,
        calibration={"mu": if_mu, "sigma": if_sigma},
        metrics=metrics,
    )
    
def compute_anomaly_calibration(model, X, kind: str = "") -> dict:
    """
    Computes calibration parameters (mu, sigma) for anomaly scores
    from the model's decision_function on data X.
    
    Args:
        model: Trained anomaly detection model with decision_function method.
        X (np.ndarray): Input data for calibration.
        kind (str, optional): Kind of model (for logging). Defaults to "".
    Returns:
        dict: Calibration parameters {"mu": ..., "sigma": ...} or empty if failed.
    """
    try:
        df_vals = model.decision_function(X)
    except Exception as e:
        logger.info(f"[retrain] anomaly_calib: decision_function failed for {kind or type(model).__name__}: {e}")
        return {}

    df_vals = np.asarray(df_vals, dtype=float).reshape(-1)
    if df_vals.size == 0:
        logger.info(f"[retrain] anomaly_calib: empty decision_function for {kind or type(model).__name__}")
        return {}

    mu = float(df_vals.mean())
    sigma = float(df_vals.std())
    if not np.isfinite(sigma) or sigma <= 1e-9:
        sigma = 1.0

    logger.info(f"[retrain] anomaly_calib[{kind or type(model).__name__}]: mu={mu:.6f} sigma={sigma:.6f}")
    return {"mu": mu, "sigma": sigma}
    
def evaluate_if_with_labels(
    if_model,
    ml_df: pd.DataFrame,
    feature_order: list[str],
    out_prefix: str = "/models/ml/if_csic_eval"
):
    """
    Evaluates IsolationForest model on labeled data and computes metrics.
    
    Args:
        if_model: Trained IsolationForest model.
        ml_df (pd.DataFrame): DataFrame with features and 'label' column.
        feature_order (list[str]): List of feature column names in the correct order.
        out_prefix (str, optional): Prefix for output files. Defaults to "/models/ml/if_csic_eval".
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

    df_raw = if_model.decision_function(X)
    df_raw = df_raw.reshape(-1)

    mu = df_raw.mean()
    sigma = df_raw.std() if df_raw.std() > 1e-9 else 1.0

    z = -(df_raw - mu) / sigma

    scores = 1.0 / (1.0 + np.exp(-z))

    # metrics
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

    logger.info(f"[retrain] IF eval: n={len(y)}, pos={int(y.sum())}, neg={int((y==0).sum())}")
    logger.info(f"[retrain] IF eval: ROC AUC={roc:.4f} AP(PR AUC)={ap:.4f}" if roc is not None and ap is not None else
        f"[retrain] IF eval: ROC={roc} AP={ap}")

    # Simple per-class stats
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

    # save scores to CSV
    out_scores_csv = f"{out_prefix}_scores.csv"
    eval_df = pd.DataFrame({"score": scores, "label": y})
    eval_df.to_csv(out_scores_csv, index=False)
    logger.info(f"[retrain] IF eval: scores saved -> {out_scores_csv}")

    # build plots
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

        # histograms for label=0 vs 1
        plt.figure()
        plt.hist(scores[y == 0], bins=50, alpha=0.7, label="label=0 (normal)")
        if (y == 1).any():
            plt.hist(scores[y == 1], bins=50, alpha=0.7, label="label=1 (attack)")
        plt.xlabel("IF calibrated score (anomaly)")
        plt.ylabel("Count")
        plt.title("IsolationForest score distribution (CSIC)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_hist.png")
        plt.close()

        logger.info(f"[retrain] IF eval: plots saved with prefix {out_prefix}_*.png")
    except Exception as e:
        logger.info(f"[retrain] IF eval: plotting failed: {e}")

def train_ocsvm(csv_path: str, out_path: str, feature_order: List[str]):
    """
    Trains a OneClassSVM model on the given CSV data.
    
    Args:
        csv_path (str): Path to the input CSV file containing features.
        out_path (str): Path to save the trained OneClassSVM model.
        feature_order (List[str]): List of feature column names in the correct order.
    """
    df = pd.read_csv(csv_path)

    max_rows = int(os.getenv("OCSVM_MAX_TRAIN_ROWS", "150000"))
    if df.shape[0] > max_rows:
        logger.info(f"[retrain][OCSVM] dataset has {df.shape[0]} rows, sampling {max_rows} for training/metrics")
        df = df.sample(n=max_rows, random_state=42)

    X = ensure_columns(df, feature_order).fillna(0.0)

    nu = float(os.getenv("OCSVM_NU", "0.05"))
    gamma = os.getenv("OCSVM_GAMMA", "scale")

    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(X.values)

    calib = compute_anomaly_calibration(model, X.values, kind="ocsvm")

    metrics = None
    if "label" in df.columns:
        try:
            y_raw = df["label"]
            # як і для IF: все >0 вважаємо "атака"
            y = (pd.to_numeric(y_raw, errors="coerce").fillna(0.0) > 0.0).astype(int).to_numpy()

            if hasattr(model, "decision_function") and len(np.unique(y)) >= 2:
                df_vals = model.decision_function(X.values).reshape(-1)

                # більші df_vals = "нормальні", робимо score "атаки"
                anom_score = -df_vals

                roc = roc_auc_score(y, anom_score)
                ap = average_precision_score(y, anom_score)

                prec, rec, thr = precision_recall_curve(y, anom_score)
                f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
                best_f1 = float(f1s.max()) if len(f1s) else None

                metrics = {
                    "csic_label_present": True,
                    "roc_auc_label": float(roc),
                    "pr_auc_label": float(ap),
                    "best_f1_label": best_f1,
                    "label_positives": int(y.sum()),
                    "label_rows": int(len(y)),
                }
                logger.info(f"[retrain][OCSVM] CSIC label metrics: ROC_AUC={roc:.4f} PR_AUC={ap:.4f} best_F1={best_f1}")
            else:
                logger.info("[retrain][OCSVM] label column present but only one class or no decision_function — skip ROC/PR metrics")
        except Exception as e:
            logger.info(f"[retrain][OCSVM] failed to compute label-based metrics: {e}")
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, outp)
    logger.info(f"[retrain] saved OCSVM -> {outp} (n_features_in_={X.shape[1]})")

    write_ocsvm_meta(
        outp,
        feature_order,
        model,
        calibration=calib,
        metrics=metrics,
    )

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

def train_autoencoder(csv_path: str, out_path: str, feature_order: list[str]):
    """
    Dense autoencoder training on the given CSV data.
    
    Args:
        csv_path (str): Path to the input CSV file containing features.
        out_path (str): Path to save the trained autoencoder model.
        feature_order (list[str]): List of feature column names in the correct order.
    """

    df = pd.read_csv(csv_path)
    X_full = ensure_columns(df, feature_order).fillna(0.0).to_numpy(dtype=float)
    y = _labels_from_df(df)

    # make training set from normal rows (label=0) if present
    if y is not None and (y == 0).any():
        X_train = X_full[y == 0]
        logger.info(f"[retrain][AE] using {X_train.shape[0]} normal rows (label=0) for training")
    else:
        X_train = X_full
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
    ae_model.fit(
        X_train_sc,
        X_train_sc,
        epochs=ae_epochs,
        batch_size=ae_batch,
        shuffle=True,
        validation_split=0.1,
        verbose=1,
    )

    # calibration set
    max_calib = int(os.getenv("AE_MAX_CALIB_ROWS", "300000"))
    X_calib = X_full
    y_calib = y

    if X_calib.shape[0] > max_calib:
        rng = np.random.RandomState(43)
        idx = rng.choice(X_calib.shape[0], size=max_calib, replace=False)
        X_calib = X_calib[idx]
        if y_calib is not None:
            y_calib = y_calib[idx]
        logger.info(f"[retrain][AE] sampling {max_calib} rows for calibration/metrics (from {X_full.shape[0]})")

    X_calib_sc = scaler.transform(X_calib)
    X_rec = ae_model.predict(X_calib_sc, batch_size=ae_batch, verbose=0)
    recon_err = np.mean((X_calib_sc - X_rec) ** 2, axis=1)

    # calibration
    calib_mu = float(np.mean(recon_err))
    calib_sigma = float(np.std(recon_err) or 1e-6)
    calib = {"mu": calib_mu, "sigma": calib_sigma}

    # metrics on CSIC label
    metrics = None
    if y_calib is not None and len(np.unique(y_calib)) >= 2:
        try:
            anom_score = recon_err
            roc = roc_auc_score(y_calib, anom_score)
            ap = average_precision_score(y_calib, anom_score)
            prec, rec, _ = precision_recall_curve(y_calib, anom_score)
            f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
            best_f1 = float(f1s.max()) if len(f1s) else None
            metrics = {
                "roc_auc_label": float(roc),
                "pr_auc_label": float(ap),
                "best_f1_label": best_f1,
                "label_positives": int(y_calib.sum()),
                "label_rows": int(len(y_calib)),
            }
            logger.info(f"[retrain][AE] CSIC label metrics: ROC_AUC={roc:.4f} PR_AUC={ap:.4f} best_F1={best_f1}")
        except Exception as e:
            logger.info(f"[retrain][AE] failed to compute label-based metrics: {e}")

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

    write_ae_meta(outp, feature_order, input_dim=input_dim, model=ae_model, calibration=calib, metrics=metrics)

def train_lstm_autoencoder(csv_path: str, out_path: str, feature_order: list[str]):
    """
    LSTM autoencoder training on the given CSV data.
    
    Args:
        csv_path (str): Path to the input CSV file containing features.
        out_path (str): Path to save the trained LSTM autoencoder model.
        feature_order (list[str]): List of feature column names in the correct order.
    """
    df = pd.read_csv(csv_path)
    X_full = ensure_columns(df, feature_order).fillna(0.0).to_numpy(dtype=float)
    y = _labels_from_df(df)

    if y is not None and (y == 0).any():
        X_train = X_full[y == 0]
        logger.info(f"[retrain][LSTM-AE] using {X_train.shape[0]} normal rows (label=0) for training")
    else:
        X_train = X_full
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

    # reshape for LSTM
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
    lstm_ae.fit(
        X_train_seq,
        X_train_seq,
        epochs=lstm_epochs,
        batch_size=lstm_batch,
        shuffle=True,
        validation_split=0.1,
        verbose=1,
    )

    # calibration set
    max_calib = int(os.getenv("LSTM_AE_MAX_CALIB_ROWS", "200000"))
    X_calib = X_full
    y_calib = y

    if X_calib.shape[0] > max_calib:
        rng = np.random.RandomState(45)
        idx = rng.choice(X_calib.shape[0], size=max_calib, replace=False)
        X_calib = X_calib[idx]
        if y_calib is not None:
            y_calib = y_calib[idx]
        logger.info(f"[retrain][LSTM-AE] sampling {max_calib} rows for calibration/metrics (from {X_full.shape[0]})")

    X_calib_sc  = scaler.transform(X_calib)
    X_calib_seq = X_calib_sc.reshape((-1, timesteps, 1))

    X_rec = lstm_ae.predict(X_calib_seq, batch_size=lstm_batch, verbose=0)
    recon_err = np.mean((X_calib_seq - X_rec) ** 2, axis=(1, 2))

    calib_mu = float(np.mean(recon_err))
    calib_sigma = float(np.std(recon_err) or 1e-6)
    calib = {"mu": calib_mu, "sigma": calib_sigma}

    metrics = None
    if y_calib is not None and len(np.unique(y_calib)) >= 2:
        try:
            anom_score = recon_err
            roc = roc_auc_score(y_calib, anom_score)
            ap = average_precision_score(y_calib, anom_score)
            prec, rec, _ = precision_recall_curve(y_calib, anom_score)
            f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
            best_f1 = float(f1s.max()) if len(f1s) else None
            metrics = {
                "roc_auc_label": float(roc),
                "pr_auc_label": float(ap),
                "best_f1_label": best_f1,
                "label_positives": int(y_calib.sum()),
                "label_rows": int(len(y_calib)),
            }
            logger.info(f"[retrain][LSTM-AE] CSIC label metrics: ROC_AUC={roc:.4f} PR_AUC={ap:.4f} best_F1={best_f1}")
        except Exception as e:
            logger.info(f"[retrain][LSTM-AE] failed to compute label-based metrics: {e}")

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

    write_lstm_ae_meta(outp, feature_order, timesteps=timesteps, model=lstm_ae, calibration=calib, metrics=metrics)

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
        # IF + OCSVM
        ensure_ml_num_data(str(ML_NUM_FILE), FEATURE_ORDER)
        train_isolation_forest(str(ML_NUM_FILE), IF_OUT, FEATURE_ORDER)
        train_ocsvm(str(ML_NUM_FILE), OCSVM_OUT, FEATURE_ORDER)

        # AE + LSTM AE
        try:
            train_autoencoder(str(ML_NUM_FILE), AE_OUT, FEATURE_ORDER)
        except Exception as e:
            logger.info(f"[retrain][AE] training failed: {e}")

        try:
            train_lstm_autoencoder(str(ML_NUM_FILE), LSTM_AE_OUT, FEATURE_ORDER)
        except Exception as e:
            logger.info(f"[retrain][LSTM-AE] training failed: {e}")

        # rules-supervised classifier
        if need_rules_clf and silver_df is not None:
            _, ml_rules_feats = build_ml_rules_dataset_from_silver(
                silver_df,
                RULES_FILE,
                ML_RULES_CSV
            )
            train_ml_rules_classifier(
                str(ML_RULES_CSV),
                ML_RULES_CLF_OUT,
                ML_RULES_META_PATH,
                ml_rules_feats,
                label_col="label_any"
            )

    # feature_order signature for inference
    sig = hashlib.sha1(",".join(FEATURE_ORDER).encode()).hexdigest()[:8]
    Path("/models/feature_order.sig").write_text(sig)
    logger.info(f"[retrain] feature_order_sig={sig}")
    logger.info("[retrain] done.")
if __name__ == "__main__":
    main()