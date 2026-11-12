#!/usr/bin/env python3
"""
Train the general and the per-attack NLP models.

Writes models/nlp.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.text_norm import norm_text  # noqa: E402

SEED = 42
MIN_POS_PER_ATTACK = 3


def retrain_module():
    spec = importlib.util.spec_from_file_location("retrain_main", ROOT / "jobs/retrain/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_tfidf_lr() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=norm_text, ngram_range=(1, 2), min_df=1, max_df=0.98,
                                  max_features=50000, strip_accents="unicode", lowercase=False,
                                  token_pattern=r"(?u)\b[\w@./-]*'?[\w@./-]*\b", sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", C=4.0)),
    ])


def make_char_sgd() -> Pipeline:
    return Pipeline([
        ("tfidf_char", TfidfVectorizer(preprocessor=norm_text, analyzer="char", ngram_range=(2, 6),
                                       min_df=1, max_df=1.0, max_features=200_000, lowercase=False)),
        ("clf", SGDClassifier(loss="log_loss", alpha=5e-6, max_iter=5000, random_state=SEED,
                              class_weight="balanced", tol=1e-4)),
    ])


def oversample(X: pd.Series, y: pd.Series):
    """Oversample positives in the training part."""
    pos = y.to_numpy() == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if 0 < n_pos < n_neg:
        reps = min(max(1, math.ceil((int(0.6 * n_neg) - n_pos) / max(1, n_pos))), 8)
        if reps > 1:
            X = pd.concat([X] + [X[pos]] * (reps - 1), ignore_index=True)
            y = pd.concat([y] + [y[pos]] * (reps - 1), ignore_index=True)
    return X, y


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)[:-1]
    return float(thr[int(np.argmax(f1s))]) if len(thr) else 0.5


def evaluate(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    out = {"n": int(len(y)), "pos": int(np.sum(y)), "threshold": float(thr),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0))}
    if len(set(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    return out


def fit_and_save(kind: str, Xtr, ytr, Xva, yva, out_path: Path, extra_eval=None, label_col="label") -> dict:
    pipe = make_tfidf_lr() if kind == "tfidf_lr" else make_char_sgd()
    if kind == "char_sgd":
        Xtr, ytr = oversample(Xtr.reset_index(drop=True), ytr.reset_index(drop=True))
    pipe.fit(Xtr, ytr)
    p_va = pipe.predict_proba(Xva)[:, 1]
    thr = best_f1_threshold(yva.to_numpy(), p_va)
    joblib.dump({"pipeline": pipe}, out_path)
    meta = {
        "model": "TFIDF+LogReg (pipeline)" if kind == "tfidf_lr" else "CharTFIDF+SGD(log_loss)",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label_col": label_col,
        "recommended_threshold": thr,
        "threshold_selected_on": "nlp_val (max F1)",
        "metrics": {"val": evaluate(yva.to_numpy(), p_va, thr)},
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    if extra_eval is not None:
        Xte, yte = extra_eval
        meta["metrics"]["test"] = evaluate(yte.to_numpy(), pipe.predict_proba(Xte)[:, 1], thr)
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/splits"))
    ap.add_argument("--out", default=str(ROOT / "models/nlp"))
    ap.add_argument("--container-models-dir", default="/models/nlp")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(data / "nlp_train.csv", dtype={"text": str}, keep_default_na=False)
    va = pd.read_csv(data / "nlp_val.csv", dtype={"text": str}, keep_default_na=False)
    te = pd.read_csv(data / "nlp_test.csv", dtype={"text": str}, keep_default_na=False)
    report = {"general": {}, "per_attack": {}}

    # general models on dataset labels
    for kind in ("tfidf_lr", "char_sgd"):
        m = fit_and_save(kind, tr.text, tr.label.astype(int), va.text, va.label.astype(int),
                         out / f"nlp_{kind}.joblib", extra_eval=(te.text, te.label.astype(int)))
        report["general"][kind] = m
        print(kind, json.dumps(m["metrics"]), flush=True)

    # per-attack topic models on rule-based labels
    rm = retrain_module()
    rules = json.loads(Path(rm.RULES_FILE).read_text(encoding="utf-8"))
    compiled = rm._compile_rules(rules)

    def silver(df):
        rows = [rm._label_from_signals(rm._scan_blob(rm._norm_text(t), compiled), rm.SILVER_MIN_PARTIAL) for t in df.text]
        return pd.DataFrame(rows, index=df.index)

    s_tr, s_va = silver(tr), silver(va)
    per_attack = {}
    for atk in rules.get("attack_patterns", {}):
        col = f"label_{atk}"
        pos_tr, pos_va = int(s_tr[col].sum()), int(s_va[col].sum())
        if pos_tr < MIN_POS_PER_ATTACK or pos_va < 1:
            report["per_attack"][atk] = {"skipped": True, "pos_train": pos_tr, "pos_val": pos_va}
            print("skip", atk, pos_tr, pos_va, flush=True)
            continue
        entry = {}
        for kind in ("tfidf_lr", "char_sgd"):
            fn = f"nlp_{kind}__{atk}.joblib"
            m = fit_and_save(kind, tr.text, s_tr[col], va.text, s_va[col], out / fn, label_col=f"silver:{col}")
            entry[kind] = f"{a.container_models_dir}/{fn}"
            report["per_attack"].setdefault(atk, {})[kind] = m["metrics"]["val"]
        per_attack[atk] = entry
        print(atk, pos_tr, pos_va, flush=True)

    (out / "nlp_meta.json").write_text(json.dumps({
        "tfidf_lr": f"{a.container_models_dir}/nlp_tfidf_lr.joblib",
        "char_sgd": f"{a.container_models_dir}/nlp_char_sgd.joblib",
        "per_attack": per_attack,
    }, indent=2))
    (out / "train_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
