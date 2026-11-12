#!/usr/bin/env python3
"""
Train the four ML anomaly detectors on benign training data.

Writes models/ml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.ml_scoring import FORMAT, calibrated, raw_anomaly  # noqa: E402

SEED = 42
N_OCSVM = 50_000
N_AE = 200_000

FEATURE_ORDER = [
    "endpoint_len", "header_count", "payload_size", "headers_entropy", "query_param_count",
    "path_depth", "suspicious_char_ratio", "is_json_payload", "http_method_code",
]


def load(p: Path):
    df = pd.read_csv(p)
    return df[FEATURE_ORDER].to_numpy(dtype="float32"), df["label"].to_numpy(dtype=int)


def sample(X: np.ndarray, n: int, seed: int = SEED):
    if X.shape[0] <= n:
        return X, np.arange(X.shape[0])
    idx = np.sort(np.random.RandomState(seed).choice(X.shape[0], size=n, replace=False))
    return X[idx], idx


def build_ae(input_dim: int):
    from tensorflow import keras
    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(64, activation="relu")(inp)
    x = keras.layers.Dense(32, activation="relu")(x)
    code = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(code)
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(input_dim, activation="linear")(x)
    m = keras.Model(inp, out, name="dense_autoencoder")
    m.compile(optimizer="adam", loss="mse")
    return m


def build_lstm_ae(timesteps: int, latent: int = 16):
    from tensorflow import keras
    inp = keras.Input(shape=(timesteps, 1))
    x = keras.layers.LSTM(latent * 2, return_sequences=True)(inp)
    x = keras.layers.LSTM(latent, return_sequences=False)(x)
    x = keras.layers.RepeatVector(timesteps)(x)
    x = keras.layers.LSTM(latent, return_sequences=True)(x)
    out = keras.layers.TimeDistributed(keras.layers.Dense(1))(x)
    m = keras.Model(inp, out, name="lstm_autoencoder")
    m.compile(optimizer="adam", loss="mse")
    return m


def metrics(y, s) -> dict:
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "n": int(len(y)), "pos": int(np.sum(y))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data/splits/features"))
    ap.add_argument("--out", default=str(ROOT / "models/ml"))
    ap.add_argument("--only", default="if,ocsvm,ae,lstm_ae")
    a = ap.parse_args()
    fd, out = Path(a.features), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    Xtr, ytr = load(fd / "ml_train_benign.csv")
    assert ytr.sum() == 0, "ML-train must be benign-only"
    Xvb, _ = load(fd / "ml_val_benign.csv")
    Xva, _ = load(fd / "ml_val_attacks.csv")
    Xte, yte = load(fd / "ml_test.csv")
    Xval = np.vstack([Xvb, Xva])
    yval = np.r_[np.zeros(len(Xvb), int), np.ones(len(Xva), int)]

    scaler = StandardScaler().fit(Xtr)
    report = {"feature_order": FEATURE_ORDER, "seed": SEED, "n_train": int(len(Xtr)), "models": {}}

    for kind in a.only.split(","):
        t0 = time.time()
        if kind == "if":
            Xfit, idx = Xtr, np.arange(len(Xtr))
            model = IsolationForest(n_estimators=200, contamination="auto", random_state=SEED, n_jobs=-1)
            model.fit(scaler.transform(Xfit))
            params = {"n_estimators": 200, "contamination": "auto"}
        elif kind == "ocsvm":
            Xfit, idx = sample(Xtr, N_OCSVM)
            model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
            model.fit(scaler.transform(Xfit))
            params = {"kernel": "rbf", "nu": 0.05, "gamma": "scale"}
        elif kind == "ae":
            Xfit, idx = sample(Xtr, N_AE)
            Xs = scaler.transform(Xfit).astype("float32")
            model = build_ae(Xs.shape[1])
            model.fit(Xs, Xs, epochs=20, batch_size=64, shuffle=True, validation_split=0.1, verbose=2)
            params = {"layers": [64, 32, 16, 32, 64], "epochs": 20, "batch": 64}
        elif kind == "lstm_ae":
            Xfit, idx = sample(Xtr, N_AE)
            Xs = scaler.transform(Xfit).astype("float32")
            seq = Xs.reshape(Xs.shape[0], Xs.shape[1], 1)
            model = build_lstm_ae(Xs.shape[1])
            model.fit(seq, seq, epochs=15, batch_size=64, shuffle=True, validation_split=0.1, verbose=2)
            params = {"latent": 16, "epochs": 15, "batch": 64, "timesteps": Xs.shape[1]}
        else:
            raise SystemExit(f"unknown model {kind}")

        bundle = {"format": FORMAT, "kind": kind, "feature_order": FEATURE_ORDER,
                  "scaler": scaler, "model": model}
        raw_vb = raw_anomaly(bundle, Xvb)
        mu, sigma = float(np.mean(raw_vb)), float(np.std(raw_vb))
        bundle["calibration"] = {"mu": mu, "sigma": sigma}
        s_val = calibrated(np.r_[raw_vb, raw_anomaly(bundle, Xva)], mu, sigma)
        s_test = calibrated(raw_anomaly(bundle, Xte), mu, sigma)

        path = out / f"{kind}_model.joblib"
        joblib.dump(bundle, path)
        np.save(out / f"{kind}_train_idx.npy", idx)
        np.save(out / f"{kind}_val_scores.npy", s_val.astype("float32"))
        meta = {
            "model": kind, "format": FORMAT, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "feature_order": FEATURE_ORDER, "n_fit": int(len(Xfit)), "params": params,
            "calibration": {"mu": mu, "sigma": sigma, "on": "ML-val benign", "direction": "higher_is_anomalous"},
            "metrics": {"ml_val": metrics(yval, s_val), "ml_test": metrics(yte, s_test)},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "train_seconds": round(time.time() - t0, 1),
        }
        (out / f"{kind}_model.meta.json").write_text(json.dumps(meta, indent=2))
        report["models"][kind] = meta
        print(kind, json.dumps(meta["metrics"]), f"{meta['train_seconds']}s", flush=True)

    np.save(out / "val_labels.npy", yval.astype("int8"))
    prev = out / "train_report.json"
    if prev.exists():
        old = json.loads(prev.read_text())
        old["models"].update(report["models"])
        report = old
    prev.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
