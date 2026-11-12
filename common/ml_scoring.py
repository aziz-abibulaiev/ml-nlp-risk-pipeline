"""
ML anomaly scoring for model bundles.

raw_anomaly() is higher for more anomalous inputs. The calibration maps it to [0, 1] with the
stored ECDF, otherwise with a sigmoid. aggregate() combines the detectors (max, mean, median).
"""
from __future__ import annotations

import numpy as np

FORMAT = "ml-bundle-1"


def is_bundle(obj) -> bool:
    return isinstance(obj, dict) and obj.get("format") == FORMAT


def raw_anomaly(bundle: dict, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    X = np.asarray(X, dtype="float32")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    Xs = bundle["scaler"].transform(X).astype("float32")
    kind = bundle["kind"]
    model = bundle["model"]
    if kind in ("if", "ocsvm"):
        return -np.asarray(model.decision_function(Xs), dtype="float64")
    if kind == "lstm_ae":
        seq = Xs.reshape(Xs.shape[0], Xs.shape[1], 1)
        rec = model.predict(seq, batch_size=batch_size, verbose=0)
        return np.mean((seq - rec) ** 2, axis=(1, 2)).astype("float64")
    if kind == "ae":
        rec = model.predict(Xs, batch_size=batch_size, verbose=0)
        return np.mean((Xs - rec) ** 2, axis=1).astype("float64")
    raise ValueError(f"unknown kind {kind}")


def calibrated(raw, mu: float, sigma: float):
    sigma = sigma if sigma > 1e-9 else 1e-9
    z = (np.asarray(raw, dtype="float64") - mu) / sigma
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def ecdf(raw, x: list, p: list):
    return np.interp(np.asarray(raw, dtype="float64"), np.asarray(x, dtype="float64"),
                     np.asarray(p, dtype="float64"), left=0.0, right=1.0)


def calibrate_bundle(bundle: dict, raw):
    c = bundle["calibration"]
    if "ecdf" in c:
        return ecdf(raw, c["ecdf"]["x"], c["ecdf"]["p"])
    return calibrated(raw, float(c["mu"]), float(c["sigma"]))


def aggregate(scores, how: str = "max"):
    a = np.asarray(scores, dtype="float64")
    if how == "mean":
        return a.mean(axis=0)
    if how == "median":
        return np.median(a, axis=0)
    return a.max(axis=0)


def score_one(bundle: dict, vec) -> float:
    raw = raw_anomaly(bundle, np.asarray(vec, dtype="float32").reshape(1, -1))
    return float(np.clip(calibrate_bundle(bundle, raw)[0], 0.0, 1.0))
