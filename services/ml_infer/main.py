import os
from common.ml_scoring import is_bundle, score_one as bundle_score_one
import json
import logging
import joblib
import hashlib
import math
import uvicorn
import numpy as np

from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional
from fastapi import FastAPI, Header, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Model registry
ANOM_CALIB: Dict[str, Dict[str, float]] = {}
MODEL_PATH_IF      = os.getenv("MODEL_PATH_IF",      os.getenv("MODEL_PATH", "/models/ml/if_model.joblib"))
MODEL_PATH_OCSVM   = os.getenv("MODEL_PATH_OCSVM",   "/models/ml/ocsvm_model.joblib")
MODEL_PATH_AE      = os.getenv("MODEL_PATH_AE",      "/models/ml/ae_model.joblib")
MODEL_PATH_LSTM_AE = os.getenv("MODEL_PATH_LSTM_AE", "/models/ml/lstm_ae_model.joblib")

MODEL_REGISTRY = {
    "if":       MODEL_PATH_IF,
    "ocsvm":    MODEL_PATH_OCSVM,
    "ae":       MODEL_PATH_AE,
    "lstm_ae":  MODEL_PATH_LSTM_AE,
}

# Impact policy
IMPACT_POLICY_PATH = Path(os.getenv("IMPACT_POLICY_PATH", "/app/policies/impact_policy.json"))
IMPACT_POLICY: dict = {}

ATTACK_TO_TOPIC = {
    # injections
    "sql_injection":          "SQL Injection",
    "nosql_injection":        "NoSQL Injection",
    "ldap_injection":         "LDAP Injection",
    "cmd_injection":          "Command Injection",
    "xss":                    "Cross-Site Scripting",

    # SSRF / cloud metadata
    "ssrf":                   "Server-Side Request Forgery",
    "ssrf_sensitive":         "Cloud Metadata SSRF",

    # secrets / data
    "api_key_exposure":       "API Key Exposure",
    "data_exposure":          "Sensitive Data Exposure",

    # auth / id
    "auth_bruteforce":        "Authentication Bruteforce",
    "credential_stuffing":    "Credential Stuffing",
    "idor":                   "Insecure Direct Object Direct Reference",

    # misc
    "misconfiguration":       "Security Misconfiguration",
    "dos":                    "Denial of Service",
    "websocket":              "WebSocket Abuse",
    "business_logic":         "Business Logic Abuse",
}

# Rules-supervised classifier
ML_RULES_CLF_OUT   = os.getenv("ML_RULES_CLF_OUT", "/models/ml/ml_rules_clf.joblib")
ML_RULES_META_PATH = os.getenv("ML_RULES_META_PATH", "/models/ml/ml_rules_clf.meta.json")

ML_RULES: Optional[dict] = None
ML_RULES_META: dict = {}
ML_RULES_FEATS: List[str] = []
ML_RULES_THR: float = 0.6

# Cache of loaded model objects
MODELS: Dict[str, object] = {}

_DEFAULT_FEATURE_ORDER = (
  "endpoint_len,header_count,payload_size,headers_entropy,"
  "query_param_count,path_depth,suspicious_char_ratio,is_json_payload,http_method_code"
)
FEATURE_ORDER = os.getenv("FEATURE_ORDER", _DEFAULT_FEATURE_ORDER)

# HTTP method encoding to numeric code
METHOD_CODE_MAP = {
    "GET": 1, "POST": 2, "PUT": 3, "DELETE": 4, "PATCH": 5, "HEAD": 6, "OPTIONS": 7,
    "OTHER": 0, "UNKNOWN": 0
}
ONEHOT_KEYS = ["method_GET","method_POST","method_PUT","method_DEL","method_DELETE",
               "method_Patch","method_PATCH","method_HEAD","method_OPTIONS","method_Other","method_OTHER"]

ML_THRESHOLD_RAW = os.getenv("ML_THRESHOLD", "").strip()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Prometheus metrics
SCORE_REQS = Counter("ml_score_requests_total", "Total /score requests", ["status"])
SCORE_LAT  = Histogram("ml_score_latency_seconds", "Latency of /score",
                       buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5))
MODEL_SCORE_LAT = Histogram("ml_model_infer_latency_seconds", "Per-model inference latency", ["kind"],
                            buckets=(0.001, 0.003, 0.007, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1))
MODEL_SCORE_ERRS = Counter("ml_model_infer_errors_total", "Per-model inference errors", ["kind"])

MODEL_EXISTS = Gauge("ml_model_file_exists", "Model file exists on disk (0/1)", ["kind"])
MODEL_LOADED = Gauge("ml_model_loaded", "Model loaded in memory (0/1)", ["kind"])

# Calibration params for Isolation Forest scoring
IF_CALIB_MU    = float(os.getenv("IF_CALIB_MU", "0.0"))
IF_CALIB_SIGMA = float(os.getenv("IF_CALIB_SIGMA", "0.1"))

CFG_PATH = Path("/models/ml/ml_config.json")

DEFAULT_CFG = {
    "fallback": {
        "max_lat_ms": 1000.0,
        "max_payload_b": 10000.0,
        "max_qp": 10.0,
        "max_rate_rps": 5.0,
        "w_lat": 0.10,
        "w_susp": 0.30,
        "w_ps": 0.10,
        "w_qp": 0.10,
        "w_json": 0.05,
        "w_ip": 0.00,
        "w_rate": 0.15,
        "w_rule_strong": 0.40,
        "w_rule_partial": 0.20,
    },
    "routing": {
        "heavy": {
            "susp_thr": 0.20,
            "psize_thr": 1000.0,
            "qp_thr": 4.0,
            "lat_thr": 500.0,
        }
    }
}

# Logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","service":"ml-inference","msg":"%(message)s"}',
)
log = logging.getLogger("ml-inference")

PRELOAD = os.getenv("PRELOAD_MODELS", "false").lower() in ("1", "true", "yes", "on")

def _load_impact_policy():
    """Lazy loading of impact_policy.json (used for topic_bump)."""
    global IMPACT_POLICY
    try:
        if IMPACT_POLICY_PATH.exists():
            IMPACT_POLICY = json.loads(IMPACT_POLICY_PATH.read_text())
        else:
            IMPACT_POLICY = {}
            log.warning('impact_policy_not_found path="%s"', str(IMPACT_POLICY_PATH))
    except Exception:
        IMPACT_POLICY = {}
        log.exception("impact_policy_load_failed")


def attack_type_to_topic_bump(attack_type: str) -> float:
    """Map internal attack_type (sql_injection, xss, ssrf_sensitive, ...)
    to impact_policy.caps.topic_bump[...] (float 0..1).

    If the attack type is not present in the mapping/policy, return 0.0.
    """
    if not attack_type:
        return 0.0
    atk = attack_type.strip().lower()
    topic_name = ATTACK_TO_TOPIC.get(atk)
    if not topic_name:
        return 0.0
    caps = IMPACT_POLICY.get("caps") or {}
    tb = caps.get("topic_bump") or {}
    try:
        return float(tb.get(topic_name, 0.0))
    except Exception:
        return 0.0

def _load_anomaly_calib():
    """Load calibration parameters for all anomaly models from *.meta.json:
      {kind: {"mu": ..., "sigma": ...}}.
    If metadata is missing for a model, skip calibration for that model.
    """
    global ANOM_CALIB
    ANOM_CALIB = {}
    for kind, path in MODEL_REGISTRY.items():
        if not path:
            continue
        mp = Path(path).with_suffix(".meta.json")
        if not mp.exists():
            continue
        try:
            meta = json.loads(mp.read_text())
            calib = meta.get("calibration") or {}
            mu = float(calib.get("mu", 0.0))
            sg = float(calib.get("sigma", 0.1))
            if sg <= 1e-9:
                sg = 0.1
            ANOM_CALIB[kind] = {"mu": mu, "sigma": sg}
            log.info(
                'anomaly_calib_loaded kind="%s" mu=%.6f sigma=%.6f',
                kind, mu, sg
            )
        except Exception:
            log.exception("anomaly_calib_read_failed kind=%s meta=%s", kind, str(mp))

def _load_rules_clf():
    """Load the rules-supervised classifier and metadata (if present on disk)."""
    global ML_RULES, ML_RULES_META, ML_RULES_FEATS, ML_RULES_THR
    ML_RULES = None
    ML_RULES_META = {}
    ML_RULES_FEATS = []
    ML_RULES_THR = 0.6
    try:
        p = Path(ML_RULES_CLF_OUT)
        if p.exists():
            ML_RULES = joblib.load(p)
        mp = Path(ML_RULES_META_PATH)
        if mp.exists():
            ML_RULES_META = json.loads(mp.read_text())
            # try to obtain the feature list from meta, or from the object itself
            ML_RULES_FEATS = (
                ML_RULES_META.get("feature_order")
                or (ML_RULES.get("features") if isinstance(ML_RULES, dict) else None)
                or []
            )
            ML_RULES_THR = float(ML_RULES_META.get("recommended_threshold", 0.6))
        log.info(
            'rules_clf_loaded exists_model=%s exists_meta=%s feats=%d thr=%.3f',
            str(Path(ML_RULES_CLF_OUT).exists()),
            str(Path(ML_RULES_META_PATH).exists()),
            len(ML_RULES_FEATS),
            ML_RULES_THR,
        )
    except Exception:
        log.exception("rules_clf_load_failed")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global IF_CALIB_MU, IF_CALIB_SIGMA
    if PRELOAD:
        for kind, path in MODEL_REGISTRY.items():
            exists = bool(path and Path(path).exists())
            MODEL_EXISTS.labels(kind).set(1.0 if exists else 0.0)
            if not exists:
                log.info(f'model_preload_skip kind="{kind}" path="{path}" exists=False')
                MODEL_LOADED.labels(kind).set(0.0)
                continue
            try:
                m = joblib.load(path)
                MODELS[kind] = m
                MODEL_LOADED.labels(kind).set(1.0)
                MODEL_EXISTS.labels(kind).set(1.0 if Path(path).exists() else 0.0)
                log.info(f'model_preloaded kind="{kind}" path="{path}" type="{type(m).__name__}"')
            except Exception:
                MODEL_LOADED.labels(kind).set(0.0)
                log.exception(f'model_preload_failed kind="{kind}" path="{path}"')
    else:
        for kind, path in MODEL_REGISTRY.items():
            MODEL_EXISTS.labels(kind).set(1.0 if (path and Path(path).exists()) else 0.0)
            MODEL_LOADED.labels(kind).set(1.0 if (kind in MODELS) else 0.0)

    # IF calibration from meta
    meta_path = Path(MODEL_REGISTRY.get("if","")).with_suffix(".meta.json")
    if meta_path.exists():
        try:
            with meta_path.open("r") as f:
                meta = json.load(f)
            calib = meta.get("calibration") or {}
            mu = float(calib.get("mu", IF_CALIB_MU))
            sg = float(calib.get("sigma", IF_CALIB_SIGMA))
            IF_CALIB_MU, IF_CALIB_SIGMA = mu, (sg if sg>1e-9 else IF_CALIB_SIGMA)
            log.info(f'if_calibration_loaded mu={IF_CALIB_MU:.6f} sigma={IF_CALIB_SIGMA:.6f}')
        except Exception:
            log.exception("if_calibration_meta_read_failed")

    _load_rules_clf()
    _load_anomaly_calib()
    _load_impact_policy()
    yield
    MODELS.clear()

app = FastAPI(title="ml-inference", version="1.0.0", lifespan=lifespan)

# Models I/O
class ScoreIn(BaseModel):
    numeric_feats: Dict[str, float] = Field(..., description="Numeric features produced by the preprocess service")
    hash_id: str = Field(..., description="Stable event identifier")

class ScoreOut(BaseModel):
    hash_id: str
    ml_score: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    threshold_hit: bool

class ScoreBatchIn(BaseModel):
    items: List[ScoreIn] = Field(..., min_items=1)

class ScoreBatchOut(BaseModel):
    results: List[Dict]

# Helpers for scoring
def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5

def _calibrated_anomaly_score(kind: str, decision_value: float) -> float:
    """
    Calibrate anomaly score (e.g., from Isolation Forest, OCSVM, AE) to [0..1] using z-score and sigmoid.
    If calibration parameters are missing for the model kind, return 0.5 (neutral score).
    
    Args:
        kind (str): Model kind (e.g., "if", "ocsvm", "ae", "lstm_ae").
        decision_value (float): Raw anomaly score from the model.
    Returns:
        float: Calibrated score in [0..1].
    """
    try:
        c = ANOM_CALIB.get(kind)
        if not c:
            return 0.5
        mu = float(c["mu"])
        sg = float(c["sigma"])
        if sg <= 1e-9:
            sg = 0.1
        z = -(float(decision_value) - mu) / sg
        return _sigmoid(z)
    except Exception:
        return 0.5

def _feature_sig() -> str:
    order_raw = os.getenv("FEATURE_ORDER", "")
    s = ",".join([x.strip() for x in order_raw.split(",") if x.strip()]).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:8] if s else ""

def _load_recommended_threshold() -> Optional[float]:
    """
    Load recommended_threshold from:
    1) ML_RULES_META (if present)
    2) *.meta.json files alongside each model in MODEL_REGISTRY
    Returns the maximum recommended_threshold found, or None if none found.
    """
    recs: List[float] = []
    # 1) rules_clf meta
    try:
        mp = Path(ML_RULES_META_PATH)
        if mp.exists():
            meta = json.loads(mp.read_text())
            rt = meta.get("recommended_threshold")
            if isinstance(rt, (int, float)):
                recs.append(float(rt))
    except Exception:
        pass
    # 2) *.meta.json files alongside each model (if/ocsvm/ae/lstm_ae)
    for kind, path in MODEL_REGISTRY.items():
        try:
            if not path:
                continue
            mp = Path(path).with_suffix(".meta.json")
            if not mp.exists():
                continue
            meta = json.loads(mp.read_text())
            rt = meta.get("recommended_threshold")
            if isinstance(rt, (int, float)):
                recs.append(float(rt))
        except Exception:
            continue
    return max(recs) if recs else None

def _effective_threshold() -> float:
    """ENV ML_THRESHOLD > meta(recommended_threshold) > 0.60."""
    if ML_THRESHOLD_RAW:
        try:
            return float(ML_THRESHOLD_RAW)
        except Exception:
            log.warning('bad_env_threshold value="%s"', ML_THRESHOLD_RAW)
    rt = _load_recommended_threshold()
    return float(rt) if isinstance(rt, (float, int)) else 0.60

def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d

CFG = deepcopy(DEFAULT_CFG)
try:
    if CFG_PATH.exists():
        with CFG_PATH.open("r") as f:
            user_cfg = json.load(f)
        _deep_update(CFG, user_cfg or {})
except Exception:
    logging.getLogger("ml-inference").exception("ml_config_load_failed")

def load_model_by_kind(kind: str):
    kind = (kind or "if").lower()
    if kind in MODELS:
        return MODELS[kind]

    path = MODEL_REGISTRY.get(kind)
    if path and Path(path).exists():
        try:
            m = joblib.load(path)
            MODELS[kind] = m
            MODEL_LOADED.labels(kind).set(1.0)
            MODEL_EXISTS.labels(kind).set(1.0 if Path(path).exists() else 0.0)
            log.info(f'model_loaded kind="{kind}" path="{path}" type="{type(m).__name__}"')
            return m
        except Exception:
            log.exception(f'model_load_failed kind="{kind}" path="{path}"')
            return None
    else:
        log.warning(f'model_missing kind="{kind}" path="{path}" (fallback scoring will be used)')
        return None

def _feature_vector(numeric_feats: Dict[str, float], order: List[str]) -> List[float]:
    if order:
        return [float(numeric_feats.get(k, 0.0)) for k in order]
    return [float(numeric_feats[k]) for k in sorted(numeric_feats.keys())]

def _fallback_score(nf: Dict[str, float]) -> float:
    fb = CFG["fallback"]

    lat   = float(nf.get("latency_ms", 0.0))
    susp  = float(nf.get("suspicious_char_ratio", 0.0))
    ps    = float(nf.get("payload_size", 0.0))
    qp    = float(nf.get("query_param_count", 0.0))
    is_js = float(nf.get("is_json_payload", 0.0))
    ip_pr = float(nf.get("ip_private", 0.0))
    rate  = float(nf.get("req_rate_60s", 0.0))

    lat_s = min(lat  / max(1.0, fb["max_lat_ms"]),    1.0)
    ps_s  = min(ps   / max(1.0, fb["max_payload_b"]), 1.0)
    qp_s  = min(qp   / max(1.0, fb["max_qp"]),        1.0)
    rt_s  = min(rate / max(0.1, fb["max_rate_rps"]),  1.0)

    rule_strong  = float(nf.get("rule_any_strong", 0.0))
    rule_partial = float(nf.get("rule_any_partial", 0.0))

    raw = (
        fb["w_lat"]  * lat_s  +
        fb["w_susp"] * susp   +
        fb["w_ps"]   * ps_s   +
        fb["w_qp"]   * qp_s   +
        fb["w_json"] * is_js  +
        fb["w_ip"]   * ip_pr  +
        fb["w_rate"] * rt_s   +
        fb["w_rule_strong"]  * rule_strong +
        fb["w_rule_partial"] * rule_partial
    )
    return max(0.0, min(1.0, raw))

def _infer_method_from_onehot(nf: Dict[str, float]) -> tuple[int, float]:
    candidates = {
        "GET": float(nf.get("method_GET", 0.0)),
        "POST": float(nf.get("method_POST", 0.0)),
        "PUT": float(nf.get("method_PUT", 0.0)),
        "DELETE": float(nf.get("method_DELETE", nf.get("method_DEL", 0.0))),
        "PATCH": float(nf.get("method_PATCH", nf.get("method_Patch", 0.0))),
        "HEAD": float(nf.get("method_HEAD", 0.0)),
        "OPTIONS": float(nf.get("method_OPTIONS", 0.0)),
        "OTHER": float(nf.get("method_Other", nf.get("method_OTHER", 0.0))),
    }
    best = max(candidates.items(), key=lambda kv: kv[1])
    if best[1] >= 0.99:
        return METHOD_CODE_MAP.get(best[0], 0), 1.0
    if best[1] > 0.0:
        return METHOD_CODE_MAP.get(best[0], 0), 0.5
    return 0, 0.0

def _try_synthesize_method(nf: Dict[str, float]) -> tuple[int, float]:
    code = int(nf.get("http_method_code", -1))
    acc  = float(nf.get("http_method_accuracy", -1.0))

    if code >= 0 and acc >= 0.0:
        return code, acc

    code_oh, acc_oh = _infer_method_from_onehot(nf)
    if code_oh != 0 or acc_oh > 0.0:
        return code_oh, acc_oh

    return 0, 0.0

def adapt_features(nf_in: Dict[str, float]) -> Dict[str, float]:
    """Return a copy of the feature dictionary and always remove method_* one-hot
    indicators after synthesizing http_method_code/http_method_accuracy.
    """
    nf = dict(nf_in)

    code, acc = _try_synthesize_method(nf)
    nf.setdefault("http_method_code", float(code))
    nf.setdefault("http_method_accuracy", float(acc))

    # always remove one-hot method indicators
    for k in list(nf.keys()):
        if k.startswith("method_"):
            nf.pop(k, None)

    return nf

def _auto_select_models(nf: Dict[str, float]) -> List[str]:
    """Select all available anomaly models:
      • if, ocsvm, ae, lstm_ae — when their model files are present,
      • nf is used only to keep the function signature consistent.

    Routing by "heaviness" (susp/psize/qp/lat) is disabled — all models
    that exist on disk participate in scoring.
    """
    sel: List[str] = []
    for kind, path in MODEL_REGISTRY.items():
        if not path:
            continue
        # use the model if the file exists
        if (kind in MODELS) or Path(path).exists():
            sel.append(kind)
    return sel

def _score_one(kind: str, vec: List[float], nf: Dict[str, float]) -> Optional[float]:
    """
    Score a single feature vector with the specified model kind.
    Supports sklearn-like models (IF, OCSVM), Keras AE/LSTM AE (as dict bundles or pure models).
    Returns the calibrated anomaly score in [0..1], or None on failure.
    1) sklearn-like anomaly models (IF, OCSVM, custom wrappers)
    2) Keras AE/LSTM AE as bundle from retrain: dict{"model", "scaler", "type", ...}
    3) pure keras.Model (without dict wrapper)
    4) None if nothing matches or on error.
    """
    kind = (kind or "if").lower()
    try:
        m = load_model_by_kind(kind)
        if m is None:
            return None

        # 0) calibrated bundles (common/ml_scoring.py)
        if is_bundle(m):
            order = [x.strip() for x in FEATURE_ORDER.split(",") if x.strip()]
            if order != list(m["feature_order"]):
                raise ValueError(f"feature order mismatch for {kind}")
            return bundle_score_one(m, vec)

        # 1) sklearn-like anomaly models (IF, OCSVM, custom wrappers)
        if hasattr(m, "decision_function"):
            raw = float(m.decision_function([vec])[0])
            return _calibrated_anomaly_score(kind, raw)

        if hasattr(m, "score_samples"):
            raw = float(m.score_samples([vec])[0])
            return _calibrated_anomaly_score(kind, raw)

        # 2) Keras AE/LSTM AE as bundle from retrain: dict{"model", "scaler", "type", ...}
        if isinstance(m, dict) and "model" in m:
            mdl = m["model"]
            scaler = m.get("scaler", None)
            mtype = str(m.get("type", "")).lower()

            x = np.array(vec, dtype="float32").reshape(1, -1)
            if scaler is not None:
                x = scaler.transform(x).astype("float32")

            if mtype == "lstm_autoencoder":
                # retrain trained on shape (batch, timesteps, 1)
                x_seq = x.reshape(x.shape[0], x.shape[1], 1)
                recon = mdl.predict(x_seq, verbose=0)
                err = float(np.mean((x_seq - recon) ** 2))
            else:
                # by default — dense autoencoder
                recon = mdl.predict(x, verbose=0)
                err = float(np.mean((x - recon) ** 2))

            # recon_err → calibrated score 0..1
            return _calibrated_anomaly_score(kind, err)

        # 3) pure keras.Model (without dict wrapper)
        if hasattr(m, "predict"):
            x = np.array(vec, dtype="float32").reshape(1, -1)
            recon = m.predict(x, verbose=0)
            try:
                err = float(np.mean((x - recon) ** 2))
            except Exception:
                err = 0.0
            return _calibrated_anomaly_score(kind, err)

        # nothing matched
        return None

    except Exception:
        MODEL_SCORE_ERRS.labels(kind).inc()
        log.exception("model_infer_failed kind=%s", kind)
        return None

ML_AGGREGATE = os.getenv("ML_AGGREGATE", "max").lower()  # max | mean | median

def _aggregate_scores(per: List[Dict[str, float]]) -> float:
    vals = [float(p["score"]) for p in per if "score" in p]
    if not vals:
        return 0.0
    if ML_AGGREGATE == "mean":
        return sum(vals) / len(vals)
    if ML_AGGREGATE == "median":
        v = sorted(vals); n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0
    return max(vals)

def _score_rules_clf(nf: Dict[str, float]) -> Optional[float]:
    """Return the probability of class 1 from the rules-supervised GBDT classifier, or None."""
    try:
        if not (ML_RULES and isinstance(ML_RULES, dict)):
            return None
        feats = ML_RULES_FEATS or ML_RULES.get("features", [])
        if not feats:
            return None
        x = np.array([[ float(nf.get(k, 0.0)) for k in feats ]], dtype=float)
        clf = ML_RULES.get("clf")
        if clf is None:
            return None
        if hasattr(clf, "predict_proba"):
            return float(clf.predict_proba(x)[0, 1])
        if hasattr(clf, "decision_function"):
            raw = float(clf.decision_function(x)[0])
            return 1.0 / (1.0 + math.exp(-raw))
        if hasattr(clf, "predict"):
            pred = float(clf.predict(x)[0])
            return max(0.0, min(1.0, pred))
    except Exception:
        log.exception("rules_clf_infer_failed")
    return None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    models_status = {
        kind: {
            "path": path,
            "exists": bool(path and Path(path).exists()),
            "loaded": (kind in MODELS),
        }
        for kind, path in MODEL_REGISTRY.items()
    }
    for kind, st in models_status.items():
        MODEL_EXISTS.labels(kind).set(1.0 if st["exists"] else 0.0)
        MODEL_LOADED.labels(kind).set(1.0 if st["loaded"] else 0.0)

    feature_order_list = [k.strip() for k in FEATURE_ORDER.split(",") if k.strip()]

    # compare with retrain metadata
    meta_mismatch = False
    meta_sig = ""
    meta_path = Path(MODEL_REGISTRY["if"]).with_suffix(".meta.json")
    if meta_path.exists():
        try:
            with meta_path.open("r") as f:
                meta = json.load(f)
            if isinstance(meta.get("feature_order"), list):
                raw = ",".join(meta["feature_order"]).encode("utf-8")
                meta_sig = hashlib.sha1(raw).hexdigest()[:8]
                meta_mismatch = (meta_sig != _feature_sig())
        except Exception:
            pass

    return {
        "models": models_status,
        "supported_kinds": list(MODEL_REGISTRY.keys()),
        "feature_order": feature_order_list,
        "feature_order_sig": _feature_sig(),
        "threshold": _effective_threshold(),
        "feature_order_match": (not meta_sig) or (meta_mismatch is False),
        "config": CFG,
        "rules_clf_loaded": bool(ML_RULES),
        "rules_clf_feats": len(ML_RULES_FEATS),
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/models")
def ml_models():
    items = {}
    for kind, path in MODEL_REGISTRY.items():
        p = Path(path) if path else Path("")
        exists = bool(path and p.exists())
        loaded = (kind in MODELS)
        items[kind] = {
            "path": path,
            "exists": exists,
            "loaded": loaded,
        }
    return {
        "threshold": _effective_threshold(),
        "feature_order_sig": _feature_sig(),
        "models": items,
    }

@app.post("/score", response_model=ScoreOut)
def score(inp: ScoreIn, response: Response):
    """Main scoring endpoint:
      • adapt features (adapt_features → adds http_method_code, removes one-hot),
      • select models (_auto_select_models),
      • compute a unified anomaly score per model via _score_one() (with calibration),
      • optionally add the rules-supervised classifier score,
      • aggregate max(per_model_scores) as ml_score,
      • if no model scores are available, fall back to _fallback_score.
    """
    with SCORE_LAT.time():
        try:
            nf_raw = dict(inp.numeric_feats or {})
            nf = adapt_features(nf_raw)

            order = [x.strip() for x in FEATURE_ORDER.split(",") if x.strip()]
            vec = _feature_vector(nf, order)

            # auto-select models (if, ocsvm, ae, lstm_ae as needed)
            kinds = _auto_select_models(nf)

            per: List[Dict[str, float]] = []
            for k in kinds:
                with MODEL_SCORE_LAT.labels(k).time():
                    s = _score_one(k, vec, nf)
                if s is not None:
                    per.append({"kind": k, "score": float(max(0.0, min(1.0, s)))})
                else:
                    MODEL_SCORE_ERRS.labels(k).inc()

            # rules-supervised GBDT
            if ML_RULES and (ML_RULES_FEATS or (isinstance(ML_RULES, dict) and ML_RULES.get("features"))):
                with MODEL_SCORE_LAT.labels("rules_clf").time():
                    rs = _score_rules_clf(nf)
                if rs is not None:
                    per.append({"kind": "rules_clf", "score": float(max(0.0, min(1.0, rs)))})
                else:
                    MODEL_SCORE_ERRS.labels("rules_clf").inc()

            # if there are no per-model scores — fallback
            if not per:
                ml_score = _fallback_score(nf)
                per_model_scores = {"fallback": ml_score}
            else:
                ml_score = _aggregate_scores(per)
                per_model_scores = {p["kind"]: p["score"] for p in per}

            thr = _effective_threshold()
            hit = bool(ml_score >= thr)

            SCORE_REQS.labels("ok").inc()
            try:
                log.info(
                    'ml_score hash_id="%s" score=%.4f thr=%.4f hit=%s per_models=%s',
                    inp.hash_id, ml_score, thr, str(hit),
                    json.dumps(per_model_scores)
                )
            except Exception:
                pass

            return ScoreOut(
                hash_id=inp.hash_id,
                ml_score=float(ml_score),
                threshold=float(thr),
                threshold_hit=hit,
            )
        except Exception:
            SCORE_REQS.labels("error").inc()
            log.exception("score_failed")
            response.status_code = 500
            fb = _fallback_score(dict(inp.numeric_feats or {}))
            thr = _effective_threshold()
            return ScoreOut(
                hash_id=inp.hash_id,
                ml_score=float(fb),
                threshold=float(thr),
                threshold_hit=bool(fb >= thr),
            )

@app.post("/score_batch")
def score_batch(inp: ScoreBatchIn, x_trace_id: Optional[str] = Header(default=None, convert_underscores=False)):
    thr = _effective_threshold()
    trace = x_trace_id or "-"
    order = [k.strip() for k in FEATURE_ORDER.split(",") if k.strip()]
    out = []
    for item in inp.items:
        try:
            nf = adapt_features(item.numeric_feats)
            vec = _feature_vector(nf, order)
            kinds = _auto_select_models(nf)

            per: List[Dict[str, float]] = []
            for k in kinds:
                with MODEL_SCORE_LAT.labels(k).time():
                    s = _score_one(k, vec, nf)
                if s is not None:
                    per.append({"kind": k, "score": float(max(0.0, min(1.0, s)))})
                else:
                    MODEL_SCORE_ERRS.labels(k).inc()
            if ML_RULES and (ML_RULES_FEATS or (isinstance(ML_RULES, dict) and ML_RULES.get("features"))):
                with MODEL_SCORE_LAT.labels("rules_clf").time():
                    rs = _score_rules_clf(nf)
                if rs is not None:
                    per.append({"kind": "rules_clf", "score": float(max(0.0, min(1.0, rs)))})
                else:
                    MODEL_SCORE_ERRS.labels("rules_clf").inc()
            if not per:
                fb = _fallback_score(nf)
                per = [{"kind": "fallback", "score": fb}]

            ml_score = _aggregate_scores(per)
            out.append({
                "hash_id": item.hash_id,
                "per_model": per,
                "ml_score": ml_score,
                "threshold": thr,
                "threshold_hit": (ml_score >= thr),
                "used_models": [p["kind"] for p in per if p.get("kind") != "fallback"],
            })
        except Exception as e:
            SCORE_REQS.labels("error").inc()
            fb = _fallback_score(item.numeric_feats or {})
            out.append({
                "hash_id": item.hash_id,
                "per_model": [{"kind": "fallback", "score": fb}],
                "ml_score": fb,
                "threshold": thr,
                "threshold_hit": (fb >= thr),
                "used_models": [],
                "meta": {"error": "exception_fallback"},
            })
    return JSONResponse({"results": out})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))