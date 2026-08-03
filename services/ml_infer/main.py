import os
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
MODEL_PATH_SUP_LR  = os.getenv("MODEL_PATH_SUP_LR",  "/models/ml/sup_lr_model.joblib")
MODEL_PATH_SUP_HGB = os.getenv("MODEL_PATH_SUP_HGB", "/models/ml/sup_hgb_model.joblib")
MODEL_PATH_SUP_RF  = os.getenv("MODEL_PATH_SUP_RF",  "/models/ml/sup_rf_model.joblib")
MODEL_PATH_SUP_ET  = os.getenv("MODEL_PATH_SUP_ET",  "/models/ml/sup_et_model.joblib")

MODEL_REGISTRY = {
    "if":       MODEL_PATH_IF,
    "ocsvm":    MODEL_PATH_OCSVM,
    "ae":       MODEL_PATH_AE,
    "lstm_ae":  MODEL_PATH_LSTM_AE,
    "sup_lr":   MODEL_PATH_SUP_LR,
    "sup_hgb":  MODEL_PATH_SUP_HGB,
    "sup_rf":   MODEL_PATH_SUP_RF,
    "sup_et":   MODEL_PATH_SUP_ET,
}
SUPERVISED_NUMERIC_KINDS = {"sup_lr", "sup_hgb", "sup_rf", "sup_et"}
NOVELTY_MODEL_KINDS = {"if", "ocsvm", "ae", "lstm_ae"}

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
ML_ENSEMBLE_CALIB: dict = {}
ML_AUX_NOVELTY_CALIB: dict = {}
MODEL_RECOMMENDED_THRESHOLDS: Dict[str, float] = {}
ENSEMBLE_RECOMMENDED_THRESHOLD: Optional[float] = None
AUX_NOVELTY_RECOMMENDED_THRESHOLD: Optional[float] = None

# Cache of loaded model objects
MODELS: Dict[str, object] = {}

_DEFAULT_FEATURE_ORDER = (
  "endpoint_len,header_count,payload_size,headers_entropy,"
  "query_param_count,path_depth,suspicious_char_ratio,is_json_payload,http_method_code,"
  "query_len,query_entropy,query_value_len_mean,query_value_len_max,empty_query_value_ratio,"
  "path_entropy,path_digit_ratio,path_upper_ratio,body_len,body_entropy,body_digit_ratio,"
  "body_punct_ratio,pct_encoded_ratio,header_value_total_len,header_value_max_len"
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
ANOM_SIGMOID_Z_CLIP = float(os.getenv("ANOM_SIGMOID_Z_CLIP", "30.0"))
ANOM_PROB_EPS = float(os.getenv("ANOM_PROB_EPS", "1e-12"))
if (not math.isfinite(ANOM_PROB_EPS)) or ANOM_PROB_EPS <= 0.0 or ANOM_PROB_EPS >= 0.1:
    ANOM_PROB_EPS = 1e-12
ML_ENSEMBLE_AGG = os.getenv("ML_ENSEMBLE_AGG", "mean_logit").strip().lower()
ML_PRIMARY_ENSEMBLE_AGG = os.getenv("ML_PRIMARY_ENSEMBLE_AGG", ML_ENSEMBLE_AGG or "mean_logit").strip().lower()
ML_AUX_NOVELTY_AGG = os.getenv("ML_AUX_NOVELTY_AGG", "mean_logit").strip().lower()
ML_AUX_NOVELTY_THRESHOLD_RAW = os.getenv("ML_AUX_NOVELTY_THRESHOLD", "").strip()

def _parse_kind_list(raw: str, default: str) -> List[str]:
    vals = [x.strip().lower() for x in str(raw or "").split(",") if x.strip()]
    if not vals:
        vals = [x.strip().lower() for x in default.split(",") if x.strip()]
    out: List[str] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out

ML_PRIMARY_CORE_KINDS = _parse_kind_list(
    os.getenv("ML_PRIMARY_CORE_KINDS", "sup_hgb,sup_rf,sup_et"),
    "sup_hgb,sup_rf,sup_et",
)
ML_AUX_NOVELTY_KINDS = _parse_kind_list(
    os.getenv("ML_AUX_NOVELTY_KINDS", "if,ocsvm,ae,lstm_ae"),
    "if,ocsvm,ae,lstm_ae",
)

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
    """Load calibration parameters for all anomaly models from *.meta.json.

    Structure per model:
      ANOM_CALIB[kind] = {
        "mu": float,
        "sigma": float,
        "orientation": "identity" | "negate",
        "temperature": float,  # optional post-calibration temperature scaling
        "attack_logistic": {"w": float, "b": float, "z_mu": float, "z_sigma": float, ...}  # optional
      }
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
            kind_norm = _normalize_anomaly_kind(kind)

            mu = float(calib.get("mu", 0.0))
            sg = float(calib.get("sigma", 0.1))
            if sg <= 1e-9:
                sg = 0.1
            try:
                temp = float(calib.get("temperature", 1.0))
            except Exception:
                temp = 1.0
            if (not math.isfinite(temp)) or temp <= 1e-6:
                temp = 1.0
            orientation = _resolve_score_orientation(kind_norm, calibration=calib)

            attack_log = calib.get("attack_logistic") or None
            if isinstance(attack_log, dict) and "w" in attack_log and "b" in attack_log:
                try:
                    w = float(attack_log["w"])
                    b0 = float(attack_log["b"])
                    has_z = ("z_mu" in attack_log) and ("z_sigma" in attack_log)
                    if (kind_norm in ("ae", "lstm_ae")) and (not has_z):
                        # Legacy AE/LSTM attack_logistic (without standardized input params)
                        # can collapse runtime scores near 0.5; use mu/sigma path instead.
                        log.info(
                            'anomaly_calib_skip_legacy_attack_logistic kind="%s" reason="missing_z_params"',
                            kind,
                        )
                        attack_log = None
                    else:
                        z_mu = float(attack_log.get("z_mu", 0.0))
                        z_sigma = float(attack_log.get("z_sigma", 1.0))
                        if (not math.isfinite(z_sigma)) or z_sigma <= 1e-12:
                            z_sigma = 1.0
                        logit_override = attack_log.get("orientation") if "orientation" in attack_log else "identity"
                        logit_orientation = _resolve_score_orientation(
                            kind_norm,
                            calibration=calib,
                            override=logit_override,
                        )
                        attack_log = {
                            "w": w,
                            "b": b0,
                            "orientation": logit_orientation,
                            "z_mu": z_mu,
                            "z_sigma": z_sigma,
                        }
                except Exception:
                    attack_log = None
            else:
                attack_log = None

            ANOM_CALIB[kind] = {
                "mu": mu,
                "sigma": sg,
                "orientation": orientation,
                "temperature": temp,
            }
            if attack_log is not None:
                ANOM_CALIB[kind]["attack_logistic"] = attack_log
            ecdf_risk = _normalize_ecdf_risk(calib.get("ecdf_risk"))
            if ecdf_risk is not None:
                ANOM_CALIB[kind]["ecdf_risk"] = ecdf_risk
            prior_cfg = calib.get("prior_correction")
            if isinstance(prior_cfg, dict) and bool(prior_cfg.get("enabled", False)):
                try:
                    p_cal = float(prior_cfg.get("calibration_prior"))
                    p_dep = float(prior_cfg.get("deployment_prior"))
                    if (
                        math.isfinite(p_cal)
                        and math.isfinite(p_dep)
                        and 0.0 < p_cal < 1.0
                        and 0.0 < p_dep < 1.0
                    ):
                        ANOM_CALIB[kind]["prior_correction"] = {
                            "enabled": True,
                            "calibration_prior": p_cal,
                            "deployment_prior": p_dep,
                            "method": str(prior_cfg.get("method", "odds_correction")),
                        }
                except Exception:
                    pass

            log.info(
                'anomaly_calib_loaded kind="%s" mu=%.6f sigma=%.6f orientation=%s temp=%.4f has_attack_logistic=%s has_ecdf_risk=%s has_prior_correction=%s',
                kind,
                mu,
                sg,
                orientation,
                temp,
                bool(attack_log),
                bool(ecdf_risk),
                bool(ANOM_CALIB[kind].get("prior_correction")),
            )
        except Exception:
            log.exception("anomaly_calib_read_failed kind=%s meta=%s", kind, str(mp))

def _load_ensemble_calib():
    """Load optional ensemble-level calibration from /models/ml/ml_ensemble.meta.json."""
    global ML_ENSEMBLE_CALIB, ML_AUX_NOVELTY_CALIB
    ML_ENSEMBLE_CALIB = {}
    ML_AUX_NOVELTY_CALIB = {}
    mp = Path("/models/ml/ml_ensemble.meta.json")
    if not mp.exists():
        return
    try:
        meta = json.loads(mp.read_text())
        calib = meta.get("calibration") or {}
        t = float(calib.get("temperature", 1.0))
        if (not math.isfinite(t)) or t <= 1e-6:
            t = 1.0
        ML_ENSEMBLE_CALIB = {"temperature": t}
        prior_cfg = calib.get("prior_correction")
        if isinstance(prior_cfg, dict) and bool(prior_cfg.get("enabled", False)):
            try:
                p_cal = float(prior_cfg.get("calibration_prior"))
                p_dep = float(prior_cfg.get("deployment_prior"))
                if (
                    math.isfinite(p_cal)
                    and math.isfinite(p_dep)
                    and 0.0 < p_cal < 1.0
                    and 0.0 < p_dep < 1.0
                ):
                    ML_ENSEMBLE_CALIB["prior_correction"] = {
                        "enabled": True,
                        "calibration_prior": p_cal,
                        "deployment_prior": p_dep,
                        "method": str(prior_cfg.get("method", "odds_correction")),
                    }
            except Exception:
                pass
        aux_calib = meta.get("aux_novelty_calibration")
        if isinstance(aux_calib, dict):
            try:
                aux_t = float(aux_calib.get("temperature", 1.0))
            except Exception:
                aux_t = 1.0
            if (not math.isfinite(aux_t)) or aux_t <= 1e-6:
                aux_t = 1.0
            ML_AUX_NOVELTY_CALIB = {"temperature": aux_t}
            aux_prior_cfg = aux_calib.get("prior_correction")
            if isinstance(aux_prior_cfg, dict) and bool(aux_prior_cfg.get("enabled", False)):
                try:
                    p_cal = float(aux_prior_cfg.get("calibration_prior"))
                    p_dep = float(aux_prior_cfg.get("deployment_prior"))
                    if (
                        math.isfinite(p_cal)
                        and math.isfinite(p_dep)
                        and 0.0 < p_cal < 1.0
                        and 0.0 < p_dep < 1.0
                    ):
                        ML_AUX_NOVELTY_CALIB["prior_correction"] = {
                            "enabled": True,
                            "calibration_prior": p_cal,
                            "deployment_prior": p_dep,
                            "method": str(aux_prior_cfg.get("method", "odds_correction")),
                        }
                except Exception:
                    pass
        log.info(
            'ml_ensemble_calib_loaded temperature=%.4f has_prior_correction=%s aux_temperature=%.4f aux_has_prior_correction=%s',
            t,
            bool(ML_ENSEMBLE_CALIB.get("prior_correction")),
            float((ML_AUX_NOVELTY_CALIB or {}).get("temperature", 1.0)),
            bool((ML_AUX_NOVELTY_CALIB or {}).get("prior_correction")),
        )
    except Exception:
        log.exception("ml_ensemble_calib_read_failed")
        ML_ENSEMBLE_CALIB = {}
        ML_AUX_NOVELTY_CALIB = {}

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


def _load_recommended_thresholds():
    """
    Load recommended thresholds from model metadata.
    - Ensemble threshold: /models/ml/ml_ensemble.meta.json:recommended_threshold
    - Per-model thresholds: <model>.meta.json:recommended_threshold
    - Rules classifier threshold: ML_RULES_META_PATH:recommended_threshold
    """
    global MODEL_RECOMMENDED_THRESHOLDS, ENSEMBLE_RECOMMENDED_THRESHOLD, AUX_NOVELTY_RECOMMENDED_THRESHOLD
    MODEL_RECOMMENDED_THRESHOLDS = {}
    ENSEMBLE_RECOMMENDED_THRESHOLD = None
    AUX_NOVELTY_RECOMMENDED_THRESHOLD = None

    try:
        em_path = Path("/models/ml/ml_ensemble.meta.json")
        if em_path.exists():
            meta = json.loads(em_path.read_text())
            rt = meta.get("recommended_threshold")
            if isinstance(rt, (int, float)):
                ENSEMBLE_RECOMMENDED_THRESHOLD = float(rt)
            rt_aux = meta.get("recommended_aux_novelty_threshold")
            if isinstance(rt_aux, (int, float)):
                AUX_NOVELTY_RECOMMENDED_THRESHOLD = float(rt_aux)
    except Exception:
        log.exception("ml_ensemble_threshold_read_failed")

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
                MODEL_RECOMMENDED_THRESHOLDS[kind] = float(rt)
        except Exception:
            continue

    try:
        mp = Path(ML_RULES_META_PATH)
        if mp.exists():
            meta = json.loads(mp.read_text())
            rt = meta.get("recommended_threshold")
            if isinstance(rt, (int, float)):
                MODEL_RECOMMENDED_THRESHOLDS["rules_clf"] = float(rt)
    except Exception:
        pass

    log.info(
        'recommended_thresholds_loaded ensemble=%s aux_novelty=%s per_model=%s',
        ENSEMBLE_RECOMMENDED_THRESHOLD,
        AUX_NOVELTY_RECOMMENDED_THRESHOLD,
        MODEL_RECOMMENDED_THRESHOLDS,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
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

    _load_rules_clf()
    _load_anomaly_calib()
    _load_ensemble_calib()
    _load_recommended_thresholds()
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
    ml_primary_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ml_aux_novelty_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ml_aux_novelty_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ml_aux_novelty_hit: Optional[bool] = None
    threshold: float = Field(..., ge=0.0, le=1.0)
    threshold_hit: bool
    per_model: List[Dict[str, object]] = Field(
        default_factory=list,
        description="Legacy list format [{'kind':..., 'score':...}] for orchestrator/fusion compatibility."
    )
    per_model_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-model scores (if/ocsvm/ae/lstm_ae/sup_*/rules_clf/fallback)"
    )
    primary_model_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Primary supervised-core model scores used for ml_score aggregation."
    )
    aux_novelty_model_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Auxiliary novelty model scores (IF/OCSVM/AE/LSTM-AE)."
    )
    per_model_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-model recommended thresholds from model metadata."
    )
    per_model_threshold_hits: Dict[str, bool] = Field(
        default_factory=dict,
        description="Per-model threshold hits (score >= per-model threshold)."
    )
    per_model_hit_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Mean of per-model threshold hits; diagnostic only."
    )

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

def _clip01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0

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
    calibration: Optional[dict] = None,
    override: Optional[str] = None,
) -> str:
    candidates: List[Optional[str]] = [override]
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

def _safe_logit(p: np.ndarray, eps: float | None = None) -> np.ndarray:
    if eps is None:
        eps = ANOM_PROB_EPS
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def _compress_monotone_knots(x: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

def _normalize_ecdf_risk(mapping: Optional[dict]) -> Optional[dict]:
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
    p = np.clip(p, ANOM_PROB_EPS, 1.0 - ANOM_PROB_EPS)
    return {
        "method": str(mapping.get("method", "ecdf_quantile")),
        "x": [float(v) for v in x],
        "p": [float(v) for v in p],
        "n_ref": int(mapping.get("n_ref", 0)) if isinstance(mapping.get("n_ref"), (int, float)) else 0,
        "n_knots": int(x.size),
    }

def _apply_ecdf_risk_mapping(stats: np.ndarray, mapping: Optional[dict]) -> Optional[np.ndarray]:
    norm = _normalize_ecdf_risk(mapping)
    if norm is None:
        return None
    x = np.asarray(norm["x"], dtype=float).reshape(-1)
    p = np.asarray(norm["p"], dtype=float).reshape(-1)
    s = np.asarray(stats, dtype=float).reshape(-1)
    out = np.interp(s, x, p)
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
    if (not math.isfinite(t)) or t <= 1e-6:
        t = 1.0
    s = np.clip(s, 0.0, 1.0)
    if abs(t - 1.0) <= 1e-9:
        return s
    logits = _safe_logit(s, eps=ANOM_PROB_EPS) / t
    return _sigmoid_stable(logits)

def _apply_prior_correction_probs(scores: np.ndarray, prior_cfg: Optional[dict]) -> np.ndarray:
    s = np.asarray(scores, dtype=float).reshape(-1)
    if not isinstance(prior_cfg, dict):
        return s
    if not bool(prior_cfg.get("enabled", False)):
        return s
    try:
        p_cal = float(prior_cfg.get("calibration_prior"))
        p_dep = float(prior_cfg.get("deployment_prior"))
    except Exception:
        return s
    if (
        (not math.isfinite(p_cal))
        or (not math.isfinite(p_dep))
        or p_cal <= 0.0
        or p_cal >= 1.0
        or p_dep <= 0.0
        or p_dep >= 1.0
    ):
        return s
    odds = np.exp(_safe_logit(np.clip(s, ANOM_PROB_EPS, 1.0 - ANOM_PROB_EPS), eps=ANOM_PROB_EPS))
    prior_ratio = (p_dep / (1.0 - p_dep)) / (p_cal / (1.0 - p_cal))
    odds_adj = odds * prior_ratio
    return odds_adj / (1.0 + odds_adj)

def _sigmoid_from_logit_with_temperature(
    logits: np.ndarray,
    temperature: float | None,
) -> np.ndarray:
    z = np.asarray(logits, dtype=float).reshape(-1)
    try:
        t = float(temperature)
    except Exception:
        t = 1.0
    if (not math.isfinite(t)) or t <= 1e-6:
        t = 1.0
    if abs(t - 1.0) > 1e-9:
        z = z / t
    return _sigmoid_stable(z)

def _score_from_stat_with_calib(
    stats,
    calibration: Optional[dict],
    kind: str,
    z0: float = 1.0,
    z1: float = 5.0,
):
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
            if (not math.isfinite(z_sigma)) or z_sigma <= 1e-12:
                z_sigma = 1.0
            oriented_for_logit = (oriented_for_logit - z_mu) / z_sigma
            z = w * oriented_for_logit + b0
            p = _sigmoid_from_logit_with_temperature(z, calib.get("temperature", 1.0))
            return _apply_prior_correction_probs(p, calib.get("prior_correction"))
        except Exception:
            pass

    p_ecdf = _apply_ecdf_risk_mapping(oriented, calib.get("ecdf_risk"))
    if p_ecdf is not None:
        p = _apply_temperature_scaling(p_ecdf, calib.get("temperature", 1.0))
        return _apply_prior_correction_probs(p, calib.get("prior_correction"))

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

    if (not math.isfinite(sigma)) or sigma <= 1e-9:
        sigma = 1.0

    z = (oriented - mu) / sigma
    p = _sigmoid_from_logit_with_temperature(z, calib.get("temperature", 1.0))
    return _apply_prior_correction_probs(p, calib.get("prior_correction"))

def _calibrated_anomaly_score(kind: str, decision_value: float) -> float:
    try:
        kind_norm = _normalize_anomaly_kind(kind)
        calib = ANOM_CALIB.get(kind_norm) or {}
        scores = _score_from_stat_with_calib(
            [decision_value],
            calib,
            kind_norm,
        )
        return _clip01(float(scores[0]))
    except Exception:
        log.exception("calibrated_anomaly_score_failed kind=%s", kind)
        return 0.5

def _feature_sig() -> str:
    order_raw = os.getenv("FEATURE_ORDER", "")
    s = ",".join([x.strip() for x in order_raw.split(",") if x.strip()]).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:8] if s else ""

def _load_recommended_threshold() -> Optional[float]:
    """
    Return recommended threshold for the aggregated ML score.
    Priority:
      1) ensemble threshold from ml_ensemble.meta.json
      2) single-model threshold when only one model threshold exists
      3) None
    """
    if isinstance(ENSEMBLE_RECOMMENDED_THRESHOLD, (int, float)):
        return float(ENSEMBLE_RECOMMENDED_THRESHOLD)
    if len(MODEL_RECOMMENDED_THRESHOLDS) == 1:
        return float(next(iter(MODEL_RECOMMENDED_THRESHOLDS.values())))
    return None


def _effective_threshold_with_source() -> tuple[float, str]:
    """
    Threshold for ml_score decision.
    ENV ML_THRESHOLD is treated as explicit alert override.
    """
    if ML_THRESHOLD_RAW:
        try:
            return float(ML_THRESHOLD_RAW), "env:ML_THRESHOLD"
        except Exception:
            log.warning('bad_env_threshold value="%s"', ML_THRESHOLD_RAW)
    rt = _load_recommended_threshold()
    if isinstance(rt, (int, float)):
        if isinstance(ENSEMBLE_RECOMMENDED_THRESHOLD, (int, float)):
            return float(rt), "meta:ml_ensemble.recommended_threshold"
        if len(MODEL_RECOMMENDED_THRESHOLDS) == 1:
            only_kind = next(iter(MODEL_RECOMMENDED_THRESHOLDS.keys()))
            return float(rt), f"meta:{only_kind}.recommended_threshold(single-model)"
    return 0.60, "default:0.60"


def _effective_threshold() -> float:
    return float(_effective_threshold_with_source()[0])


def _effective_aux_novelty_threshold() -> float:
    if ML_AUX_NOVELTY_THRESHOLD_RAW:
        try:
            return float(ML_AUX_NOVELTY_THRESHOLD_RAW)
        except Exception:
            log.warning('bad_env_aux_novelty_threshold value="%s"', ML_AUX_NOVELTY_THRESHOLD_RAW)
    if isinstance(AUX_NOVELTY_RECOMMENDED_THRESHOLD, (int, float)):
        return float(AUX_NOVELTY_RECOMMENDED_THRESHOLD)
    return 0.8


def _per_model_threshold_diagnostics(per: List[Dict[str, float]]) -> tuple[Dict[str, float], Dict[str, bool], Optional[float]]:
    """
    Diagnostic only:
      - threshold per model (from model metadata)
      - hit per model (score >= threshold)
      - mean hit ratio across models with known thresholds
    """
    thresholds: Dict[str, float] = {}
    hits: Dict[str, bool] = {}
    for item in (per or []):
        kind = str(item.get("kind") or "").strip().lower()
        if not kind or kind == "fallback":
            continue
        thr = MODEL_RECOMMENDED_THRESHOLDS.get(kind)
        if not isinstance(thr, (int, float)):
            continue
        try:
            score = float(item.get("score", 0.0))
            thresholds[kind] = float(thr)
            hits[kind] = bool(score >= float(thr))
        except Exception:
            continue

    ratio: Optional[float] = None
    if hits:
        ratio = float(sum(1.0 if v else 0.0 for v in hits.values()) / len(hits))
    return thresholds, hits, ratio

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
      • if, ocsvm, ae, lstm_ae and supervised baselines (sup_*),
        when their model files are present,
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

        # 0) supervised numeric baselines: obtain raw attack probability first,
        # then pass through shared runtime calibration from *.meta.json.
        if kind in SUPERVISED_NUMERIC_KINDS:
            est = m.get("clf") if isinstance(m, dict) and ("clf" in m) else m
            x = np.array(vec, dtype=float).reshape(1, -1)
            if hasattr(est, "predict_proba"):
                raw = float(est.predict_proba(x)[0, 1])
                return _calibrated_anomaly_score(kind, raw)
            if hasattr(est, "decision_function"):
                raw = float(_sigmoid(float(est.decision_function(x)[0])))
                return _calibrated_anomaly_score(kind, raw)
            if hasattr(est, "predict"):
                raw = float(est.predict(x)[0])
                return _calibrated_anomaly_score(kind, raw)
            return None

        # 1) sklearn-like anomaly models (IF, OCSVM, custom wrappers)
        # Prefer score_samples for IF/OCSVM to keep scoring semantics aligned with retrain.
        if kind in ("if", "ocsvm") and hasattr(m, "score_samples"):
            raw = float(m.score_samples([vec])[0])
            return _calibrated_anomaly_score(kind, raw)
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

def _aggregate_scores(
    per: List[Dict[str, float]],
    method: Optional[str] = None,
    calibration: Optional[dict] = None,
) -> float:
    if not per:
        return 0.0
    vals = np.asarray([float(p["score"]) for p in per if "score" in p], dtype=float)
    if vals.size == 0:
        return 0.0
    if vals.size == 1:
        return _clip01(float(vals[0]))
    method_norm = (method or ML_ENSEMBLE_AGG or "mean_logit").strip().lower()
    if method_norm in ("max", "or", "max_per_model"):
        out = _clip01(float(np.max(vals)))
    elif method_norm in ("mean", "avg", "average"):
        out = _clip01(float(np.mean(vals)))
    else:
        vals = np.clip(vals, 1e-6, 1.0 - 1e-6)
        logits = np.log(vals / (1.0 - vals))
        v = float(np.mean(logits))
        out = _clip01(float(_sigmoid_stable(np.array([v], dtype=float))[0]))

    try:
        calib = calibration if isinstance(calibration, dict) else (ML_ENSEMBLE_CALIB or {})
        t = float((calib or {}).get("temperature", 1.0))
        if math.isfinite(t) and t > 1e-6 and abs(t - 1.0) > 1e-9:
            out = float(_apply_temperature_scaling(np.array([out], dtype=float), t)[0])
    except Exception:
        pass
    try:
        calib = calibration if isinstance(calibration, dict) else (ML_ENSEMBLE_CALIB or {})
        prior_cfg = (calib or {}).get("prior_correction")
        if isinstance(prior_cfg, dict) and bool(prior_cfg.get("enabled", False)):
            out = float(_apply_prior_correction_probs(np.array([out], dtype=float), prior_cfg)[0])
    except Exception:
        pass
    return _clip01(out)

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

    thr, thr_source = _effective_threshold_with_source()

    return {
        "models": models_status,
        "supported_kinds": list(MODEL_REGISTRY.keys()),
        "primary_core_kinds": list(ML_PRIMARY_CORE_KINDS),
        "aux_novelty_kinds": list(ML_AUX_NOVELTY_KINDS),
        "feature_order": feature_order_list,
        "feature_order_sig": _feature_sig(),
        "threshold": float(thr),
        "threshold_source": thr_source,
        "aux_novelty_threshold": float(_effective_aux_novelty_threshold()),
        "per_model_recommended_thresholds": MODEL_RECOMMENDED_THRESHOLDS,
        "ensemble_recommended_threshold": ENSEMBLE_RECOMMENDED_THRESHOLD,
        "aux_novelty_recommended_threshold": AUX_NOVELTY_RECOMMENDED_THRESHOLD,
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
        "aux_novelty_threshold": _effective_aux_novelty_threshold(),
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
      • aggregate per_model_scores as ml_score (method via ML_ENSEMBLE_AGG),
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
                ml_primary_score = ml_score
                ml_aux_score = None
                ml_aux_thr = _effective_aux_novelty_threshold()
                ml_aux_hit = False
                per = [{"kind": "fallback", "score": float(ml_score)}]
                per_model_scores = {"fallback": ml_score}
                primary_model_scores = {}
                aux_novelty_model_scores = {}
            else:
                primary_per = [p for p in per if str(p.get("kind", "")).lower() in ML_PRIMARY_CORE_KINDS]
                if not primary_per:
                    # Graceful fallback when supervised-core artifacts are not present.
                    primary_per = [p for p in per if str(p.get("kind", "")).lower() in SUPERVISED_NUMERIC_KINDS]
                if not primary_per:
                    primary_per = list(per)

                aux_per = [p for p in per if str(p.get("kind", "")).lower() in ML_AUX_NOVELTY_KINDS]

                ml_primary_score = _aggregate_scores(
                    primary_per,
                    method=ML_PRIMARY_ENSEMBLE_AGG,
                    calibration=ML_ENSEMBLE_CALIB,
                )
                ml_score = float(ml_primary_score)
                ml_aux_score = (
                    _aggregate_scores(
                        aux_per,
                        method=ML_AUX_NOVELTY_AGG,
                        calibration=ML_AUX_NOVELTY_CALIB,
                    ) if aux_per else None
                )
                ml_aux_thr = _effective_aux_novelty_threshold()
                ml_aux_hit = bool((ml_aux_score is not None) and (float(ml_aux_score) >= float(ml_aux_thr)))
                per_model_scores = {p["kind"]: p["score"] for p in per}
                primary_model_scores = {p["kind"]: p["score"] for p in primary_per}
                aux_novelty_model_scores = {p["kind"]: p["score"] for p in aux_per}

            thr = _effective_threshold()
            hit = bool(ml_score >= thr)
            per_thr, per_hits, per_ratio = _per_model_threshold_diagnostics(per)

            SCORE_REQS.labels("ok").inc()
            try:
                log.info(
                    'ml_score hash_id="%s" score=%.4f primary=%.4f aux=%s aux_thr=%.4f thr=%.4f hit=%s per_models=%s',
                    inp.hash_id,
                    ml_score,
                    float(ml_primary_score),
                    f"{float(ml_aux_score):.4f}" if ml_aux_score is not None else "null",
                    float(ml_aux_thr),
                    thr,
                    str(hit),
                    json.dumps(per_model_scores)
                )
            except Exception:
                pass

            return ScoreOut(
                hash_id=inp.hash_id,
                ml_score=float(ml_score),
                ml_primary_score=float(ml_primary_score),
                ml_aux_novelty_score=(float(ml_aux_score) if ml_aux_score is not None else None),
                ml_aux_novelty_threshold=float(ml_aux_thr),
                ml_aux_novelty_hit=bool(ml_aux_hit),
                threshold=float(thr),
                threshold_hit=bool(hit),
                per_model=[{"kind": str(p["kind"]), "score": float(p["score"])} for p in per],
                per_model_scores={k: float(v) for k, v in per_model_scores.items()},
                primary_model_scores={k: float(v) for k, v in primary_model_scores.items()},
                aux_novelty_model_scores={k: float(v) for k, v in aux_novelty_model_scores.items()},
                per_model_thresholds={k: float(v) for k, v in per_thr.items()},
                per_model_threshold_hits={k: bool(v) for k, v in per_hits.items()},
                per_model_hit_ratio=(float(per_ratio) if per_ratio is not None else None),
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
                ml_primary_score=float(fb),
                ml_aux_novelty_score=None,
                ml_aux_novelty_threshold=float(_effective_aux_novelty_threshold()),
                ml_aux_novelty_hit=False,
                threshold=float(thr),
                threshold_hit=bool(fb >= thr),
                per_model=[{"kind": "fallback", "score": float(fb)}],
                per_model_scores={"fallback": float(fb)},
                primary_model_scores={},
                aux_novelty_model_scores={},
                per_model_thresholds={},
                per_model_threshold_hits={},
                per_model_hit_ratio=None,
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
                primary_per = []
                aux_per = []
                ml_primary_score = float(fb)
                ml_aux_score = None
            else:
                primary_per = [p for p in per if str(p.get("kind", "")).lower() in ML_PRIMARY_CORE_KINDS]
                if not primary_per:
                    primary_per = [p for p in per if str(p.get("kind", "")).lower() in SUPERVISED_NUMERIC_KINDS]
                if not primary_per:
                    primary_per = list(per)
                aux_per = [p for p in per if str(p.get("kind", "")).lower() in ML_AUX_NOVELTY_KINDS]
                ml_primary_score = _aggregate_scores(
                    primary_per,
                    method=ML_PRIMARY_ENSEMBLE_AGG,
                    calibration=ML_ENSEMBLE_CALIB,
                )
                ml_aux_score = (
                    _aggregate_scores(
                        aux_per,
                        method=ML_AUX_NOVELTY_AGG,
                        calibration=ML_AUX_NOVELTY_CALIB,
                    ) if aux_per else None
                )

            ml_score = float(ml_primary_score)
            ml_aux_thr = _effective_aux_novelty_threshold()
            ml_aux_hit = bool((ml_aux_score is not None) and (float(ml_aux_score) >= float(ml_aux_thr)))
            per_thr, per_hits, per_ratio = _per_model_threshold_diagnostics(per)
            out.append({
                "hash_id": item.hash_id,
                "per_model": per,
                "ml_score": ml_score,
                "ml_primary_score": ml_primary_score,
                "ml_aux_novelty_score": ml_aux_score,
                "ml_aux_novelty_threshold": ml_aux_thr,
                "ml_aux_novelty_hit": ml_aux_hit,
                "threshold": thr,
                "threshold_hit": (ml_score >= thr),
                "primary_model_scores": {p["kind"]: p["score"] for p in primary_per},
                "aux_novelty_model_scores": {p["kind"]: p["score"] for p in aux_per},
                "per_model_thresholds": per_thr,
                "per_model_threshold_hits": per_hits,
                "per_model_hit_ratio": per_ratio,
                "used_models": [p["kind"] for p in per if p.get("kind") != "fallback"],
            })
        except Exception as e:
            SCORE_REQS.labels("error").inc()
            fb = _fallback_score(item.numeric_feats or {})
            out.append({
                "hash_id": item.hash_id,
                "per_model": [{"kind": "fallback", "score": fb}],
                "ml_score": fb,
                "ml_primary_score": fb,
                "ml_aux_novelty_score": None,
                "ml_aux_novelty_threshold": _effective_aux_novelty_threshold(),
                "ml_aux_novelty_hit": False,
                "threshold": thr,
                "threshold_hit": (fb >= thr),
                "primary_model_scores": {},
                "aux_novelty_model_scores": {},
                "per_model_thresholds": {},
                "per_model_threshold_hits": {},
                "per_model_hit_ratio": None,
                "used_models": [],
                "meta": {"error": "exception_fallback"},
            })
    return JSONResponse({"results": out})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
