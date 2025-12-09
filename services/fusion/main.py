import os
import time
import hashlib
import json, logging
import statistics
import redis
from typing import Annotated, List, Optional, Dict, Any, Tuple

import uvicorn, asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response, Header, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

FUSION_ALPHA = float(os.getenv("FUSION_ALPHA", os.getenv("ALPHA", "0.6")))
FUSION_BETA  = float(os.getenv("FUSION_BETA",  os.getenv("BETA",  "0.4")))
FUSION_MODE  = os.getenv("FUSION_MODE", os.getenv("MODE", "weighted")).lower()
FUSION_THR   = float(os.getenv("FUSION_FINAL_THR", os.getenv("THETA", "0.7")))
FUSION_MAX_SCORES = int(os.getenv("FUSION_MAX_SCORES", "16"))

RISK_MODE = os.getenv("FUSION_RISK_MODE", "score_only").lower()
IMPACT_POLICY_PATH = os.getenv("IMPACT_POLICY_FILE", "/app/policies/impact_policy.json")
# Risk level "0.3,0.6,0.85,0.97" will set low=0.3, medium=0.6, high=0.85, critical=0.97
def _parse_levels(env: str) -> Tuple[float,float,float,float]:
    raw = os.getenv(env, "0.3,0.6,0.85,0.97").split(",")
    vals = [float(x.strip()) for x in raw if x.strip()]
    while len(vals) < 4:
        vals.append(vals[-1] if vals else 0.9)
    return tuple(vals[:4])

# Strong NLP topic (for overrides)
TOPIC_STRONG = float(os.getenv("FUSION_TOPIC_STRONG", "0.8"))

# Topics that escalate risk
TOPIC_ESCALATION = os.getenv("FUSION_TOPIC_ESCALATION", "list").lower()

# Hysteresis
HYST_ENABLED    = os.getenv("HYST_ENABLED", "1").lower() not in ("0", "false", "no")
HYST_MARGIN     = float(os.getenv("HYST_MARGIN", "0.07"))       # downgrade
HYST_UP_MARGIN  = float(os.getenv("HYST_UP_MARGIN", "0.00"))     # upgrade
HYST_TTL        = float(os.getenv("HYST_TTL", "600"))            # seconds
HYST_KEY_FIELD  = os.getenv("HYST_KEY_FIELD", "hkey")
ORCH_TOKEN      = os.getenv("ORCH_TOKEN", "").strip()

# FastAPI
LOG = logging.getLogger("fusion")
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))

# Metrics
FUSION_REQ = Counter('fusion_requests_total', 'Total /fuse requests', ['status'])
FUSION_LAT = Histogram('fusion_latency_seconds', 'Latency of /fuse handler',
                       buckets=[0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5])
FUSION_FIN = Histogram('fusion_final_score', 'Final score distribution',
                       buckets=[0,0.2,0.4,0.6,0.7,0.8,0.9,0.95,0.98,1.0])
IMPACT_HIST = Histogram('fusion_impact_score', 'Impact score distribution',
                        buckets=[0,0.1,0.2,0.3,0.4,0.55,0.7,0.9,1.0])
FUSION_DEC = Counter('fusion_decisions_total', 'Decisions by level/anomaly', ['level','anomaly'])
FUSION_OVR = Counter('fusion_topic_overrides_total', 'Topic overrides applied', ['level','topic'])
FUSION_HLD = Counter('fusion_hysteresis_holds_total', 'Hysteresis holds by level', ['held_level'])
FUSION_STATE = Gauge('fusion_hysteresis_state_size', 'Items in hysteresis state')

Score = Annotated[float, Field(ge=0.0, le=1.0)]
ScoreList = Annotated[List[Score], Field(min_length=0)]

# Pydantic models
class PolicyIn(BaseModel):
    alpha: float = Field(default=FUSION_ALPHA, description="Weight of ML")
    beta:  float = Field(default=FUSION_BETA,  description="Weight of NLP")
    mode:  str   = Field(default=FUSION_MODE,  description="weighted|max|mean|weighted_adaptive")
    threshold_final: float = Field(default=FUSION_THR, description="Final decision threshold")

class FusionIn(BaseModel):
    hash_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    ml_scores: Optional[ScoreList] = None
    nlp_scores: Optional[ScoreList] = None
    ml_score: Optional[Score] = None
    nlp_score: Optional[Score] = None
    policy: PolicyIn = Field(default_factory=PolicyIn)
    
class FusionOut(BaseModel):
    hash_id: str
    final_score: float
    final_threshold: float
    is_anomaly: bool
    risk_level: str
    agg_ml: Optional[float] = None
    agg_nlp: Optional[float] = None
    mode: str = FUSION_MODE
    used_fallback: bool = False
    reasons: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)

# Internal state for hysteresis
_HYST_STATE: Dict[str, Dict[str, Any]] = {}

# Redis URL
REDIS_URL = os.getenv("REDIS_URL", "").strip()
if REDIS_URL:
    try:
        _redis =  redis.Redis.from_url(REDIS_URL, decode_responses=True)
        LOG.info("Connected to Redis at %s", REDIS_URL)
        _redis.ping()
    except Exception:
        _redis = None
# Parsing topics from ENV
def _parse_topics(env: str, default: set[str]) -> set[str]:
    raw = os.getenv(env, "")
    return {t.strip() for t in raw.split(",") if t.strip()} or set(default)

# OWASP Top 10 2021–aligned topic escalation
CRITICAL_TOPICS = _parse_topics(
    "FUSION_CRITICAL_TOPICS",
    {
        # A03: Injection
        "SQL Injection",
        "Command Injection",

        # A10: SSRF (metadata endpoints)
        "Cloud Metadata SSRF",
    },
)

HIGH_TOPICS = _parse_topics(
    "FUSION_HIGH_TOPICS",
    {
        # A03: Injection
        "NoSQL Injection",
        "LDAP Injection",
        "Cross-Site Scripting",
        "Server-Side Request Forgery",

        # A01: Broken Access Control
        "Insecure Direct Object Reference",
        "Business Logic Abuse",

        # A02: Cryptographic / Sensitive Data failures
        "API Key Exposure",
        "Sensitive Data Exposure",

        # A07: Identification & Authentication Failures
        "Authentication Bruteforce",
        "Credential Stuffing",

        # A05: Security Misconfiguration
        "Security Misconfiguration",
        "WebSocket Abuse",

        # A06: Vulnerable and Outdated Components
        "Vulnerable and Outdated Components",

        # A08: Software and Data Integrity Failures
        "Software and Data Integrity Failures",

        # A09: Security Logging and Monitoring Failures
        "Security Logging and Monitoring Failures",

        "Denial of Service",
    },
)
# Loading impact policy
_IMPACT = {"weights": {}, "caps": {"topic_bump": {}, "max_impact": 1.0}, "bins":{"impact":[0.25,0.55]},
           "matrix":[["low","medium","high"],["medium","high","high"],["high","high","critical"],["high","critical","critical"]]}
try:
    with open(IMPACT_POLICY_PATH, "r", encoding="utf-8") as f:
        _IMPACT.update(json.load(f))
    LOG.info('impact_policy_loaded path="%s"', IMPACT_POLICY_PATH)
except FileNotFoundError:
    LOG.warning('impact_policy_missing path="%s" using_defaults=1', IMPACT_POLICY_PATH)
except Exception as e:
    LOG.warning('impact_policy_failed path="%s" err=%s using_defaults=1', IMPACT_POLICY_PATH, type(e).__name__)

def _conf_bin(s: float) -> int:
    s = max(0.0, min(1.0, float(s)))
    if s >= L_HIGH: return 3
    if s >= L_MED:  return 2
    if s >= L_LOW:  return 1
    return 0

def _impact_score(ctx: dict, topics: list[dict]) -> float:
    w = _IMPACT.get("weights", {})
    sc = 0.0
    # expected keys in context
    env        = str(ctx.get("env","dev"))
    endpoint_k = str(ctx.get("endpoint_class","")).lower()
    data_cls   = str(ctx.get("data_class","none")).lower()
    auth_st    = str(ctx.get("auth_state","anonymous")).lower()
    exposure   = str(ctx.get("exposure","public")).lower()
    blast      = str(ctx.get("blast","low")).lower()

    sc += w.get("env",{}).get(env, 0.0)
    sc += w.get("endpoint",{}).get(endpoint_k, 0.0)
    sc += w.get("data_class",{}).get(data_cls, 0.0)
    sc += w.get("auth_state",{}).get(auth_st, 0.0)
    sc += w.get("exposure",{}).get(exposure, 0.0)
    sc += w.get("blast",{}).get(blast, 0.0)

    # small bump for some "rotten" topics
    if topics:
        top_name = max(topics, key=lambda t: float(t.get("score",0.0))).get("topic")
        sc += _IMPACT.get("caps",{}).get("topic_bump",{}).get(str(top_name), 0.0)

    return min(float(_IMPACT.get("caps",{}).get("max_impact", 1.0)), max(0.0, sc))

def _impact_bin(v: float) -> int:
    b = _IMPACT.get("bins",{}).get("impact",[0.25,0.55])
    if v >= b[1]: return 2
    if v >= b[0]: return 1
    return 0

def risk_level_from_score_only(s: float) -> str:
    if s >= L_HIGH: return "critical"
    if s >= L_MED:  return "high"
    if s >= L_LOW:  return "medium"
    if s > 0.0:     return "low"
    return "normal"

def risk_level_from_matrix(s: float, ctx: dict, topics: list[dict]) -> str:
    ci = _conf_bin(s)
    ii = _impact_bin(_impact_score(ctx, topics))
    mx = _IMPACT.get("matrix") or []
    try:
        return str(mx[ci][ii])
    except Exception:
        return risk_level_from_score_only(s)

# Helpers
def _state_get(hkey: str) -> Optional[Dict[str, Any]]:
    if _redis:
        raw = _redis.get(f"fusion:{hkey}")
        return json.loads(raw) if raw else None
    return _HYST_STATE.get(hkey)

def _state_set(hkey: str, obj: Dict[str, Any]):
    if _redis:
        _redis.setex(f"fusion:{hkey}", int(HYST_TTL), json.dumps(obj))
    else:
        _HYST_STATE[hkey] = obj

def _aggregate(scores: List[float], how: str) -> float:
    if not scores:
        return 0.0
    how = (how or "max").lower()
    if how == "mean":
        return float(statistics.fmean(scores))
    return max(scores)

def _clean_scores(scores: Optional[List[float]]) -> List[float]:
    if not scores:
        return []
    out = []
    for v in scores:
        try:
            x = float(v)
            if x != x:  # NaN
                continue
            if x < 0.0: x = 0.0
            if x > 1.0: x = 1.0
            out.append(x)
        except Exception:
            continue
    return out

def _normalize_weights(a: float, b: float) -> tuple[float, float]:
    s = a + b
    if s <= 0:
        return 0.5, 0.5
    return a / s, b / s

def _ensure_hash_id(inp: FusionIn) -> str:
    if inp.hash_id:
        return inp.hash_id
    seed = {
        "ctx": inp.context or {},
        "ml": (inp.ml_scores or []) + ([inp.ml_score] if inp.ml_score is not None else []),
        "nlp": (inp.nlp_scores or []) + ([inp.nlp_score] if inp.nlp_score is not None else []),
        "policy": {
            "alpha": inp.policy.alpha,
            "beta": inp.policy.beta,
            "mode": inp.policy.mode,
            "thr": inp.policy.threshold_final,
        },
    }
    blob = json.dumps(seed, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def _gap_confidence(vals: List[float]) -> float:
    if not vals:
        return 0.0
    v = sorted(vals, reverse=True)
    top = v[0]
    second = v[1] if len(v) > 1 else 0.0
    gap = max(top - second, 0.0)
    return float(min(max(gap * 2.0, 0.0), 1.0))

def _risk_level(score: float) -> str:
    if score >= LVL_CRIT: return "critical"
    if score >= LVL_HIGH: return "high"
    if score >= LVL_MED:  return "medium"
    if score >= LVL_LOW:  return "low"
    return "normal"

def _topic_overrides(topics: List[Dict[str, float]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if TOPIC_ESCALATION == "disabled":
        return None, None, None

    best = None
    best_topic = None
    for t in topics or []:
        name = t.get("topic")
        sc = float(t.get("score", 0.0))
        if not name:
            continue
        if sc < TOPIC_STRONG:
            continue
        if TOPIC_ESCALATION == "all_critical":
            return ("critical", f"topic:{name}({sc:.3f})>=strong(all_critical)", name)
        if TOPIC_ESCALATION == "all_high" and not best:
            best, best_topic = ("high", f"topic:{name}({sc:.3f})>=strong(all_high)"), name
        if TOPIC_ESCALATION == "list":
            if name in CRITICAL_TOPICS:
                return ("critical", f"topic:{name}({sc:.3f})>=strong(list)", name)
            if name in HIGH_TOPICS and not best:
                best, best_topic = ("high", f"topic:{name}({sc:.3f})>=strong(list)"), name
    if best:
        return best[0], best[1], best_topic
    return None, None, None

def _apply_adaptive(alpha: float, beta: float, ml_scores: List[float], nlp_scores: List[float]) -> Tuple[float,float]:
    a, b = _normalize_weights(alpha, beta)
    cm = _gap_confidence(ml_scores)
    cn = _gap_confidence(nlp_scores)
    if cm == 0.0 and cn == 0.0:
        return a, b
    sm = cm + cn
    wm = cm / sm if sm > 0 else 0.5
    wn = cn / sm if sm > 0 else 0.5
    a2 = (a + wm) / 2.0
    b2 = (b + wn) / 2.0
    return _normalize_weights(a2, b2)

def _reasons(final: float, thr: float, mode: str, agg_ml: float, agg_nlp: float, alpha: float, beta: float, overrides: Optional[str], hyst_hold: Optional[str]) -> List[str]:
    rs = [
        f"mode={mode}",
        f"alpha={alpha:.3f},beta={beta:.3f}",
        f"agg_ml={agg_ml:.3f},agg_nlp={agg_nlp:.3f}",
        f"final={final:.3f} vs thr={thr:.3f}",
    ]
    if overrides: rs.append(f"override={overrides}")
    if hyst_hold: rs.append(f"hysteresis_hold={hyst_hold}")
    return rs

# Hysteresis
LVL_LOW, LVL_MED, LVL_HIGH, LVL_CRIT = _parse_levels("FUSION_LEVELS")

L_LOW, L_MED, L_HIGH = LVL_LOW, LVL_MED, LVL_HIGH

_LEVELS = ["normal", "low", "medium", "high", "critical"]
_BOUNDARY = {
    "normal": 0.0,
    "low":    LVL_LOW,
    "medium": LVL_MED,
    "high":   LVL_HIGH,
    "critical": LVL_CRIT,
}

def _idx(level: str) -> int:
    try: return _LEVELS.index(level)
    except ValueError: return 0

def _cleanup_state(now: float):
    if not _HYST_STATE: return
    for k, v in list(_HYST_STATE.items()):
        if now - v.get("ts", 0.0) > HYST_TTL:
            _HYST_STATE.pop(k, None)
    try:
        FUSION_STATE.set(len(_HYST_STATE))
    except Exception:
        pass


def _hkey(ctx: Dict[str, Any], default: str) -> str:
    if not isinstance(ctx, dict): return default
    return (str(ctx.get(HYST_KEY_FIELD))
            or str(ctx.get("session") or "")
            or f"{ctx.get('endpoint','')}|{ctx.get('method','')}"
            or default)

def _apply_hysteresis(prev_level: str, proposed_level: str, final_score: float) -> Tuple[str, Optional[str], bool]:
    if not HYST_ENABLED:
        return proposed_level, None, False

    pi = _idx(prev_level)
    ni = _idx(proposed_level)

    if ni > pi:
        # upper boundary of the proposed level
        need = _BOUNDARY[proposed_level] + HYST_UP_MARGIN
        if final_score >= need:
            return proposed_level, None, False
        else:
            return prev_level, f"upgrade blocked (need≥{need:.3f})", True

    if ni < pi:
        # lower boundary of the current level
        need_low = _BOUNDARY[prev_level] - HYST_MARGIN
        if final_score <= need_low:
            return proposed_level, None, False
        else:
            return prev_level, f"downgrade blocked (need≤{need_low:.3f})", True

    # Level does not change
    return proposed_level, None, False

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    if HYST_ENABLED:
        async def _loop():
            while True:
                await asyncio.sleep(10)
                _cleanup_state(time.time())
        task = asyncio.create_task(_loop(), name="fusion-hyst-keeper")
    yield
    if task:
        task.cancel()

app = FastAPI(title="fusion-decision", version="1.3.0", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)

# Service endpoints
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    return {
        "status": "ok",
        "defaults": {
            "alpha": FUSION_ALPHA, "beta": FUSION_BETA,
            "mode": FUSION_MODE, "threshold": FUSION_THR,
            "levels": {"low": LVL_LOW, "medium": LVL_MED, "high": LVL_HIGH, "critical": LVL_CRIT},
            "topic_strong": TOPIC_STRONG,
            "hysteresis": {
                "enabled": HYST_ENABLED,
                "margin": HYST_MARGIN,
                "up_margin": HYST_UP_MARGIN,
                "key_field": HYST_KEY_FIELD,
                "ttl": HYST_TTL,
            }
        }
    }

@app.get("/policy")
def policy():
    return {
        "alpha": FUSION_ALPHA, "beta": FUSION_BETA,
        "mode": FUSION_MODE, "threshold": FUSION_THR,
        "levels": {"low": LVL_LOW, "medium": LVL_MED, "high": LVL_HIGH, "critical": LVL_CRIT},
        "topic_strong": TOPIC_STRONG,
        "critical_topics": sorted(CRITICAL_TOPICS),
        "high_topics": sorted(HIGH_TOPICS),
        "hysteresis": {
            "enabled": HYST_ENABLED,
            "margin": HYST_MARGIN,
            "up_margin": HYST_UP_MARGIN,
            "key_field": HYST_KEY_FIELD,
            "ttl": HYST_TTL,
        },
        "version": app.version,
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

_fuse_sem = asyncio.Semaphore(int(os.getenv("FUSION_MAX_CONCURRENCY", "64")))

@app.post("/fuse", response_model=FusionOut)
async def fuse(
    inp: FusionIn,
    request: Request,
    x_orch_token: str | None = Header(default=None, alias="X-Orch-Token"),
):
    async with _fuse_sem:
        if ORCH_TOKEN and (x_orch_token or "") != ORCH_TOKEN:
            try:
                LOG.warning(
                    'unauthorized fuse call path="/fuse" hdr_present=%s peer="%s"',
                    bool(x_orch_token), 
                    getattr(request.client, "host", "?")
                )
                FUSION_REQ.labels(status="unauth").inc()
            finally:
                raise HTTPException(status_code=401, detail="unauthorized")
        t0 = time.perf_counter()
        try:
            hash_id = _ensure_hash_id(inp)

            ml_scores = _clean_scores(list(inp.ml_scores or [])[:FUSION_MAX_SCORES])
            nlp_scores = _clean_scores(list(inp.nlp_scores or [])[:FUSION_MAX_SCORES])
            if inp.ml_score is not None:  ml_scores.append(float(inp.ml_score))
            if inp.nlp_score is not None: nlp_scores.append(float(inp.nlp_score))

            if not ml_scores and not nlp_scores:
                try:
                    LOG.warning(
                        "fusion_no_scores hash_id=%s ctx_keys=%s",
                        hash_id,
                        list((inp.context or {}).keys()),
                    )
                except Exception:
                    pass

                FUSION_REQ.labels(status="no_scores").inc()
                return FusionOut(
                    hash_id=hash_id,
                    final_score=0.0,
                    final_threshold=float(inp.policy.threshold_final),
                    is_anomaly=False,
                    risk_level="normal",
                    agg_ml=0.0,
                    agg_nlp=0.0,
                    mode=(inp.policy.mode or FUSION_MODE),
                    used_fallback=False,
                    reasons=["no_ml_scores", "no_nlp_scores"],
                    context=inp.context or {},
                )

            # Aggregation
            agg_ml  = _aggregate(ml_scores,  "max")
            agg_nlp = _aggregate(nlp_scores, "max")

            # Policy
            mode  = (inp.policy.mode or FUSION_MODE).lower()
            alpha = float(inp.policy.alpha)
            beta  = float(inp.policy.beta)
            thr   = float(inp.policy.threshold_final)

            used_fallback = False

            # Final score
            if mode == "weighted":
                a, b = _normalize_weights(alpha, beta)
                final = a * agg_ml + b * agg_nlp
            elif mode == "weighted_adaptive":
                a, b = _apply_adaptive(alpha, beta, ml_scores, nlp_scores)
                final = a * agg_ml + b * agg_nlp
                alpha, beta = a, b
            elif mode == "max":
                final = max(agg_ml, agg_nlp)
            elif mode == "mean":
                vals = [v for v in (agg_ml, agg_nlp) if v is not None]
                final = float(statistics.fmean(vals)) if vals else 0.0
            else:
                a, b = _normalize_weights(FUSION_ALPHA, FUSION_BETA)
                final = a * agg_ml + b * agg_nlp
                mode = "weighted"
                alpha, beta = a, b
                used_fallback = True

            # 1) Extract topics from context
            ctx = inp.context if isinstance(inp.context, dict) else {}
            topics_any = ctx.get("topics") or ctx.get("nlp_topics") or []
            topics = topics_any if isinstance(topics_any, list) else []

            # 2) Base level depending on risk mode
            if RISK_MODE == "score_x_impact":
                base_level = risk_level_from_matrix(final, ctx, topics)
                try:
                    imp_val = _impact_score(ctx, topics)
                    IMPACT_HIST.observe(float(imp_val))
                except Exception:
                    pass
            else:
                base_level = _risk_level(final)

            # 3) Overrides by topics
            topic_thr = float(os.getenv("FUSION_TOPIC_STRONG", str(TOPIC_STRONG)))
            top_hit = max(
                (float(t.get("score", 0.0)) for t in topics
                if t.get("topic") in (CRITICAL_TOPICS | HIGH_TOPICS)),
                default=0.0
            )
            override = "topic_escalation" if top_hit >= topic_thr else None

            min_level, override_reason, override_topic = (None, None, None)
            if topics:
                min_level, override_reason, override_topic = _topic_overrides(topics)

            # 4) Final level = base_level with overrides
            risk_level = base_level
            order = ["normal", "low", "medium", "high", "critical"]
            if min_level and order.index(min_level) > order.index(base_level):
                risk_level = min_level
                if override_topic:
                    FUSION_OVR.labels(level=min_level, topic=override_topic).inc()

            # Hysteresis
            hkey = _hkey(inp.context, hash_id)
            prev = _state_get(hkey)
            hyst_reason = None
            if prev:
                adjusted, reason, held = _apply_hysteresis(prev.get("level", "normal"), risk_level, final)
                if held:
                    FUSION_HLD.labels(held_level=prev.get("level", "normal")).inc()
                risk_level, hyst_reason = adjusted, reason

            now = time.time()
            _state_set(hkey, {"level": risk_level, "ts": now, "score": final})
            # Cleanup state periodically
            if int(now) % 17 == 0:
                _cleanup_state(now)

            is_anom = bool(final >= thr or override is not None)
            reasons = _reasons(
                final, thr, mode, agg_ml, agg_nlp, alpha, beta,
                (override or override_reason), hyst_reason
            )
            if is_anom:
                reasons.append("anomaly_reason=" + ("threshold" if final >= thr else "topic_escalation"))
            # Log
            try:
                LOG.info(json.dumps({
                    "event": "fusion_decision",
                    "hash_id": hash_id,
                    "agg_ml": agg_ml,
                    "agg_nlp": agg_nlp,
                    "final": final,
                    "thr": thr,
                    "risk_level": risk_level,
                    "is_anomaly": is_anom,
                    "mode": mode,
                    "reasons": reasons,
                    "hkey": hkey,
                }, ensure_ascii=False))
            except Exception:
                pass

            # Metrics
            FUSION_REQ.labels(status="ok").inc()
            FUSION_FIN.observe(final)
            FUSION_DEC.labels(level=risk_level, anomaly=str(is_anom).lower()).inc()

            return FusionOut(
                hash_id=hash_id,
                final_score=float(final),
                final_threshold=thr,
                is_anomaly=is_anom,
                risk_level=risk_level,
                agg_ml=agg_ml,
                agg_nlp=agg_nlp,
                mode=mode,
                used_fallback=used_fallback,
                reasons=reasons,
                context=inp.context or {},
            )
        except Exception:
            FUSION_REQ.labels(status="error").inc()
            raise
        finally:
            FUSION_LAT.observe(max(time.perf_counter() - t0, 0.0))


if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )