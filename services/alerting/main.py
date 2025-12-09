import os
import json
import time
import ssl
import smtplib
import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

import uvicorn
import httpx
import redis
from email.mime.text import MIMEText

from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from contextlib import asynccontextmanager

SERVICE_NAME = "alerting"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","service":"'+SERVICE_NAME+'","msg":"%(message)s"}',
)
LOG = logging.getLogger(SERVICE_NAME)

# Orchestrator auth token
ORCH_TOKEN = os.getenv("ORCH_TOKEN", "").strip()

ALERT_SINKS = [s.strip() for s in os.getenv("ALERT_SINKS", "local,opensearch").split(",") if s.strip()]

# Recommended actions
ACTIONS_FILE = os.getenv("ALERT_ACTIONS_FILE", "/app/policies/recommended_actions.json")
_ACTIONS: dict = {
    "default": [
        "Review recent requests from same IP/user",
        "Apply temporary rate limiting or block",
        "Validate inputs and review server logs around timestamp",
    ],
    "topics": {}
}

# Alert gating (by risk level)
ALERT_MIN_RISK = os.getenv("ALERT_MIN_RISK", "medium").lower()
_SEV_NAME_TO_NUM = {"normal":0, "low":1, "medium":2, "high":3, "critical":4}
_MIN_SEV_NUM = _SEV_NAME_TO_NUM.get(ALERT_MIN_RISK, 2)  # default: medium

# Slack (optional)
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()

# Email (optional)
EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
EMAIL_USER      = os.getenv("EMAIL_USER", "")
EMAIL_PASS      = os.getenv("EMAIL_PASS", "")
EMAIL_TO        = os.getenv("EMAIL_TO", "")

# OpenSearch
OS_URL  = os.getenv("OPENSEARCH_URL", "http://opensearch:9200").rstrip("/")
OS_USER = os.getenv("OPENSEARCH_USER", "admin")
OS_PASS = os.getenv("OPENSEARCH_PASS", "admin")
OS_INCIDENT_INDEX_PREFIX = os.getenv("OPENSEARCH_INCIDENT_INDEX_PREFIX", "incidents")

# ServiceNow (optional)
SN_URL  = os.getenv("SERVICENOW_URL", "").rstrip("/")
SN_USER = os.getenv("SERVICENOW_USER", "")
SN_PASS = os.getenv("SERVICENOW_PASS", "")

# OpenSearch config
OS_URL   = os.getenv("OPENSEARCH_URL", "").rstrip("/")
OS_USER  = os.getenv("OPENSEARCH_USER", "")
OS_PASS  = os.getenv("OPENSEARCH_PASS", "")
OS_INDEX_PREFIX = os.getenv("OPENSEARCH_INDEX_PREFIX", "alerting")
OS_TIMEOUT = float(os.getenv("ALERT_TIMEOUT", "3.0"))
OS_RETRIES = int(os.getenv("ALERT_RETRIES", "3"))
OS_VERIFY  = os.getenv("OPENSEARCH_VERIFY", "1").lower() not in ("0","false","no")
OS_CA_BUNDLE = os.getenv("OPENSEARCH_CA_BUNDLE", "").strip() or None
OS_REFRESH = os.getenv("OPENSEARCH_REFRESH", "").strip()  # "", "wait_for", "true", "false"

# Redis for idempotency + DLQ
REDIS_URL = os.getenv("REDIS_URL", "").strip()
DEDUP_TTL = int(os.getenv("ALERT_DEDUP_TTL", "3600"))
DLQ_KEY   = os.getenv("ALERT_DLQ_KEY", "alerting:dlq")
DLQ_REPLAY_BATCH = int(os.getenv("ALERT_DLQ_REPLAY_BATCH", "100"))

# Behavior flags
DRY_RUN = os.getenv("ALERT_DRY_RUN", "0").lower() in ("1","true","yes")
STIX_ENABLED = os.getenv("STIX_ENABLED", "0").lower() in ("1","true","yes")

# Metrics
REQS   = Counter("alert_requests_total", "Total /alert requests", ["status"])
LAT    = Histogram("alert_latency_seconds", "Latency of /alert handler")
SENT   = Counter("alert_sent_total", "Alerts successfully delivered", ["sink"])
FAILED = Counter("alert_failed_total", "Alerts failed after retries", ["sink"])
DEDUP  = Counter("alert_deduplicated_total", "Deduplicated alerts")
DLQ_ENQ= Counter("alert_dlq_enqueued_total", "DLQ enqueued")
DLQ_DEQ= Counter("alert_dlq_dequeued_total", "DLQ dequeued")
STATE  = Gauge("alert_state_items", "In-memory dedup cache size")

class FusionSnapshot(BaseModel):
    hash_id: str
    final_score: float
    final_threshold: float
    is_anomaly: bool
    risk_level: str
    agg_ml: Optional[float] = None
    agg_nlp: Optional[float] = None
    mode: Optional[str] = None
    used_fallback: Optional[bool] = None
    reasons: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    topics: List[Dict[str, Any]] = Field(default_factory=list)

class AlertAck(BaseModel):
    hash_id: str
    delivered: bool
    sink: str
    status: str
    index: Optional[str] = None
    reason: Optional[str] = None
    dlq: Optional[bool] = None
    preview: Optional[dict] = None

#  State/clients
_http: Optional[httpx.AsyncClient] = None
_redis: Optional[redis.Redis] = None

# Local fallback for idempotency (when Redis is not available)
_DEDUP_LOCAL: Dict[str, float] = {}

# Helpers

def _top_topic(topics: list[dict]) -> tuple[str, float] | tuple[None, float]:
    if not topics:
        return (None, 0.0)
    best = max(topics, key=lambda t: float(t.get("score", 0.0)))
    return best.get("topic"), float(best.get("score", 0.0))

def _load_actions() -> None:
    global _ACTIONS
    try:
        with open(ACTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            df = data.get("default")
            tp = data.get("topics")
            if isinstance(df, list):  # validation
                _ACTIONS["default"] = [str(x) for x in df]
            if isinstance(tp, dict):
                _ACTIONS["topics"] = {str(k): [str(x) for x in v] for k, v in tp.items() if isinstance(v, list)}
            LOG.info('actions_loaded file="%s" topics=%d', ACTIONS_FILE, len(_ACTIONS["topics"]))
    except FileNotFoundError:
        LOG.warning('actions_file_missing path="%s" using_defaults=1', ACTIONS_FILE)
    except Exception as e:
        LOG.warning('actions_load_failed path="%s" err="%s" using_defaults=1', ACTIONS_FILE, type(e).__name__)

def build_human_alert(dec: dict) -> dict:
    ctx = dec.get("context") or {}
    topics = dec.get("topics") or ctx.get("topics") or []
    topic_name, topic_score = _top_topic(topics)

    ts = ctx.get("ts") or "-"
    user_id = ctx.get("user_id") or "-"
    ip = ctx.get("source_ip") or "-"
    endpoint = ctx.get("endpoint") or "-"
    method = (ctx.get("method") or "-").upper()
    risk = dec.get("risk_level") or "normal"
    score = float(dec.get("final_score", 0.0))
    trace = ctx.get("trace_id") or dec.get("hash_id") or "-"

    if topic_name:
        title = f"[{risk.upper()}] Suspicious activity: {topic_name}"
    else:
        title = f"[{risk.upper()}] Suspicious activity detected"

    actions = _ACTIONS.get("topics", {}).get(topic_name, _ACTIONS.get("default", []))
    actions_md = "\n".join(f"- {a}" for a in actions)
    
    text = (
        f"At '{ts} UTC', user '{user_id}' from IP '{ip}' triggered activity that "
        f"matches '{topic_name or 'an anomalous request'}' on endpoint '{endpoint}' via '{method}'.\n\n"
        f"- Risk level: '{risk}' (score: {score:.3f})\n"
        f"- Top topic: '{topic_name or '-'}' (confidence: {topic_score:.2f})\n"
        f"- Hash/Trace: `{trace}`\n\n"
        "Recommended actions:\n"
        f"{actions_md}\n"
    )

    return {
        "title": title,
        "text": text,
        "risk_level": risk,
        "score": score,
        "topic": topic_name,
        "topic_score": topic_score,
        "timestamp": ts,
        "labels": {
            "endpoint": endpoint,
            "method": method,
            "user_id": user_id,
            "source_ip": ip,
        },
    }

def _preview_from_snapshot(snap: "FusionSnapshot") -> dict:
    dec = {
        "hash_id":         snap.hash_id,
        "final_score":     float(snap.final_score),
        "final_threshold": float(snap.final_threshold),
        "risk_level":      snap.risk_level,
        "context":         snap.context or {},
        "topics":          snap.topics or [],
    }
    return build_human_alert(dec)
  
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _month_index(prefix: str) -> str:
    # Returns index name with current month suffix
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

def _severity_num(level: str) -> int:
    return {"normal":0, "low":1, "medium":2, "high":3, "critical":4}.get(level, 0)

def _mk_title(f: FusionSnapshot) -> str:
    ctx = f.context or {}
    endpoint = ctx.get("endpoint") or ctx.get("text_fields", {}).get("endpoint") or "-"
    method   = (ctx.get("method") or ctx.get("text_fields", {}).get("method") or "-").upper()
    topic    = "-"
    topics = ctx.get("topics") or []
    if isinstance(topics, list) and topics:
        topic = topics[0].get("topic") or "-"
    return f"[{f.risk_level.upper()}] {method} {endpoint} — {topic} (score {f.final_score:.3f})"

def _mk_summary(f: FusionSnapshot) -> str:
    ml  = f.agg_ml if f.agg_ml is not None else 0.0
    nlp = f.agg_nlp if f.agg_nlp is not None else 0.0
    try:
        final_s = float(f.final_score)
    except Exception:
        final_s = 0.0
    try:
        final_thr = float(f.final_threshold)
    except Exception:
        final_thr = 0.0
    return (
        f"Risk: {f.risk_level} (final={final_s:.3f} thr={final_thr:.3f}); "
        f"Mode: {f.mode or '-'}; ML={ml:.3f}, NLP={nlp:.3f}"
    )

def _xai_stub(f: FusionSnapshot) -> Dict[str, Any]:
    # Simple XAI stub returning top topic and first 3 reasons
    topics = f.context.get("topics") if isinstance(f.context, dict) else None
    top_topic = None
    if isinstance(topics, list) and topics:
        tt = sorted(topics, key=lambda x: x.get("score", 0.0), reverse=True)[0]
        top_topic = {"topic": tt.get("topic"), "score": float(tt.get("score", 0.0))}
    return {
        "top_topic": top_topic,
        "reasons": f.reasons[:3],
        "note": "XAI stub (SHAP/LIME can be plugged later)"
    }

def _stix_stub(f: FusionSnapshot) -> Dict[str, Any]:
    # Simple STIX note object stub
    return {
        "type": "note",
        "spec_version": "2.1",
        "created": _now_iso(),
        "modified": _now_iso(),
        "content": _mk_summary(f),
        "object_refs": [f"indicator--{f.hash_id[:32]}"],
        "custom_properties": {
            "risk_level": f.risk_level,
            "final_score": f.final_score,
            "topics": f.context.get("topics", []),
        },
    }
    
async def _sleep_jitter(attempt: int):
    await asyncio.sleep(min(0.25 * attempt, 2.0) * (0.5 + random.random()))

async def send_slack(payload: dict) -> dict:
    if not SLACK_WEBHOOK_URL:
        return {"sink":"slack", "status":"skipped"}
    msg = build_human_alert(payload)
    body = {
        "text": msg["title"],
        "blocks": [
            {"type":"section", "text":{"type":"mrkdwn", "text": f"*{msg['title']}*"}},
            {"type":"section", "text":{"type":"mrkdwn", "text": msg["text"]}},
        ],
    }
    for attempt in range(1, int(os.getenv("ALERT_RETRIES","3")) + 1):
        try:
            async with httpx.AsyncClient(timeout=float(os.getenv("ALERT_TIMEOUT","3.0"))) as cli:
                r = await cli.post(SLACK_WEBHOOK_URL, json=body)
                r.raise_for_status()
            return {"sink":"slack", "status":"ok"}
        except Exception as e:
            if attempt >= int(os.getenv("ALERT_RETRIES","3")):
                return {"sink":"slack", "status":"error", "error": str(e)}
            await _sleep_jitter(attempt)

async def send_local(payload: dict) -> dict:
    msg = build_human_alert(payload)
    log = logging.getLogger("alerting")
    log.info('human_alert %s', json.dumps(msg, ensure_ascii=False))
    log.info('human_alert_title "%s"', msg["title"])
    for line in msg["text"].splitlines():
        log.info('human_alert_text "%s"', line)
    return {"sink":"local", "status":"ok"}

async def send_email(payload: dict) -> dict:
    if not (EMAIL_SMTP_HOST and EMAIL_TO):
        return {"sink":"email", "status":"skipped"}
    msg = build_human_alert(payload)
    mime = MIMEText(msg["text"], "plain", "utf-8")
    mime["Subject"] = msg["title"]
    mime["From"] = EMAIL_USER or "alerts@example.com"
    mime["To"] = EMAIL_TO
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=5) as s:
            s.starttls(context=ctx)
            if EMAIL_USER and EMAIL_PASS:
                s.login(EMAIL_USER, EMAIL_PASS)
            s.sendmail(mime["From"], [EMAIL_TO], mime.as_string())
        return {"sink":"email", "status":"ok"}
    except Exception as e:
        return {"sink":"email", "status":"error", "error": str(e)}

async def send_opensearch_incident(payload: dict) -> dict:
    idx = f"{OS_INCIDENT_INDEX_PREFIX}-{time.strftime('%Y.%m.%d', time.gmtime())}"
    doc_ts = payload.get("context", {}).get("ts") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc = build_human_alert(payload) | {
        "@timestamp": doc_ts,
        "hash_id": payload.get("hash_id"),
        "final_score": payload.get("final_score"),
        "final_threshold": payload.get("final_threshold"),
        "reasons": payload.get("reasons") or [],
        "mode": payload.get("mode"),
        "service": "alerting",
    }
    auth = (OS_USER, OS_PASS) if OS_USER or OS_PASS else None
    url = f"{OS_URL}/{idx}/_doc"
    for attempt in range(1, int(os.getenv("ALERT_RETRIES","3")) + 1):
        try:
            async with httpx.AsyncClient(timeout=float(os.getenv("ALERT_TIMEOUT","3.0"))) as cli:
                r = await cli.post(url, json=doc, auth=auth)
                r.raise_for_status()
            return {"sink":"opensearch", "status":"ok", "index": idx}
        except Exception as e:
            if attempt >= int(os.getenv("ALERT_RETRIES","3")):
                return {"sink":"opensearch", "status":"error", "error": str(e)}
            await _sleep_jitter(attempt)

async def send_servicenow(payload: dict) -> dict:
    if not (SN_URL and SN_USER and SN_PASS):
        return {"sink":"servicenow", "status":"skipped"}
    msg = build_human_alert(payload)
    body = {
        "short_description": msg["title"],
        "description": msg["text"],
        "severity": 2 if msg["risk_level"] in ("critical","high") else 3,
        "urgency": 2 if msg["risk_level"] in ("critical","high") else 3,
        "category": "security",
    }
    for attempt in range(1, int(os.getenv("ALERT_RETRIES","3")) + 1):
        try:
            async with httpx.AsyncClient(timeout=float(os.getenv("ALERT_TIMEOUT","3.0"))) as cli:
                r = await cli.post(f"{SN_URL}/api/now/table/incident", json=body, auth=(SN_USER,SN_PASS))
                r.raise_for_status()
            return {"sink":"servicenow", "status":"ok"}
        except Exception as e:
            if attempt >= int(os.getenv("ALERT_RETRIES","3")):
                return {"sink":"servicenow", "status":"error", "error": str(e)}
            await _sleep_jitter(attempt)

SINK_IMPL = {
    "local":      send_local,
    "opensearch": send_opensearch_incident,  # incidents index in OpenSearch
    "slack":      send_slack,
    "email":      send_email,
    "servicenow": send_servicenow,
}
            
async def _fanout_selected(decision: dict, sinks: list[str]):
    # Extract risk level from decision
    risk = str(decision.get("risk_level") or decision.get("risk") or "normal").lower()
    sev = _SEV_NAME_TO_NUM.get(risk, 0)

    valid: list[str] = []
    for s in sinks:
        if s not in SINK_IMPL:
            continue
        # Incidents only for high/critical
        if s == "opensearch" and sev < _SEV_NAME_TO_NUM.get("high", 3):
            continue
        valid.append(s)

    if not valid:
        return
    results = await asyncio.gather(*(SINK_IMPL[s](decision) for s in valid), return_exceptions=True)
    for res in results:
        if isinstance(res, Exception):
            LOG.warning('sink_error err="%s"', type(res).__name__)
            FAILED.labels("sink").inc()
        else:
            SENT.labels(res.get("sink","?")).inc()
            
async def _deliver_to_opensearch(doc: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    if DRY_RUN or not OS_URL:
        return True, None, "dry-run" if DRY_RUN else "no-opensearch"
    index = _month_index(OS_INDEX_PREFIX)
    url = f"{OS_URL}/{index}/_doc"
    params = {}
    if OS_REFRESH:
        params["refresh"] = OS_REFRESH

    for attempt in range(1, OS_RETRIES + 1):
        try:
            r = await _http.post(url, json=doc, params=params)
            if r.status_code in (200, 201):
                return True, index, None
            reason = f"bad_status:{r.status_code}"
            LOG.warning('os_write_failed status=%s body=%s', r.status_code, r.text[:200])
        except Exception as e:
            reason = f"exc:{type(e).__name__}"
            LOG.warning('os_post_exc attempt=%d reason=%s', attempt, reason)
        await asyncio.sleep(min(0.25 * attempt, 2.0) * (0.5 + random.random()))
    return False, index, reason

def _dedup_key(hash_id: str) -> str:
    return f"alerted:{hash_id}"

def _dedup_seen(hash_id: str) -> bool:
    """True if already seen within the last TTL."""
    if _redis:
        ok = _redis.set(_dedup_key(hash_id), "1", ex=DEDUP_TTL, nx=True)
        return not bool(ok)
    # local fallback
    now = time.time()
    ts = _DEDUP_LOCAL.get(hash_id)
    if ts and (now - ts) < DEDUP_TTL:
        return True
    _DEDUP_LOCAL[hash_id] = now
    try:
        STATE.set(len(_DEDUP_LOCAL))
    except Exception:
        pass
    return False

def _cleanup_local_dedup():
    if not _DEDUP_LOCAL: return
    now = time.time()
    for k, ts in list(_DEDUP_LOCAL.items()):
        if now - ts >= DEDUP_TTL:
            _DEDUP_LOCAL.pop(k, None)
    try:
        STATE.set(len(_DEDUP_LOCAL))
    except Exception:
        pass

def _enq_dlq(item: Dict[str, Any]):
    if _redis:
        _redis.rpush(DLQ_KEY, json.dumps(item))
        DLQ_ENQ.inc()

async def _drain_dlq(limit: int) -> int:
    if not _redis:
        return 0
    drained = 0
    for _ in range(limit):
        raw = _redis.lpop(DLQ_KEY)
        if not raw:
            break
        try:
            item = json.loads(raw)
            ok, index, reason = await _deliver_to_opensearch(item)
            if ok:
                SENT.labels("opensearch").inc()
                DLQ_DEQ.inc()
                drained += 1
            else:
                LOG.warning('dlq_retry_failed reason="%s"', reason)
                _redis.rpush(DLQ_KEY, raw)  # re-queue
                break
        except Exception as e:
            LOG.exception('dlq_item_error')
            _redis.rpush(DLQ_KEY, raw)
            break
    return drained

# FastAPI app and endpoints

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http, _redis
    _load_actions()

    verify = OS_CA_BUNDLE if OS_CA_BUNDLE else OS_VERIFY
    auth = (OS_USER, OS_PASS) if (OS_USER or OS_PASS) else None
    _http = httpx.AsyncClient(
        headers={"X-Service": SERVICE_NAME},
        verify=verify,
        auth=auth,
        timeout=OS_TIMEOUT,
    )
    # Redis
    if REDIS_URL:
        try:
            _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            _redis.ping()
            LOG.info('redis_connected url="%s"', REDIS_URL)
        except Exception:
            _redis = None
            LOG.warning("redis_connect_failed")
    # Download recommended actions
    _load_actions()
    # Periodic cleanup of local deduplication cache
    task = asyncio.create_task(_housekeeper(), name="alerting-housekeeper")
    yield
    task.cancel()
    await _http.aclose()

async def _housekeeper():
    while True:
        await asyncio.sleep(15)
        _cleanup_local_dedup()

app = FastAPI(title="alerting", version="1.0.0", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)

@app.get("/health")
def health():
    return {"status": "ok", "ts": _now_iso()}

@app.get("/ready")
async def ready():
    ok_os = False
    if OS_URL and _http:
        try:
            r = await _http.get(f"{OS_URL}/_cluster/health?timeout=0.5s")
            ok_os = r.status_code == 200
        except Exception:
            ok_os = False
    return {"status": "ok", "deps": {"redis": bool(_redis), "opensearch": ok_os}, "version": app.version}

@app.get("/policy")
def policy():
    return {
        "orch_header": "X-Orch-Token",
        "dedup_ttl_sec": DEDUP_TTL,
        "opensearch": {
            "url": OS_URL, "index_prefix": OS_INDEX_PREFIX,
            "verify": OS_VERIFY, "timeout": OS_TIMEOUT, "retries": OS_RETRIES
        },
        "dlq": {"key": DLQ_KEY, "batch": DLQ_REPLAY_BATCH, "redis": bool(_redis)},
        "stix_enabled": STIX_ENABLED,
        "dry_run": DRY_RUN,
        "version": app.version,
        "alert_sinks": ALERT_SINKS,
        "min_risk": ALERT_MIN_RISK,
        "version": app.version,
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/reload_actions")
def reload_actions():
    _load_actions()
    return {"ok": True, "topics": len(_ACTIONS.get("topics", {}))}

@app.post("/alert", response_model=AlertAck)
async def alert(
    payload: Dict[str, Any],
    request: Request,
    x_orch_token: Optional[str] = Header(default=None, alias="X-Orch-Token"),
):
    t0 = time.perf_counter()
    trace_id = request.headers.get("X-Trace-Id") or "-"
    try:
        # Authorization
        if ORCH_TOKEN and (x_orch_token or "") != ORCH_TOKEN:
            REQS.labels("unauth").inc()
            raise HTTPException(status_code=401, detail="unauthorized")
        # Validate payload
        try:
            snap = FusionSnapshot(**(payload or {}))
        except Exception as e:
            REQS.labels("bad").inc()
            try:
                raw_preview = json.dumps(payload or {}, ensure_ascii=False)[:1000]
            except Exception:
                raw_preview = "<unserializable>"
            LOG.warning(
                'fusion_snapshot_validation_failed err="%s" payload=%s',
                type(e).__name__,
                raw_preview,
            )
            return AlertAck(
                hash_id=str((payload or {}).get("hash_id", "?")),
                delivered=False,
                sink="none",
                status="invalid",
                reason="validation_error",
                preview=None,
            )

        preview = _preview_from_snapshot(snap)
        
        # Deduplication check
        if _dedup_seen(snap.hash_id):
            DEDUP.inc()
            LOG.info('dedup_hit hash_id="%s" ttl=%d', snap.hash_id, DEDUP_TTL)
            return AlertAck(
                hash_id=snap.hash_id, delivered=True, sink="none",
                status="duplicate", reason="deduplicated",
                preview=_preview_from_snapshot(snap)
            )
        # If alerts at all - skip only "normal" or lower level
        sev_num = _severity_num(snap.risk_level)
        if sev_num < _MIN_SEV_NUM or not snap.is_anomaly:
            REQS.labels("skipped").inc()
            LOG.info('alert_skipped hash_id="%s" risk="%s"', snap.hash_id, snap.risk_level)
            return AlertAck(hash_id=snap.hash_id, delivered=False, sink="none",
                            status="skipped", reason="below_min_risk",
                            preview=_preview_from_snapshot(snap))
        # Making SIEM-doc for alerts-*
        doc = {
            "@timestamp": _now_iso(),
            "trace_id": trace_id,
            "service": SERVICE_NAME,
            "hash_id": snap.hash_id,
            "risk_level": snap.risk_level,
            "severity": _severity_num(snap.risk_level),
            "final_score": snap.final_score,
            "final_threshold": snap.final_threshold,
            "is_anomaly": snap.is_anomaly,
            "agg_ml": snap.agg_ml,
            "agg_nlp": snap.agg_nlp,
            "mode": snap.mode,
            "reasons": snap.reasons,
            "context": snap.context,
            "title": _mk_title(snap),
            "summary": _mk_summary(snap),
            "xai": _xai_stub(snap),
        }
        human = build_human_alert(snap.model_dump())
        
        if STIX_ENABLED:
            doc["stix"] = _stix_stub(snap)

        # Which sinks are allowed
        allowed = [s for s in ALERT_SINKS if s]  # already stripped above

        # 1) alerts-* if opensearch is allowed and min risk level is met
        wrote_alert = False
        os_index = None
        if "opensearch" in allowed:
            ok, os_index, reason = await _deliver_to_opensearch(doc)
            if ok:
                wrote_alert = True
                SENT.labels("opensearch").inc()
            else:
                _enq_dlq(doc)
                FAILED.labels("opensearch").inc()
                REQS.labels("dlq").inc()

        # 2) incidents-* only for HIGH/CRITICAL if allowed
        want_incidents = (sev_num >= _SEV_NAME_TO_NUM["high"]) and ("opensearch" in allowed)

        fanout_sinks = [s for s in allowed if s in SINK_IMPL and s != "opensearch"]

        # incidents fan-out
        if want_incidents:
            asyncio.create_task(_fanout_selected(snap.model_dump(), ["opensearch"]))

        # Launch fan-out for the rest of the sinks asynchronously
        if fanout_sinks:
            asyncio.create_task(_fanout_selected(snap.model_dump(), fanout_sinks))

        REQS.labels("ok").inc()
        return AlertAck(
            hash_id=snap.hash_id,
            delivered=True,
            sink=("opensearch" if wrote_alert else ("fanout" if fanout_sinks else "none")),
            status=("stored" if wrote_alert else ("sent" if fanout_sinks else "noop")),
            index=os_index,
            preview=preview,
        )

    except HTTPException:
        raise
    except Exception as e:
        REQS.labels("error").inc()
        LOG.exception("alert_failed")
        raise HTTPException(status_code=500, detail=f"alert_failed:{type(e).__name__}")
    finally:
        LAT.observe(max(time.perf_counter() - t0, 0.0))

@app.post("/replay_dlq")
async def replay_dlq(limit: Optional[int] = None):
    n = await _drain_dlq(limit or DLQ_REPLAY_BATCH)
    return {"drained": n}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL","info").lower(),
    )