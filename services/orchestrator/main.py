import os
import logging
import time
import asyncio
import httpx
import requests
import uvicorn
import json
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from threading import Lock
from statistics import mean
from fastapi import FastAPI, Header, HTTPException, Request, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from common.utils import getenv_str, safe_log_json
from policy import DEFAULT_POLICY, pick_route

SERVICE_NAME = "orchestrator"
ORCH_TOKEN = getenv_str("ORCH_TOKEN", "")
HEADERS_ORCH = {"X-Orch-Token": ORCH_TOKEN} if ORCH_TOKEN else {}

# Downstream service URLs
URL_PRE = getenv_str("URL_PRE", "http://preprocess:8000/featurize")
URL_ML  = getenv_str("URL_ML",  "http://ml-inference:8000/score")
URL_NLP = getenv_str("URL_NLP", "http://nlp-inference:8000/score")
URL_FUS = getenv_str("URL_FUS", "http://fusion:8000/fuse")
URL_ALR = getenv_str("URL_ALR", "http://alerting:8000/alert")
TIMEOUT = float(getenv_str("HTTP_TIMEOUT", "5.0"))

# Prometheus metrics
REQ_PROCESS_LATENCY = Histogram(
    "orc_process_latency_seconds",
    "End-to-end /process latency",
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
)

METHOD_CONF_THR = float(os.getenv("METHOD_CONF_THR", "0.60"))

CORE_KEEP = {
    "endpoint_len", "header_count", "payload_size", "latency_ms",
    "headers_entropy", "query_param_count", "path_depth",
    "suspicious_char_ratio", "is_json_payload", "ip_private",
    "http_method_code", "http_method_accuracy", "req_rate_60s",
    "method_GET", "method_POST", "method_PUT", "method_DEL", "method_Other",
}

PRE_CALLS = Counter("orc_preprocess_calls_total", "Preprocess calls", ["status"])
PRE_LAT   = Histogram("orc_preprocess_latency_seconds", "Preprocess latency seconds")

ML_CALLS  = Counter("orc_ml_calls_total", "ML calls", ["status"])
ML_LAT    = Histogram("orc_ml_latency_seconds", "ML latency seconds")

NLP_CALLS = Counter("orc_nlp_calls_total", "NLP calls", ["status"])
NLP_LAT   = Histogram("orc_nlp_latency_seconds", "NLP latency seconds")

FUS_CALLS = Counter("orc_fusion_calls_total", "Fusion calls", ["status"])
FUS_LAT   = Histogram("orc_fusion_latency_seconds", "Fusion latency seconds")

ALR_CALLS = Counter("orc_alert_calls_total", "Alert pushes", ["status"])

FINAL_DECISIONS = Counter(
    "orc_final_decisions_total",
    "Final decision categories",
    ["is_anomaly"]
)

LOG_LEVEL = getenv_str("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","service":"'+SERVICE_NAME+'","msg":"%(message)s"}',
)
log = logging.getLogger(SERVICE_NAME)

# Helpers for log replay endpoint
ACCESS_RE = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] "(?P<method>\S+)\s+(?P<endpoint>\S+)'
)

def _apache_to_iso(s: str) -> str:
    try:
        s_clean = s.strip()
        if s_clean.startswith("[") and s_clean.endswith("]"):
            s_clean = s_clean[1:-1]
        dt = datetime.strptime(s_clean, "%d/%b/%Y:%H:%M:%S %z")
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return s

def parse_access_log_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single Apache/Nginx-style access log line into an event dict
    compatible with ApiEvent/`/process`.
    """
    m = ACCESS_RE.match(line)
    if not m:
        return None
    ts_iso = _apache_to_iso(m.group("ts"))
    return {
        "ts": ts_iso,
        "endpoint": m.group("endpoint"),
        "method": m.group("method"),
        "headers": {},
        "payload": {},
        "latency_ms": 0,
        "source_ip": m.group("ip"),
        "user_id": "",
    }

# Extended log replay helpers for CloudTrail/CloudWatch/JSONL
def parse_cloudtrail_record(rec: dict) -> Optional[Dict[str, Any]]:
    """
    Convert a single AWS CloudTrail record into an event dict compatible with /process.
    """
    try:
        ts = rec.get("eventTime")
        endpoint = f'/{rec.get("eventSource","unknown")}:{rec.get("eventName","")}'
        user_agent = rec.get("userAgent", "")
        tls_ver = (rec.get("tlsDetails") or {}).get("tlsVersion", "")
        payload = rec.get("requestParameters", {}) or {}
        src_ip = rec.get("sourceIPAddress", "")
        user = (rec.get("userIdentity") or {}).get("userName", "")
        return {
            "ts": ts,
            "endpoint": endpoint,
            "method": "AWS",
            "headers": {"userAgent": user_agent, "tlsVersion": tls_ver},
            "payload": payload,
            "latency_ms": 0,
            "source_ip": src_ip,
            "user_id": user or "",
        }
    except Exception:
        return None

def parse_cloudwatch_entry(obj: dict) -> Optional[Dict[str, Any]]:
    """
    Convert a CloudWatch Logs JSON entry into an event dict.
    Assumes JSONL with keys like 'timestamp', 'message', 'logGroup', 'logStream'.
    """
    try:
        ts = obj.get("timestamp")
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
            ts_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            ts_iso = str(ts) if ts else ""
        message = obj.get("message", "")
        log_group = obj.get("logGroup", "")
        log_stream = obj.get("logStream", "")
        endpoint = f"/cloudwatch/{log_group or 'logs'}"
        return {
            "ts": ts_iso,
            "endpoint": endpoint,
            "method": "LOG",
            "headers": {"logStream": log_stream},
            "payload": {"message": message},
            "latency_ms": 0,
            "source_ip": obj.get("sourceIPAddress", ""),
            "user_id": obj.get("user", "") or "",
        }
    except Exception:
        return None

def parse_log_line(kind: Optional[str], line: str) -> list[Dict[str, Any]]:
    """
    Dispatch parsing for different log kinds.
    If `kind` is None/empty, auto-detect format per line:

    - non-JSON text            → Apache/Nginx access log
    - JSONL with `Records`     → CloudTrail
    - JSONL with message+logGroup/logStream → CloudWatch Logs
    - other JSONL              → already ApiEvent-compatible dict
    """
    events: list[Dict[str, Any]] = []
    kind_l = (kind or "").lower()

    # access log
    if kind_l == "access_log":
        ev = parse_access_log_line(line)
        if ev:
            events.append(ev)
        return events

    # JSON-based kinds
    if kind_l:
        try:
            obj = json.loads(line)
        except Exception:
            return events

        if kind_l in ("cloudtrail_jsonl", "cloudtrail"):
            recs = obj.get("Records") or []
            if isinstance(recs, dict):
                recs = [recs]
            for rec in recs:
                if not isinstance(rec, dict):
                    continue
                ev = parse_cloudtrail_record(rec)
                if ev:
                    events.append(ev)
            return events

        if kind_l in ("cloudwatch_jsonl", "cloudwatch"):
            ev = parse_cloudwatch_entry(obj)
            if ev:
                events.append(ev)
            return events

        if kind_l in ("event_jsonl", "jsonl"):
            if isinstance(obj, dict):
                # provide a few safe defaults if missing
                ev = {
                    "ts": obj.get("ts") or obj.get("timestamp"),
                    "endpoint": obj.get("endpoint", "/"),
                    "method": obj.get("method", "GET"),
                    "headers": obj.get("headers", {}),
                    "payload": obj.get("payload", {}),
                    "latency_ms": obj.get("latency_ms", 0),
                    "source_ip": obj.get("source_ip", ""),
                    "user_id": obj.get("user_id", ""),
                }
                events.append(ev)
            return events

    # Auto-detect kind
    if not kind_l:
        stripped = line.lstrip()

        # 1) Trying JSON
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                obj = json.loads(stripped)
            except Exception:
                # Fallback: as access_log
                ev = parse_access_log_line(line)
                if ev:
                    events.append(ev)
                return events

            # CloudTrail JSONL
            if isinstance(obj, dict) and "Records" in obj:
                recs = obj.get("Records") or []
                if isinstance(recs, dict):
                    recs = [recs]
                for rec in recs:
                    if not isinstance(rec, dict):
                        continue
                    ev = parse_cloudtrail_record(rec)
                    if ev:
                        events.append(ev)
                return events

            # CloudWatch JSONL
            if isinstance(obj, dict) and "message" in obj and ("logGroup" in obj or "logStream" in obj):
                ev = parse_cloudwatch_entry(obj)
                if ev:
                    events.append(ev)
                return events

            # Fallback to ApiEvent-like JSONL
            if isinstance(obj, dict):
                ev = {
                    "ts": obj.get("ts") or obj.get("timestamp"),
                    "endpoint": obj.get("endpoint", "/"),
                    "method": obj.get("method", "GET"),
                    "headers": obj.get("headers", {}),
                    "payload": obj.get("payload", {}),
                    "latency_ms": obj.get("latency_ms", 0),
                    "source_ip": obj.get("source_ip", ""),
                    "user_id": obj.get("user_id", ""),
                }
                events.append(ev)
                return events

            return events

        # 2) Fallback to access_log
        ev = parse_access_log_line(line)
        if ev:
            events.append(ev)
        return events

    # Unknown kind
    return events

class In(BaseModel):
    event: Dict[str, Any]
    eval_label: Optional[int] = None
    eval_attack_type: Optional[str] = None

# For log replay endpoint
from typing import List

class ReplayRequest(BaseModel):
    path: str
    max_events: Optional[int] = None
    kind: Optional[str] = None

class EvalStats:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, label: int, score: float, pred: int, attack_type: str = ""):
        self.rows.append({
            "label": int(label),
            "score": float(score),
            "pred": int(pred),
            "attack_type": attack_type or "",
        })

    def _metrics_for_rows(self, rows: list[dict]) -> Dict[str, float]:
        if not rows:
            return {
                "count": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.5,
                "mean_score_normal": 0.0,
                "mean_score_attack": 0.0,
            }

        labels = [r["label"] for r in rows]
        preds  = [r["pred"]  for r in rows]
        scores = [r["score"] for r in rows]

        n = len(rows)
        tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
        tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
        fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

        acc = (tp + tn) / n if n > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        auc = _roc_auc(scores, labels)

        normal_scores = [s for s, y in zip(scores, labels) if y == 0]
        attack_scores = [s for s, y in zip(scores, labels) if y == 1]
        mean_norm   = mean(normal_scores) if normal_scores else 0.0
        mean_attack = mean(attack_scores) if attack_scores else 0.0

        return {
            "count": n,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "mean_score_normal": mean_norm,
            "mean_score_attack": mean_attack,
        }

    def overall(self) -> Dict[str, float]:
        return self._metrics_for_rows(self.rows)

    def by_attack(self) -> Dict[str, Dict[str, float]]:
        buckets: Dict[str, list[dict]] = {}
        for r in self.rows:
            at = r.get("attack_type") or "unknown"
            buckets.setdefault(at, []).append(r)
        return {at: self._metrics_for_rows(rs) for at, rs in buckets.items()}


def _roc_auc(scores: list[float], labels: list[int]) -> float:
    assert len(scores) == len(labels)
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)]
    pairs.sort(key=lambda x: x[0])

    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank = 1
    sum_ranks_pos = 0.0
    i = 0
    m = len(pairs)
    while i < m:
        j = i + 1
        while j < m and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_avg = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                sum_ranks_pos += rank_avg
        rank += (j - i)
        i = j

    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


EVAL_LOCK = Lock()
EVAL_STATS: Dict[str, EvalStats] = {
    "ml": EvalStats(),
    "nlp": EvalStats(),
    "fusion": EvalStats(),
}

def _istrue(x: str | None) -> bool:
    return (x or "").strip().lower() in ("1","true","yes","on")

def _redact_preprocess_section(pre: dict) -> dict:
    pre = dict(pre or {})
    nf = dict(pre.get("numeric_feats") or {})
    nf = {k: v for k, v in nf.items() if not k.startswith("rule_")}

    nf = {k: v for k, v in nf.items() if (k in CORE_KEEP) and (v != 0.0 or k in {"latency_ms","http_method_code","http_method_accuracy","req_rate_60s"})}

    http_find = (pre.get("HTTP_method_findings") or {})
    det = str(http_find.get("detected") or "").upper()
    acc = float(http_find.get("accuracy") or 0.0)

    for k in list(nf.keys()):
        if k.startswith("method_"):
            nf.pop(k, None)

    if acc > METHOD_CONF_THR and det:
        nf[f"method_{det}"] = 1.0
    else:
        nf["method_Other"] = 1.0

    out = dict(pre)
    out["numeric_feats"] = nf
    return out
  
def _human_text_fallback(dec: dict) -> str:
    ctx = dec.get("context") or {}
    topics = dec.get("topics") or ctx.get("topics") or []
    top = None
    try:
        if isinstance(topics, list) and topics:
            top = max(topics, key=lambda t: float(t.get("score", 0.0)))
    except Exception:
        top = None
    topic = (top or {}).get("topic") or "an anomalous request"
    tscore = float((top or {}).get("score", 0.0))
    ts = ctx.get("ts") or "-"
    user_id = (ctx.get("user_id") or "-")
    ip = (ctx.get("source_ip") or "-")
    endpoint = (ctx.get("endpoint") or "-")
    method = (ctx.get("method") or "-").upper()
    risk = dec.get("risk_level") or "normal"
    score = float(dec.get("final_score", 0.0))
    trace = ctx.get("trace_id") or dec.get("hash_id") or "-"
    lines = [
        f"At **{ts} UTC**, user **{user_id}** from IP **{ip}** triggered activity that matches *{topic}* on endpoint **{endpoint}** via **{method}**.",
        "",
        f"- Risk level: **{risk}** (score: {score:.3f})",
        f"- Top topic: **{topic}** (confidence: {tscore:.2f})",
        f"- Hash/Trace: `{trace}`",
    ]
    return "\n".join(lines) + "\n"

def _filter_zero_feats(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        try:
            fv = float(v)
            if fv != 0.0:
                out[k] = v
        except Exception:
            out[k] = v
    return out

def _apply_output_policy(out: dict, detailed: bool) -> dict:
    if detailed:
        return out  # whole object without filtering

    pp = out.get("preprocess")
    if isinstance(pp, dict) and isinstance(pp.get("numeric_feats"), dict):
        pp["numeric_feats"] = _filter_zero_feats(pp["numeric_feats"])

    for blk in ("ml", "nlp"):
        b = out.get(blk)
        if isinstance(b, dict):
            b.pop("per_model", None)
            b.pop("used_models", None)

    fus = out.get("fusion")
    if isinstance(fus, dict):
        fus.pop("context", None)
        fus.pop("topics", None)

    out.pop("human", None)
    out.pop("human_text", None)

    return out
  
def _update_eval(
    label: int,
    attack_type: str,
    ml: Dict[str, Any],
    nlp: Dict[str, Any],
    dec: Dict[str, Any],
) -> Dict[str, Any]:
    if label not in (0, 1):
        return {}

    # ML
    ml_score = float(ml.get("ml_score", 0.0))
    ml_thr   = float(ml.get("threshold", 0.5))
    ml_pred  = int(ml.get("threshold_hit", ml_score >= ml_thr))

    # NLP
    nlp_score = float(nlp.get("nlp_score", 0.0))
    nlp_thr   = float(nlp.get("threshold", 0.5))
    nlp_pred  = int(nlp.get("threshold_hit", nlp_score >= nlp_thr))

    # Fusion
    fus_score = float(dec.get("final_score", 0.0))
    fus_thr   = float(dec.get("final_threshold", 0.7))
    fus_pred  = int(dec.get("is_anomaly", fus_score >= fus_thr))

    with EVAL_LOCK:
        EVAL_STATS["ml"].add(label, ml_score, ml_pred, attack_type)
        EVAL_STATS["nlp"].add(label, nlp_score, nlp_pred, attack_type)
        EVAL_STATS["fusion"].add(label, fus_score, fus_pred, attack_type)

        overall = {
            "ml": EVAL_STATS["ml"].overall(),
            "nlp": EVAL_STATS["nlp"].overall(),
            "fusion": EVAL_STATS["fusion"].overall(),
        }
    return overall

app = FastAPI(title="orchestrator", version="1.0.0")
POLICY = DEFAULT_POLICY

# Helpers
def _call_fusion(payload: dict, trace_id: Optional[str] = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if trace_id:
        headers["X-Trace-Id"] = trace_id
    if ORCH_TOKEN:
        headers["X-Orch-Token"] = ORCH_TOKEN
    r = requests.post(URL_FUS, json=payload, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()
  
# Asynchronous calls to downstream services
async def call_ml(url: str, payload: dict, timeout: float) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as cli:
        r = await cli.post(url, json=payload)
        r.raise_for_status()
        return r.json()

async def call_nlp(url: str, payload: dict, timeout: float) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as cli:
        r = await cli.post(url, json=payload)
        r.raise_for_status()
        return r.json()


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    return {
        "policy_defaults": POLICY.get("defaults", {}),
        "targets": {"pre": URL_PRE, "ml": URL_ML, "nlp": URL_NLP, "fus": URL_FUS, "alr": URL_ALR},
        "has_orch_token": bool(ORCH_TOKEN),
    }


@app.post("/replay_access_log")
async def replay_access_log(
    req: ReplayRequest,
    request: Request,
    concurrency: int = Query(10, ge=1, le=256),
    x_trace_id: Optional[str] = Header(default=None, convert_underscores=False),
):
    """
    Read a log file (access.log, CloudTrail JSONL, CloudWatch JSONL, or event JSONL)
    from the orchestrator container and replay its entries through the full /process pipeline.
    """
    path = req.path
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail=f"log file not found: {path}")

    kind = (req.kind or "").lower()

    # Build list of events from the log file
    events: list[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed_events = parse_log_line(kind, line)
                if not parsed_events:
                    continue
                for ev in parsed_events:
                    events.append(ev)
                    if req.max_events is not None and len(events) >= req.max_events:
                        break
                if req.max_events is not None and len(events) >= req.max_events:
                    break
    except Exception as e:
        log.error('replay_read_error path="%s" err="%s"', path, e)
        raise HTTPException(status_code=500, detail="failed to read log file")

    if not events:
        return {"status": "ok", "events_total": 0, "events_sent": 0}

    base_url = str(request.base_url).rstrip("/")
    # Force detailed responses from /process
    process_url = base_url + "/process?detailed=1"

    # Reuse trace ID if provided
    trace = x_trace_id or request.headers.get("X-Trace-Id") or "-"

    sent = 0
    previews: list[str] = []

    # Effective concurrency
    eff_conc = max(1, int(concurrency))
    t0 = time.perf_counter()
    sem = asyncio.Semaphore(eff_conc)

    async with httpx.AsyncClient(timeout=TIMEOUT) as cli:

        async def _send_one(ev: Dict[str, Any]):
            nonlocal sent, previews
            async with sem:
                try:
                    headers = {
                        "X-Trace-Id": trace,
                        "Content-Type": "application/json",
                    }
                    if ORCH_TOKEN:
                        headers["X-Orch-Token"] = ORCH_TOKEN

                    resp = await cli.post(
                        process_url,
                        json={"event": ev},
                        headers=headers,
                    )
                    resp.raise_for_status()

                    # Count successful sends
                    sent += 1

                    # Try to extract a human-friendly preview from the response
                    try:
                        obj = resp.json()
                    except Exception:
                        obj = None

                    if isinstance(obj, dict):
                        preview = (
                            (obj.get("alert") or {}).get("preview", {}).get("text")
                            or obj.get("human_text")
                            or (obj.get("human") or {}).get("text")
                        )
                        if preview:
                            previews.append(preview)
                except Exception as e:
                    log.error('replay_process_error path="%s" err="%s"', path, e)

        # Fire off tasks with concurrency limiting
        tasks = [asyncio.create_task(_send_one(ev)) for ev in events]
        if tasks:
            await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t0

    return {
        "status": "ok",
        "events_total": len(events),
        "events_sent": sent,
        "log_file": path,
        "elapsed_seconds": round(elapsed, 3),
        "avg_ms_per_event": round((elapsed / sent * 1000.0), 1) if sent else None,
        "concurrency": eff_conc,
        "previews": previews,
    }

@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    log.exception('unhandled_exception path="%s"', request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.post("/process")
async def process(
    inp: In,
    request: Request,
    x_trace_id: Optional[str] = Header(default=None, convert_underscores=False),
):
    trace = x_trace_id or request.headers.get("X-Trace-Id") or "-"
    base_headers = {"X-Trace-Id": trace, "Content-Type": "application/json"}
    if ORCH_TOKEN:
        base_headers["X-Orch-Token"] = ORCH_TOKEN

    log.info('process_start trace_id="%s" event=%s', trace, safe_log_json(inp.event))

    with REQ_PROCESS_LATENCY.time():
        # 1) Preprocess
        t0 = time.perf_counter()
        try:
            r = requests.post(URL_PRE, json={"event": inp.event}, headers=base_headers, timeout=TIMEOUT)
            r.raise_for_status()
            fv = r.json() or {}
            PRE_CALLS.labels("ok").inc()
        except requests.RequestException as e:
            PRE_CALLS.labels("error").inc()
            PRE_LAT.observe(time.perf_counter() - t0)
            log.error('preprocess_error trace_id="%s" err="%s"', trace, e)
            raise HTTPException(status_code=502, detail="preprocess unavailable")
        PRE_LAT.observe(time.perf_counter() - t0)

        tf  = fv.get("text_fields") or {}
        num = fv.get("numeric_feats") or {}
        rid = fv.get("hash_id")
        if not isinstance(tf, dict) or not isinstance(num, dict) or not rid:
            log.error('preprocess_invalid trace_id="%s" keys=%s', trace, list(fv.keys()))
            raise HTTPException(status_code=502, detail="preprocess invalid output")
          
        pre_redacted = _redact_preprocess_section(fv)

        route = pick_route(POLICY, tf, num)

        # 2) ML & NLP in parallel
        t_ml = time.perf_counter()
        t_nlp = t_ml
        ml = {}
        nlp = {}
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT, headers=base_headers) as cli:
                ml_req  = cli.post(URL_ML,  json={"hash_id": rid, "numeric_feats": num})
                nlp_req = cli.post(URL_NLP, json={"hash_id": rid, "text_fields": tf, "priv_context": fv.get("priv_context")})
                ml_resp, nlp_resp = await asyncio.gather(ml_req, nlp_req)

            ml_resp.raise_for_status()
            ML_CALLS.labels("ok").inc()
            ml = ml_resp.json()
        except Exception as e:
            ML_CALLS.labels("error").inc()
            log.error('ml_call_error trace_id="%s" err="%s"', trace, e)
        finally:
            ML_LAT.observe(time.perf_counter() - t_ml)

        try:
            nlp_resp.raise_for_status()
            NLP_CALLS.labels("ok").inc()
            nlp = nlp_resp.json()
        except Exception as e:
            NLP_CALLS.labels("error").inc()
            log.error('nlp_call_error trace_id="%s" err="%s"', trace, e)
        finally:
            NLP_LAT.observe(time.perf_counter() - t_nlp)

        # 3) Fusion
        ml_scores  = [p.get("score", 0.0) for p in (ml.get("per_model") or [])]
        nlp_scores = [p.get("score", 0.0) for p in (nlp.get("per_model") or [])]

        priv_ctx  = fv.get("priv_context") or {}
        risk_ctx  = fv.get("risk_context") or {}
        topics_li = nlp.get("topics") or []

        context = {
            "ts":        fv.get("ts"),
            "endpoint":  fv.get("endpoint"),
            "method":   (tf.get("method") or "").upper(),
            "source_ip": priv_ctx.get("source_ip") if isinstance(priv_ctx, dict) else None,
            "user_id":   priv_ctx.get("user_id")   if isinstance(priv_ctx, dict) else None,
            **(risk_ctx if isinstance(risk_ctx, dict) else {}),
            "topics": topics_li if isinstance(topics_li, list) else [],
            "trace_id": trace,
            "hkey": priv_ctx.get("source_ip") or rid,
        }

        # Prepare payload for fusion with both per-model and aggregated scores
        fusion_in = {
            "hash_id": rid,
            "context": context,
            "ml_scores": ml_scores,
            "nlp_scores": nlp_scores,
            "ml_score": float(ml.get("ml_score")) if isinstance(ml.get("ml_score"), (int, float)) else ml.get("ml_score"),
            "nlp_score": float(nlp.get("nlp_score")) if isinstance(nlp.get("nlp_score"), (int, float)) else nlp.get("nlp_score"),
        }

        ORCH_FUSION_OVERRIDE = os.getenv("ORCH_FUSION_OVERRIDE", "0").lower() in ("1","true","yes","on")
        if ORCH_FUSION_OVERRIDE:
            fusion_in["policy"] = {
                "alpha": route.get("fusion", {}).get("alpha", 0.6),
                "beta":  route.get("fusion", {}).get("beta",  0.4),
                "mode":  route.get("fusion", {}).get("mode",  "weighted"),
                "threshold_final": route.get("thresholds", {}).get("final", POLICY["defaults"]["thresholds"]["final"]),
            }

        t_fus = time.perf_counter()
        try:
            dec = requests.post(URL_FUS, json=fusion_in, headers=base_headers, timeout=TIMEOUT).json()
            FUS_CALLS.labels("ok").inc()
        except Exception as e:
            FUS_CALLS.labels("error").inc()
            log.error('fusion_error trace_id="%s" err="%s"', trace, e)
            # fallback with simple logic
            dec = {
                "hash_id": rid,
                "final_score": max((ml_scores + nlp_scores) or [0.0]),
                "final_threshold": route.get("thresholds", {}).get("final", POLICY["defaults"]["thresholds"]["final"]),
                "is_anomaly": False,
                "used_fallback": True,
                "risk_level": "normal",
                "context": {"endpoint": fv.get("endpoint"), "ts": fv.get("ts")},
                "mode": "fallback",
            }
        finally:
            FUS_LAT.observe(time.perf_counter() - t_fus)

        FINAL_DECISIONS.labels("true" if dec.get("is_anomaly") else "false").inc()
        log.info('fusion_decision trace_id="%s" decision=%s', trace, json.dumps(dec))

        # 4) Alerting
        dec.setdefault("context", {}).update(context)
        dec["topics"] = context.get("topics", [])

        ackj = {}
        try:
            ack = requests.post(URL_ALR, json=dec, headers=base_headers, timeout=TIMEOUT)
            ack.raise_for_status()
            ALR_CALLS.labels("ok").inc()
            if "application/json" in (ack.headers.get("content-type", "").lower()):
                ackj = ack.json()
        except requests.RequestException as e:
            ALR_CALLS.labels("error").inc()
            log.error('alert_error trace_id="%s" err="%s"', trace, e)

        eval_summary = None
        if inp.eval_label is not None:
            try:
                lbl = 1 if int(inp.eval_label) == 1 else 0
                atk = inp.eval_attack_type or ""
                eval_summary = _update_eval(lbl, atk, ml, nlp, dec)
            except Exception as e:
                log.error('eval_update_error trace_id="%s" err="%s"', trace, e)
                eval_summary = None

        out = {
            "hash_id": rid,
            "preprocess": pre_redacted,
            "ml": ml,
            "nlp": nlp,
            "fusion": dec,
            "alert": ackj,
            "route": route,
            "status": "processed",
            "trace_id": trace,
        }

        if eval_summary is not None:
            out["eval"] = {
                "label": int(inp.eval_label),
                "attack_type": inp.eval_attack_type,
                "summary": eval_summary,
            }

        prev = (ackj or {}).get("preview") or {}
        if prev:
            out["human"] = prev
            out["human_text"] = prev.get("text")

        is_priv = bool(ORCH_TOKEN) and (request.headers.get("X-Orch-Token") == ORCH_TOKEN)
        want_detailed = is_priv and (
            _istrue(request.query_params.get("detailed")) or
            _istrue(request.headers.get("X-Detailed"))
        )

        if not want_detailed:
            txt = (prev.get("text") if isinstance(prev, dict) else None) \
                  or out.get("human_text") \
                  or _human_text_fallback(dec)

            if eval_summary:
                fus = eval_summary.get("fusion", {})
                txt += "\n[Eval] Fusion F1={f1:.3f}, ROC AUC={auc:.3f}, N={n}\n".format(
                    f1=fus.get("f1", 0.0),
                    auc=fus.get("roc_auc", 0.5),
                    n=int(fus.get("count", 0)),
                )

            log.info('process_ok trace_id="%s" hash_id="%s" mode="text"', trace, rid)
            if not txt.endswith("\n"):
                txt = txt + "\n"
            if not txt.startswith("\n"):
                txt = "\n" + txt
            return Response(content=txt, media_type="text/plain; charset=utf-8",
                            headers={"X-Trace-Id": trace})

        out = _apply_output_policy(out, detailed=True)
        log.info('process_ok trace_id="%s" hash_id="%s" mode="json"', trace, rid)
        return JSONResponse(out, headers={"X-Trace-Id": trace})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(getenv_str("PORT", "8000")))