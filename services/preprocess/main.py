import os
import re2 as re
import logging
import uvicorn
import time
import json
import asyncio
import math
import string
import collections

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, Union
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from collections import deque, defaultdict
from urllib.parse import urlparse, parse_qs

from common.utils import (
  getenv_str,
  getenv_int,
  make_hash_id,
  safe_log_json, 
  shannon_entropy as shannon_entropy,
  is_private_ip,
  is_json_like,
  ensure_compact_json,
  suspicious_ratio,
  path_depth,
  query_param_count,
  mask_pii_in_obj
)
from common.schemas import ApiEvent

SERVICE_NAME = "preprocess"

START_TIME = time.time()

# Configs
ORCH_TOKEN = os.getenv("ORCH_TOKEN", "")
MAX_EVENT_BYTES   = getenv_int("MAX_EVENT_BYTES", 2_000_000) # 2 MB
MAX_TEXT_BYTES    = getenv_int("MAX_TEXT_BYTES",  20_000)       # Truncating text fields
MAX_HEADERS_BYTES = getenv_int("MAX_HEADERS_BYTES", 64_000)     # Limits on headers
MAX_PAYLOAD_BYTES = getenv_int("MAX_PAYLOAD_BYTES", 256_000)    # Limits on payload
ACCESS_RE = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] "(?P<method>\S+)\s+(?P<endpoint>\S+)'
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


LOG_LEVEL = getenv_str("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","service":"'+SERVICE_NAME+'","msg":"%(message)s"}',
)
log = logging.getLogger(SERVICE_NAME)

HTTP_METHODS = {
    "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS",
    "TRACE", "CONNECT", "PROPFIND", "REPORT", "MKCOL", "COPY", "MOVE",
}

METHOD_CODE_MAP = {
    "GET": 1, "POST": 2, "PUT": 3, "DELETE": 4, "PATCH": 5, "HEAD": 6, "OPTIONS": 7,
    "OTHER": 0, "UNKNOWN": 0
}

REQ_RATE_WINDOW_SEC = int(os.getenv("REQ_RATE_WINDOW_SEC", "60"))
# Aggregation key for rate calculation
REQ_RATE_KEY = os.getenv("REQ_RATE_KEY", "endpoint")

_RATE_WINDOWS: dict[str, deque] = defaultdict(deque)  # NEW

# Regexes for capturing method in headers or endpoint
# Method in endpoint path or string
RE_METHOD_IN_ENDPOINT = re.compile(r"/?(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\b")
# X-HTTP-Method-Override header
RE_METHOD_OVERRIDE    = re.compile(r"X-HTTP-Method-Override\s*:\s*(GET|POST|PUT|DELETE|PATCH)")
# Access-Control-Request-Method header
RE_ACCESS_CONTROL     = re.compile(r"Access-Control-Request-Method\s*:\s*(GET|POST|PUT|DELETE|PATCH)")

RULES_FILE = Path(os.getenv("PREPROC_RULES_FILE", "/app/rules/base_rules.json"))
_RE_MAX_REGEX_INPUT_LEN = int(os.getenv("RE_MAX_REGEX_INPUT_LEN", "5000"))
COMPILED_RULES_PP = {}

# Prometheus metrics
REQS    = Counter("preprocess_requests_total", "Total requests", ["status", "log_kind"])
LATENCY = Histogram("preprocess_latency_seconds", "Request latency in seconds")

# Concurrency limit
PREPROCESS_CONCURRENCY = getenv_int("PREPROCESS_CONCURRENCY", 20)
SEMAPHORE = asyncio.Semaphore(PREPROCESS_CONCURRENCY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_rules_for_preproc()
    log.info('rules_loaded file="%s" groups=%d', str(RULES_FILE), len(COMPILED_RULES_PP))
    yield

app = FastAPI(title="preprocess", version="1.0.0", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)

# Input/output models
class In(BaseModel, extra="forbid"):
    event: Union[str, Dict[str, Any]]

class Out(BaseModel, extra="forbid"):
    hash_id: str
    ts: Optional[str]
    endpoint: Optional[str]
    numeric_feats: Dict[str, float]
    text_fields: Dict[str, str]
    priv_context: Optional[Dict[str, str]] = None
    HTTP_method_findings: Optional[Dict[str, Any]] = None
    risk_context: Optional[Dict[str, Any]] = None

def _endpoint_class(endpoint: str) -> str:
    e = (endpoint or "").lower()

    # payment / billing
    if e.startswith("/pay") or "invoice" in e or "billing" in e or "card" in e:
        return "payments"

    # login / auth / token / oauth
    if e.startswith("/auth") or "/login" in e or "/token" in e or "oauth" in e:
        return "auth"

    # admin-panel / backoffice
    if e.startswith("/admin") or "/root" in e or "/backoffice" in e:
        return "admin"

    # profile / account
    if "/profile" in e or "/me" in e or "/account" in e:
        return "profile"

    # session / login-status
    if "/session" in e or "/sessions" in e or "login-status" in e:
        return "session"

    # reset password
    if "reset-password" in e or "forgot-password" in e:
        return "password_reset"

    # files / upload / download
    if "/upload" in e or "/download" in e or "/files" in e or "/attachments" in e:
        return "files"

    # webhooks
    if "/webhook" in e or "/hooks/" in e:
        return "webhook"

    # public API
    if e.startswith("/api/") or "/api/v1" in e or "/api/v2" in e:
        return "public_api"

    # exports / reports
    if "export" in e or "report" in e:
        return "report_export"

    return ""

def _data_class(text_fields: dict) -> str:
    blob = " ".join([
        str(text_fields.get("payload", "")),
        str(text_fields.get("headers", "")),
        str(text_fields.get("error_text", "")),
    ]).lower()

    # 1) explicit secrets/keys
    if (
        "-----begin private key-----" in blob
        or "aws_secret_access_key" in blob
        or "secret_access_key" in blob
        or "gcloud auth activate-service-account" in blob
        or "azurestorage" in blob
    ):
        return "secrets"

    # 1.1) Cloud metadata SSRF patterns
    if (
        "169.254.169.254" in blob
        or "latest/meta-data" in blob
        or "metadata.google.internal" in blob
        or "metadata/instance" in blob
    ):
        return "secrets"

    # 2) tokens/api-keys
    if (
        "bearer " in blob
        or "authorization:" in blob
        or "api_key" in blob
        or "x-api-key" in blob
        or "access_token" in blob
        or "refresh_token" in blob
        or "id_token" in blob
        or "jwt" in blob
    ):
        return "tokens"

    # 3) auth data/passwords
    if (
        "password" in blob
        or "passwd" in blob
        or "pwd=" in blob
        or "login=" in blob
        or "credential" in blob
    ):
        return "credentials"

    # 4) PII
    if (
        "ssn" in blob
        or "social security" in blob
        or "passport" in blob
        or "national id" in blob
    ):
        return "pii"

    # 5) payment cards
    if (
        "card number" in blob
        or "creditcard" in blob
        or "cvv" in blob
        or "cvc" in blob
    ):
        return "pci"

    # 6) contact data (email/phone)
    if EMAIL_RE.search(blob) or "phone" in blob or "tel" in blob:
        return "contact"

    # 7) by default - "generic"
    return "generic"

def _auth_state(headers: dict | None, user_id: str | None = "") -> str:
    # service-account by user_id
    if user_id:
        uid = str(user_id).lower()
        if uid.startswith(("svc_", "service-", "system:")) or uid.endswith("@service"):
            return "service"

    if not headers:
        return "anonymous"

    h = {str(k).lower(): str(v).lower() for k, v in headers.items()}

    # explicit service accounts
    if "x-service-account" in h or "x-forwarded-service" in h:
        return "service"

    # third-party integration/OAuth-clients
    if "x-oauth-client-id" in h or "x-third-party-app" in h:
        return "third_party_app"

    # classic user
    if "authorization" in h:
        if "bearer" in h["authorization"]:
            return "user"
        if "basic " in h["authorization"]:
            return "user"

    if h.get("x-user-role") == "admin":
        return "admin"

    return "anonymous"

def _exposure(ip: str) -> str:
    try:
        from common.utils import is_private_ip
        return "internal" if is_private_ip(ip) else "public"
    except Exception:
        return "public"

def _blast(
    env: str,
    endpoint: str,
    endpoint_class: str,
    req_rate_rps: float,
    headers: dict | None = None,
    user_id: str | None = "",
) -> str:
    env_l = (env or "").lower()
    e     = (endpoint or "").lower()
    h     = {str(k).lower(): str(v).lower() for k, v in (headers or {}).items()}

    # 1) dev/test environments
    if env_l.startswith("dev") or env_l.startswith("test"):
        return "low"

    # 2) tenant-scoped admin endpoints
    is_tenant_scoped = (
        "/tenants/" in e
        or "/tenant/" in e
        or "/orgs/" in e
        or "/organizations/" in e
        or "x-tenant-id" in h
        or "x-org-id" in h
    )

    if endpoint_class == "admin":
        if is_tenant_scoped:
            return "med"
        return "high" 

    # 3) payments/configs
    if endpoint_class == "payments":
        if "config" in e or "settings" in e or "plans" in e or "billing" in e:
            return "high"

    # 4) fallback by request rate
    if req_rate_rps >= 2.0:
        return "high"
    if req_rate_rps >= 0.5:
        return "med"
    return "low"
  
def _rate_key(endpoint: str, method: str, source_ip: str, user_id: str) -> str:  # NEW
    k = (REQ_RATE_KEY or "endpoint").lower()
    if k == "source_ip":
        base = source_ip or "-"
    elif k == "user_id":
        base = user_id or "-"
    elif k in ("method_endpoint", "endpoint_method", "method:endpoint"):
        base = f"{method}:{endpoint}"
    else:
        base = endpoint or "-"
    return base

def _rate_bump_get(key: str, now_ts: float) -> float:
    dq = _RATE_WINDOWS[key]
    cutoff = now_ts - REQ_RATE_WINDOW_SEC
    while dq and dq[0] < cutoff:
        dq.popleft()
    dq.append(now_ts)
    return float(len(dq)) / float(max(1, REQ_RATE_WINDOW_SEC))
  
def _load_rules_for_preproc():
    global COMPILED_RULES_PP
    COMPILED_RULES_PP = {}
    try:
        d = json.loads(RULES_FILE.read_text(encoding="utf-8"))
        for atk, group in (d.get("attack_patterns") or {}).items():
            strong = []
            for rx in (group.get("strong") or []):
                try:
                    strong.append(re.compile(rx))
                except Exception:
                    strong.append({"unsafe_pattern": rx})
            tokens = (group.get("partial_tokens") or group.get("tokens") or [])
            COMPILED_RULES_PP[atk] = {"strong": strong, "tokens": [str(t).lower() for t in tokens]}
    except Exception:
        COMPILED_RULES_PP = {}
        
def _normalize_text_pp(s: str) -> str:
    import urllib.parse, html
    if not s: return ""
    try: s = urllib.parse.unquote_plus(s)
    except Exception: pass
    s = html.unescape(s)
    s = s.replace("\\'", "'").replace('\\"', '"').replace("\\u0027","'").replace("\\u003d","=")
    return " ".join(s.split()).lower()
  
def _normalize_headers(h) -> tuple[str, float]:
    if h is None:
        return "", 0.0
    if isinstance(h, dict):
        try:
            s = ";".join(f"{str(k)}:{str(v)}" for k, v in h.items())
        except Exception:
            s = str(h)
        return s, float(len(h))
    if isinstance(h, list):
        pairs = []
        for item in h:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                pairs.append(f"{str(item[0])}:{str(item[1])}")
            else:
                pairs.append(str(item))
        s = ";".join(pairs)
        return s, float(max(1, s.count(":")))
    s = str(h)
    return s, float(max(1 if s else 0, s.count(":")))

def _count_contains_pp(blob: str, pats):
    return sum(1 for p in pats if p in blob)

def _char_entropy_pp(s: str) -> float:
    if not s:
        return 0.0
    cnt = collections.Counter(s)
    n = len(s)
    return float(-sum((c / n) * math.log2(c / n) for c in cnt.values()))

def _suspicious_char_ratio_pp(s: str) -> float:
    if not s:
        return 0.0
    sus = sum(ch in "<>\"'`;$(){}[]\\|&" for ch in s)
    return float(sus / max(1, len(s)))

def _digit_ratio_pp(s: str) -> float:
    if not s:
        return 0.0
    digs = sum(ch.isdigit() for ch in s)
    return float(digs / max(1, len(s)))

def _punct_ratio_pp(s: str) -> float:
    if not s:
        return 0.0
    p = sum(ch in string.punctuation for ch in s)
    return float(p / max(1, len(s)))


def _upper_ratio_pp(s: str) -> float:
    if not s:
        return 0.0
    ups = sum(ch.isupper() for ch in s)
    return float(ups / max(1, len(s)))


def _pct_encoded_ratio_pp(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    i = 0
    pct_units = 0
    while i + 2 < n:
        ch = s[i]
        if ch == "%":
            a = s[i + 1]
            b = s[i + 2]
            if (
                ("0" <= a <= "9" or "a" <= a.lower() <= "f")
                and ("0" <= b <= "9" or "a" <= b.lower() <= "f")
            ):
                pct_units += 1
                i += 3
                continue
        i += 1
    pct_units += s.count("+")
    return float(pct_units / max(1, n))

def _token_count_pp(s: str) -> int:
    return int(len(s.split())) if s else 0

def _featurize_for_ml_from_text_pp(text: str) -> dict:
    t = _normalize_text_pp(text)
    return {
        "text_len": len(t),
        "char_entropy": _char_entropy_pp(t),
        "suspicious_char_ratio": _suspicious_char_ratio_pp(t),
        "digit_ratio": _digit_ratio_pp(t),
        "punct_ratio": _punct_ratio_pp(t),
        "token_count": _token_count_pp(t),
    }

def rule_features_for_ml(text_fields: dict) -> dict:
    payload = (text_fields.get("payload") or "")
    headers = (text_fields.get("headers") or "")
    endpoint= (text_fields.get("endpoint") or "")
    error   = (text_fields.get("error_text") or "")
    method  = (text_fields.get("method") or "")

    raw_blob = " ".join([payload, headers, endpoint, error, method])[:20000]
    lb = _normalize_text_pp(raw_blob)

    feats = _featurize_for_ml_from_text_pp(lb)

    strong_any = 0.0
    partial_any = 0.0
    for atk, spec in COMPILED_RULES_PP.items():
        s_strong = 0.0
        s_partial = 0.0
        toks = spec.get("tokens", [])
        if toks:
            s_partial = min(1.0, _count_contains_pp(lb, toks) / 3.0)

        for patt in spec.get("strong", []):
            if isinstance(patt, dict) and "unsafe_pattern" in patt:
                if len(lb) <= _RE_MAX_REGEX_INPUT_LEN:
                    try:
                        if re.compile(patt["unsafe_pattern"]).search(lb):
                            s_strong = 1.0
                            break
                    except Exception:
                        pass
                continue
            if len(lb) <= _RE_MAX_REGEX_INPUT_LEN:
                if patt.search(lb):
                    s_strong = 1.0
                    break

        feats[f"rule_{atk}_strong"]  = float(s_strong)
        feats[f"rule_{atk}_partial"] = float(s_partial)
        strong_any  = max(strong_any, s_strong)
        partial_any = max(partial_any, s_partial)

    feats["rule_any_strong"]  = float(strong_any)
    feats["rule_any_partial"] = float(partial_any)
    return feats

def detect_http_method(method_str: str, headers: dict | None = None, endpoint: str | None = None) -> tuple[str, float]:
    if method_str:
        m = method_str.strip().upper()
        if m in HTTP_METHODS:
            return m, 1.0
        if m.startswith("GET"):
            return "GET", 0.9
        if m.startswith("POST"):
            return "POST", 0.9

    if headers:
        joined = " ".join(f"{k}:{v}" for k, v in headers.items())
        if mo := RE_METHOD_OVERRIDE.search(joined):
            return mo.group(1).upper(), 0.85
        if mo := RE_ACCESS_CONTROL.search(joined):
            return mo.group(1).upper(), 0.8

    if endpoint:
        endpoint_upper = endpoint.upper()
        for m in HTTP_METHODS:
            if re.search(rf"\b{m}\b", endpoint_upper):
                return m, 0.6
        if mo := RE_METHOD_IN_ENDPOINT.search(endpoint_upper):
            return mo.group(1).upper(), 0.5

    return "UNKNOWN", 0.0
  
def parse_access_log(line:str) -> Optional[ApiEvent]:
    m = ACCESS_RE.match(line)
    if not m: return None
    return ApiEvent(
        ts=_apache_to_iso(m.group("ts")),
        endpoint=m.group("endpoint"),
        method=m.group("method"),
        headers={"user-agent":"...", "referer":"..."},
        payload={}, latency_ms=0, source_ip=m.group("ip"), user_id=""
    )

def parse_cloudtrail(obj:dict) -> Optional[ApiEvent]:
    try:
        rec = obj["Records"][0]
        return ApiEvent(
            ts=rec["eventTime"],
            endpoint=f'/{rec["eventSource"]}:{rec["eventName"]}',
            method="AWS",
            headers={"userAgent": rec.get("userAgent",""),
                     "tlsVersion": rec.get("tlsDetails",{}).get("tlsVersion","")},
            payload=rec.get("requestParameters",{}),
            latency_ms=0,
            source_ip=rec.get("sourceIPAddress",""),
            user_id=rec.get("userIdentity",{}).get("userName","")
        )
    except Exception:
        return None

def auto_parse(raw: str|dict) -> Tuple[Optional[str], Optional[ApiEvent]]:
    if isinstance(raw, str):
        ev = parse_access_log(raw)
        return ("access_log", ev) if ev else (None, None)
    if isinstance(raw, dict) and "Records" in raw:
        ev = parse_cloudtrail(raw)
        return ("cloudtrail", ev) if ev else (None, None)
    return (None, None)

def _apache_to_iso(s: str) -> str:
    try:
        s_clean = s.strip()
        if s_clean.startswith("[") and s_clean.endswith("]"):
            s_clean = s_clean[1:-1]
        dt = datetime.strptime(s_clean, "%d/%b/%Y:%H:%M:%S %z")
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return s
  
@app.get("/health")
def health():
    uptime = round(time.time() - START_TIME, 1)
    return {"status": "ok", "uptime_sec": uptime}

@app.get("/ready")
def ready():
    return {"ready": True, "version": app.version}
  
@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id") or "-"
    with trace.get_tracer(__name__).start_as_current_span("featurize") as span:
        span.set_attribute("trace_id", trace_id)
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response

@app.exception_handler(Exception)
async def unhandled_exc_handler(request, exc: Exception):
    log.exception('unhandled_exception path="%s"', request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.post("/featurize", response_model=Out)
async def featurize(
    inp: In,
    request: Request,
    x_trace_id: Optional[str] = Header(default=None, convert_underscores=False),
    x_orchestrator_token: Optional[str] = Header(default=None, alias="X-Orch-Token"),
):
    async with SEMAPHORE:
        start = time.perf_counter()
        status = "ok"
        log_kind = None
        try:
            event = inp.event or {}
            trace_id_local = x_trace_id or "-"

            has_env = bool(ORCH_TOKEN)
            header_present = x_orchestrator_token is not None
            match = bool(has_env and header_present and (x_orchestrator_token == ORCH_TOKEN))
            log.debug('auth_debug has_env=%s header_present=%s match=%s', has_env, header_present, match)

            try:
                kind, ev = auto_parse(event)
                if ev is not None:
                    log_kind = kind  # "access_log" | "cloudtrail"
                    event = ev.model_dump()
            except Exception:
                pass

            ts        = (str(event.get("ts") or "") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            endpoint  = str(event.get("endpoint") or "")
            method    = str((event.get("method") or "")).upper()
            headers   = event.get("headers")
            payload   = event.get("payload")
            latency   = event.get("latency_ms") or 0
            user_id   = str(event.get("user_id") or "")
            source_ip = str(event.get("source_ip") or "")
            error_text = str(event.get("error_text") or event.get("error") or event.get("message") or event.get("msg") or event.get("detail") or "")

            det_method, det_acc = detect_http_method(
                method,
                headers if isinstance(headers, dict) else None,
                endpoint,
            )
            method_code = float(METHOD_CODE_MAP.get(det_method, 0))

            headers_str, header_count = _normalize_headers(headers)
            headers_bytes = len(headers_str.encode("utf-8"))
            if headers_bytes > MAX_HEADERS_BYTES:
                log.warning('headers_too_large trace_id="%s" size=%d', trace_id_local, headers_bytes)
                raise HTTPException(status_code=413, detail="Headers too large")

            if isinstance(payload, (dict, list)):
                payload_text = ensure_compact_json(payload)
            else:
                payload_text = str(payload or "")
            payload_bytes = len(payload_text.encode("utf-8"))
            if payload_bytes > MAX_PAYLOAD_BYTES:
                log.warning('payload_too_large trace_id="%s" size=%d', trace_id_local, payload_bytes)
                raise HTTPException(status_code=413, detail="Payload too large")

            approx_total_bytes = headers_bytes + payload_bytes + len(endpoint.encode("utf-8")) + 256
            if approx_total_bytes > MAX_EVENT_BYTES:
                log.warning('event_too_large_total trace_id="%s" size=%d', trace_id_local, approx_total_bytes)
                raise HTTPException(status_code=413, detail="Event too large")
              
            ts_slim = (str(ts) or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))[:19]

            hash_id = str(event.get("hash_id") or make_hash_id(
                endpoint=endpoint,
                user_id=user_id,
                payload=payload_text[:2048],
                ts=ts_slim,
            ))         
            parsed = urlparse(endpoint or "/")
            path = parsed.path or "/"

            # 1) endpoint_len & path_depth – як у CSIC
            endpoint_path = path
            endpoint_len_val = float(len(endpoint_path))
            p_depth = float(sum(1 for seg in endpoint_path.split("/") if seg))

            # 2) query-derived features – кількість/довжина/ентропія query частини
            query_str = parsed.query or ""
            query_map = parse_qs(query_str, keep_blank_values=True)
            qp_count = float(len(query_map))
            query_values = [str(v or "") for vals in query_map.values() for v in vals]
            query_value_lengths = [len(v) for v in query_values]
            query_values_n = len(query_values)
            query_empty_n = sum(1 for v in query_values if not v)

            # 3) suspicious_char_ratio – ближче до CSIC: URL + payload + headers
            combined_for_susp = " ".join([endpoint or "", payload_text or "", headers_str or ""])
            susp_ratio = float(suspicious_ratio(combined_for_susp))

            # 4) is_json_payload – по Content-Type header, як у CSIC
            ct_val = ""
            if isinstance(headers, dict):
                for k, v in headers.items():
                    if str(k).lower() == "content-type":
                        ct_val = str(v)
                        break
            is_json = 1.0 if ("json" in ct_val.lower()) else 0.0
            ip_priv    = 1.0 if is_private_ip(source_ip) else 0.0
            path_entropy_val = float(shannon_entropy(endpoint_path))
            path_digit_ratio_val = float(_digit_ratio_pp(endpoint_path))
            path_upper_ratio_val = float(_upper_ratio_pp(endpoint_path))

            body_len_val = float(len(payload_text))
            body_entropy_val = float(shannon_entropy(payload_text[:MAX_TEXT_BYTES]))
            body_digit_ratio_val = float(_digit_ratio_pp(payload_text))
            body_punct_ratio_val = float(_punct_ratio_pp(payload_text))
            pct_encoded_ratio_val = float(_pct_encoded_ratio_pp(" ".join([endpoint or "", payload_text or ""])))

            header_value_lengths = []
            if isinstance(headers, dict):
                header_value_lengths = [len(str(v or "")) for v in headers.values()]
            elif isinstance(headers, list):
                for item in headers:
                    if isinstance(item, (tuple, list)) and len(item) == 2:
                        header_value_lengths.append(len(str(item[1] or "")))
                    else:
                        header_value_lengths.append(len(str(item or "")))
            elif headers_str:
                header_value_lengths = [len(headers_str)]

            # one-hot методи
            method_onehot = {
                "method_GET":  1.0 if det_method == "GET"  else 0.0,
                "method_POST": 1.0 if det_method == "POST" else 0.0,
                "method_PUT":  1.0 if det_method == "PUT"  else 0.0,
                "method_DEL":  1.0 if det_method in ("DELETE", "DEL") else 0.0,
                "method_Other":0.0,
            }
            if sum(method_onehot.values()) == 0.0:
                method_onehot["method_Other"] = 1.0

            # text_fields, щоб rule_features_for_ml бачило все
            text_fields = {
                "endpoint":   endpoint,
                "method":     method,
                "headers":    headers_str[:MAX_TEXT_BYTES],
                "payload":    payload_text[:MAX_TEXT_BYTES],
                "error_text": error_text[:MAX_TEXT_BYTES],
                "user_id":    (user_id[:64] if user_id else ""),
                "source_ip":  "[IP]" if source_ip else "",
                "log_kind":   log_kind or "generic",
            }
            text_fields = {k: ("" if v is None else str(v)) for k, v in text_fields.items()}

            # правило-базовані фічі
            rule_feats = rule_features_for_ml(text_fields)

            # rate-ключ і rps (один раз, використовуємо і в blast, і в numeric_feats)
            rate_key = _rate_key(endpoint or "-", det_method or "-", source_ip or "-", user_id or "-")
            req_rate_60s = _rate_bump_get(rate_key, time.time())

            # базові numeric фічі
            numeric_feats = {
                "endpoint_len": endpoint_len_val,
                "header_count": float(header_count),
                "payload_size": float(payload_bytes),
                "headers_entropy": float(shannon_entropy(headers_str[:MAX_TEXT_BYTES])),
                "query_param_count": qp_count,
                "path_depth": p_depth,
                "suspicious_char_ratio": susp_ratio,
                "is_json_payload": is_json,
                "query_len": float(len(query_str)),
                "query_entropy": float(shannon_entropy(query_str)),
                "query_value_len_mean": float(sum(query_value_lengths) / query_values_n) if query_values_n else 0.0,
                "query_value_len_max": float(max(query_value_lengths)) if query_value_lengths else 0.0,
                "empty_query_value_ratio": float(query_empty_n / query_values_n) if query_values_n else 0.0,
                "path_entropy": path_entropy_val,
                "path_digit_ratio": path_digit_ratio_val,
                "path_upper_ratio": path_upper_ratio_val,
                "body_len": body_len_val,
                "body_entropy": body_entropy_val,
                "body_digit_ratio": body_digit_ratio_val,
                "body_punct_ratio": body_punct_ratio_val,
                "pct_encoded_ratio": pct_encoded_ratio_val,
                "header_value_total_len": float(sum(header_value_lengths)),
                "header_value_max_len": float(max(header_value_lengths)) if header_value_lengths else 0.0,
                "http_method_code": method_code,
                "latency_ms": float(latency or 0.0),
                "ip_private": float(ip_priv),
                "req_rate_60s": float(req_rate_60s),
            }

            # додаємо метод one-hot + rule-фічі
            numeric_feats.update(method_onehot)
            numeric_feats.update(rule_feats)
            
            headers_dict = headers if isinstance(headers, dict) else None

            env_tag      = os.getenv("PREPROC_ENV", "dev")
            endpoint_cls = _endpoint_class(endpoint or "")
            data_cls     = _data_class(text_fields)
            auth_state   = _auth_state(headers_dict, user_id)
            exposure     = _exposure(source_ip or "")
            
            blast = _blast(env_tag, endpoint or "", endpoint_cls, req_rate_60s, headers_dict, user_id)

            risk_ctx = {
                "env": env_tag,
                "endpoint_class": endpoint_cls,
                "data_class": data_cls,
                "auth_state": auth_state,
                "exposure": exposure,
                "blast": blast,
            }
            
            numeric_feats_out = numeric_feats

            # HTTP-method findings
            http_find = {
                "detected": det_method,
                "accuracy": float(det_acc),
                "code": int(method_code),
            }

            priv_context = {
                "source_ip": source_ip,
                "user_id":   user_id,
                "ts":        ts,
            } if match else None

            resp = {
                "hash_id": hash_id,
                "ts": ts,
                "endpoint": endpoint,
                "numeric_feats": numeric_feats_out,
                "text_fields": text_fields,
                "HTTP_method_findings": http_find,
                "priv_context": priv_context,
                "risk_context": risk_ctx,
            }

            log.info('featurized trace_id="%s" features=%s',
                     request.headers.get("X-Trace-Id","-"),
                     safe_log_json(mask_pii_in_obj(resp)))

            REQS.labels(status="ok", log_kind=text_fields["log_kind"]).inc()
            return resp

        except HTTPException:
            REQS.labels(status="err", log_kind=log_kind or "generic").inc()
            raise
        except Exception:
            REQS.labels(status="err", log_kind=log_kind or "generic").inc()
            log.exception('featurize_failed trace_id="%s"', x_trace_id or "-")
            raise
        finally:
            LATENCY.observe(time.perf_counter() - start)
  
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(getenv_str("PORT", "8000")),
        access_log=False,
        timeout_keep_alive=5,
    )
