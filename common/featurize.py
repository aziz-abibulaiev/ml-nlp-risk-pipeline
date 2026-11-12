"""
Request-to-event conversion and numeric ML features.

Shared by the preprocess service and the offline scripts, so training and serving use the
same features.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from common.utils import (
    ensure_compact_json,
    is_json_like,
    is_private_ip,
    path_depth,
    query_param_count,
    shannon_entropy,
    suspicious_ratio,
)

METHOD_CODE_MAP = {
    "GET": 1, "POST": 2, "PUT": 3, "DELETE": 4, "PATCH": 5, "HEAD": 6, "OPTIONS": 7,
    "OTHER": 0, "UNKNOWN": 0,
}

# Combined Log Format: ip ident user [ts] "METHOD target PROTO" status size "referer" "ua"
ACCESS_RE = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] '
    r'"(?P<method>\S+)\s+(?P<target>\S+)(?:\s+(?P<proto>[^"]*))?" '
    r'(?P<status>\d{3}|-) (?P<size>\S+)'
    r'(?: "(?P<referer>[^"]*)" "(?P<ua>[^"]*)")?'
)

_EMPTY = {"", "-", "nan", "None"}


def _clean(v: Any) -> str:
    s = "" if v is None else str(v)
    return "" if s.strip() in _EMPTY else s


def apache_to_iso(s: str) -> str:
    try:
        s = s.strip().strip("[]")
        dt = datetime.strptime(s, "%d/%b/%Y:%H:%M:%S %z")
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return s


def request_to_event(
    method: str,
    url: str,
    user_agent: str = "",
    referrer: str = "",
    status: Any = "",
    body: str = "",
    ts: str = "",
    source_ip: str = "",
) -> Dict[str, Any]:
    """
    Build a full event from request fields.

    endpoint = path + query (as received); payload = query string, followed by the
    request body when present (joined with '&'); headers carry User-Agent/Referer.
    """
    url = _clean(url)
    try:
        parts = urlsplit(url)
        path_q = parts.path + (("?" + parts.query) if parts.query else "")
        query = parts.query
    except ValueError:
        path_q, query = url, (url.split("?", 1)[1] if "?" in url else "")
    if parts_has_scheme(url):
        endpoint = path_q or "/"
    else:
        endpoint = url or "/"
    body = _clean(body)
    payload = "&".join(p for p in (query, body) if p)
    headers = {}
    if _clean(user_agent):
        headers["User-Agent"] = _clean(user_agent)
    if _clean(referrer):
        headers["Referer"] = _clean(referrer)
    st = _clean(status)
    return {
        "ts": ts or "",
        "endpoint": endpoint,
        "method": (_clean(method) or "GET").upper(),
        "headers": headers,
        "payload": payload,
        "status_code": int(st) if st.isdigit() else None,
        "latency_ms": 0,
        "source_ip": _clean(source_ip),
        "user_id": "",
    }


def parts_has_scheme(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def parse_access_log_line(line: str) -> Optional[Dict[str, Any]]:
    m = ACCESS_RE.match(line)
    if not m:
        return None
    return request_to_event(
        method=m.group("method"),
        url=m.group("target"),
        user_agent=m.group("ua") or "",
        referrer=m.group("referer") or "",
        status=m.group("status"),
        ts=apache_to_iso(m.group("ts")),
        source_ip=m.group("ip"),
    )


def headers_string(h: Any) -> tuple[str, float]:
    if h is None:
        return "", 0.0
    if isinstance(h, dict):
        return ";".join(f"{k}:{v}" for k, v in h.items()), float(len(h))
    if isinstance(h, list):
        pairs = [f"{i[0]}:{i[1]}" if isinstance(i, (tuple, list)) and len(i) == 2 else str(i) for i in h]
        s = ";".join(pairs)
        return s, float(max(1, s.count(":")))
    s = str(h)
    return s, float(max(1 if s else 0, s.count(":")))


def payload_text(payload: Any) -> str:
    if isinstance(payload, (dict, list)):
        return "" if not payload else ensure_compact_json(payload)
    return "" if payload is None else str(payload)


def numeric_features(event: Dict[str, Any], det_method: str, det_acc: float,
                     max_text_bytes: int = 20_000) -> Dict[str, float]:
    """Deterministic numeric features (everything except req_rate_60s and rule_* features)."""
    endpoint = str(event.get("endpoint") or "")
    method = str(event.get("method") or "").upper()
    hs, hcount = headers_string(event.get("headers"))
    pt = payload_text(event.get("payload"))
    latency = event.get("latency_ms") or 0
    onehot = {
        "method_GET": 1.0 if method == "GET" else 0.0,
        "method_POST": 1.0 if method == "POST" else 0.0,
        "method_PUT": 1.0 if method == "PUT" else 0.0,
        "method_DEL": 1.0 if method in ("DELETE", "DEL") else 0.0,
        "method_Other": 0.0,
    }
    if sum(onehot.values()) == 0.0:
        onehot["method_Other"] = 1.0
    return {
        "endpoint_len": float(len(endpoint)),
        "header_count": float(hcount),
        "payload_size": float(len(pt.encode("utf-8"))),
        "latency_ms": float(latency) if isinstance(latency, (int, float)) else 0.0,
        "headers_entropy": float(shannon_entropy(hs[:max_text_bytes])),
        **onehot,
        "query_param_count": float(query_param_count(endpoint)),
        "path_depth": float(path_depth(endpoint.split("?", 1)[0])),
        "suspicious_char_ratio": float(suspicious_ratio(pt)),
        "is_json_payload": 1.0 if (pt and is_json_like(event.get("payload"))) else 0.0,
        "ip_private": 1.0 if is_private_ip(str(event.get("source_ip") or "")) else 0.0,
        "http_method_code": float(METHOD_CODE_MAP.get(det_method, 0)),
        "http_method_accuracy": float(det_acc),
    }


def simple_detect_method(method: str) -> tuple[str, float]:
    """Offline equivalent of preprocess.detect_http_method for events whose method is known."""
    m = (method or "").strip().upper()
    if m in METHOD_CODE_MAP or m in {"TRACE", "CONNECT", "PROPFIND", "REPORT", "MKCOL", "COPY", "MOVE"}:
        return m, 1.0
    if m.startswith("GET"):
        return "GET", 0.9
    if m.startswith("POST"):
        return "POST", 0.9
    return "UNKNOWN", 0.0
