import collections
import math
import re
import string
from urllib.parse import parse_qs, urlparse


FEATURE_COLS = [
    "endpoint_len",
    "header_count",
    "payload_size",
    "headers_entropy",
    "query_param_count",
    "path_depth",
    "suspicious_char_ratio",
    "is_json_payload",
    "http_method_code",
    "query_len",
    "query_entropy",
    "query_value_len_mean",
    "query_value_len_max",
    "empty_query_value_ratio",
    "path_entropy",
    "path_digit_ratio",
    "path_upper_ratio",
    "body_len",
    "body_entropy",
    "body_digit_ratio",
    "body_punct_ratio",
    "pct_encoded_ratio",
    "header_value_total_len",
    "header_value_max_len",
]

METHOD_MAP = {
    "GET": 1,
    "POST": 2,
    "HEAD": 3,
    "PUT": 4,
    "DELETE": 5,
    "OPTIONS": 6,
    "TRACE": 7,
    "CONNECT": 8,
}

SUSPICIOUS_CHARS = set("<>\"'`;$(){}[]\\|&")

# Keep suspicious ratio stable for huge payloads.
MAX_SUSPICIOUS_TEXT_LEN = 8192
ENCODED_TOKEN_RE = re.compile(r"%[0-9A-Fa-f]{2}")


def clean_text(value) -> str:
    if isinstance(value, str):
        s = value.strip()
    else:
        s = str(value).strip() if value is not None else ""
    if not s:
        return ""
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    cnt = collections.Counter(text)
    n = len(text)
    return float(-sum((c / n) * math.log2(c / n) for c in cnt.values()))


def suspicious_char_ratio_from_string(text: str) -> float:
    if not text:
        return 0.0
    s = text[:MAX_SUSPICIOUS_TEXT_LEN]
    sus = sum(ch in SUSPICIOUS_CHARS for ch in s)
    return float(sus / max(1, len(s)))


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digs = sum(ch.isdigit() for ch in text)
    return float(digs / max(1, len(text)))


def upper_ratio(text: str) -> float:
    if not text:
        return 0.0
    ups = sum(ch.isupper() for ch in text)
    return float(ups / max(1, len(text)))


def punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    p = sum(ch in string.punctuation for ch in text)
    return float(p / max(1, len(text)))


def pct_encoded_ratio(text: str) -> float:
    if not text:
        return 0.0
    s = text[:MAX_SUSPICIOUS_TEXT_LEN]
    units = len(ENCODED_TOKEN_RE.findall(s)) + s.count("+")
    return float(units / max(1, len(s)))


def http_method_code(method_value) -> int:
    m = clean_text(method_value).upper()
    return METHOD_MAP.get(m, 0)


def is_json_payload(content_type_value) -> int:
    return 1 if "json" in clean_text(content_type_value).lower() else 0


def normalize_url(raw_url, host: str = "", default_host: str = "local") -> str:
    u = clean_text(raw_url)
    if u.startswith("http://") or u.startswith("https://"):
        return u
    if not u:
        u = "/"
    if not u.startswith("/"):
        u = "/" + u
    h = clean_text(host) or default_host
    return f"http://{h}{u}"


def featurize_http_request(
    *,
    url: str,
    method,
    header_values: list[str] | tuple[str, ...],
    body_text="",
    content_type=None,
    suspicious_extra_text="",
) -> dict:
    parsed = urlparse(clean_text(url) or "/")
    path = parsed.path or "/"
    query = parse_qs(parsed.query, keep_blank_values=True)
    query_str = parsed.query or ""

    headers_clean = [clean_text(v) for v in (header_values or [])]
    headers_clean = [v for v in headers_clean if v]
    headers_blob = " ".join(headers_clean)

    body = clean_text(body_text)

    payload_parts = [path]
    if parsed.query:
        payload_parts.append("?" + parsed.query)
    if body:
        payload_parts.append("BODY=" + body)
    if headers_blob:
        payload_parts.append("HDR=" + headers_blob)
    payload_text = " ".join(payload_parts)

    suspicious_text = payload_text
    extra = clean_text(suspicious_extra_text)
    if extra:
        suspicious_text = suspicious_text + " " + extra

    query_values = [clean_text(v) for vals in query.values() for v in vals]
    query_value_lengths = [len(v) for v in query_values]
    query_values_n = len(query_values)
    query_empty_n = sum(1 for v in query_values if not v)

    path_entropy = char_entropy(path)
    body_entropy = char_entropy(body)
    headers_total_len = sum(len(v) for v in headers_clean)
    headers_max_len = max((len(v) for v in headers_clean), default=0)
    encoded_text = " ".join([path, query_str, body])

    return {
        "endpoint_len": len(path),
        "header_count": int(len(headers_clean)),
        "payload_size": int(len(payload_text)),
        "headers_entropy": float(char_entropy(headers_blob)),
        "query_param_count": int(len(query)),
        "path_depth": int(sum(1 for seg in path.split("/") if seg)),
        "suspicious_char_ratio": float(suspicious_char_ratio_from_string(suspicious_text)),
        "is_json_payload": int(is_json_payload(content_type)),
        "http_method_code": int(http_method_code(method)),
        "query_len": int(len(query_str)),
        "query_entropy": float(char_entropy(query_str)),
        "query_value_len_mean": float(sum(query_value_lengths) / query_values_n) if query_values_n else 0.0,
        "query_value_len_max": float(max(query_value_lengths)) if query_value_lengths else 0.0,
        "empty_query_value_ratio": float(query_empty_n / query_values_n) if query_values_n else 0.0,
        "path_entropy": float(path_entropy),
        "path_digit_ratio": float(digit_ratio(path)),
        "path_upper_ratio": float(upper_ratio(path)),
        "body_len": int(len(body)),
        "body_entropy": float(body_entropy),
        "body_digit_ratio": float(digit_ratio(body)),
        "body_punct_ratio": float(punct_ratio(body)),
        "pct_encoded_ratio": float(pct_encoded_ratio(encoded_text)),
        "header_value_total_len": int(headers_total_len),
        "header_value_max_len": int(headers_max_len),
    }
