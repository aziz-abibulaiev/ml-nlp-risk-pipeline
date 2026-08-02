"""Shared normalization — the SAME code runs in training and at inference (D16.1).

No offline/online divergence: `decode_fixed_point` and the field assembly here are imported
by both train.py and predict.py. The regex used elsewhere for auditing/explanation NEVER
appears here — this module only decodes and shapes text, it never matches attack tokens.
"""

from __future__ import annotations

import json
from urllib.parse import unquote_plus

from .config import DECODE_MAX_PASSES


def decode_fixed_point(s: str, max_passes: int = DECODE_MAX_PASSES) -> tuple[str, int]:
    """Apply unquote_plus repeatedly until the string stops changing or the cap is hit.

    Returns (decoded_string, passes_used). `passes_used` is the encoding_depth feature.
    Fixes the measured double-encoding blind spot (attacks that retain %XX after one pass).
    """
    if not s:
        return "", 0
    cur = s
    passes = 0
    for _ in range(max_passes):
        nxt = unquote_plus(cur)
        if nxt == cur:
            break
        cur = nxt
        passes += 1
    return cur, passes


def _clean(v) -> str:
    if v is None:
        return ""
    if not isinstance(v, str):
        v = str(v)
    return v


def headers_get(headers, *names) -> str:
    """Case-insensitive header lookup. `headers` may be a dict or a JSON string."""
    if isinstance(headers, str):
        try:
            headers = json.loads(headers) if headers else {}
        except Exception:
            headers = {}
    if not isinstance(headers, dict):
        return ""
    low = {str(k).lower(): _clean(val) for k, val in headers.items()}
    for n in names:
        if n.lower() in low and low[n.lower()]:
            return low[n.lower()]
    return ""


def assemble_fields(*, url_path_raw="", url_query="", request_body="", headers=None) -> dict:
    """Build the two decoded text fields + carry the raw request-target string.

    field_a: request target + body (where injection payloads live)
    field_b: user-agent + referer (headers channel, weighted separately per D16.3)
    raw_target: the undecoded url_path_raw+query+body, used for encoding_depth
    """
    raw_target = " ".join([_clean(url_path_raw), _clean(url_query), _clean(request_body)]).strip()
    ua = headers_get(headers, "user-agent")
    ref = headers_get(headers, "referer", "referrer")
    field_a, depth_a = decode_fixed_point(raw_target)
    field_b, depth_b = decode_fixed_point(" ".join([ua, ref]).strip())
    return {
        "field_a": field_a,
        "field_b": field_b,
        "raw_target": raw_target,
        "encoding_depth": max(depth_a, depth_b),
    }


def request_from_corpus_row(row) -> dict:
    """Adapt a corpus parquet row (dict-like) to the request shape both channels consume."""
    return assemble_fields(
        url_path_raw=row.get("url_path_raw", ""),
        url_query=row.get("url_query", ""),
        request_body=row.get("request_body", ""),
        headers=row.get("headers", ""),
    )
