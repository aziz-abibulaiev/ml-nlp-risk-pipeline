"""Content detector inference — the working-product surface (D16 deliverable).

`predict(method, path, query, body, headers)` returns:
  { is_attack, p_attack, family, family_confidence, explanation, channel:{p_text,p_numeric} }

The `explanation` field is a POST-HOC regex match, fully decoupled from the score: the
regex never enters a model feature or label (D10/D16.7). It is an analyst-facing hint about
*why* a flagged request looks malicious, computed only for display.
"""

from __future__ import annotations

import os
import re

import joblib

from . import config as C
from .model import ContentDetector  # noqa: F401 (needed for joblib unpickling)

# ---- post-hoc explanation rules (display only; NOT features, NOT labels) ----
_EXPLAIN_RULES = {
    "sqli": re.compile(r"(union\s+select|\bor\s+1=1\b|'\s*or\s*'1'='1|information_schema|sleep\s*\(|waitfor\s+delay|--\s|;\s*drop\s+table)", re.I),
    "xss": re.compile(r"(<script|onerror\s*=|onload\s*=|javascript:|<svg|<img[^>]+src|document\.cookie|alert\s*\()", re.I),
    "path_traversal": re.compile(r"(\.\./|\.\.\\|%2e%2e%2f|/etc/passwd|/etc/shadow|win\.ini|boot\.ini|php://)", re.I),
    "cmd_injection": re.compile(r"(;\s*(cat|ls|id|whoami|uname|nc|curl|wget)|\|\s*(id|sh|bash)|\$\(|`[^`]+`|&&\s*\w+)", re.I),
    "ssti": re.compile(r"(\{\{.*?\}\}|\$\{.*?\}|<%=|#\{.*?\}|\{%.*?%\})", re.I),
    "ssrf": re.compile(r"(169\.254\.169\.254|localhost|127\.0\.0\.1|metadata\.google|gopher://|dict://|file://|0\.0\.0\.0)", re.I),
    "nosql_injection": re.compile(r"(\$ne\b|\$gt\b|\$where\b|\$regex\b|\$in\b|\$exists\b)", re.I),
    "scan": re.compile(r"(/\.git|/\.env|/wp-admin|/phpinfo|/\.htpasswd|/actuator|/swagger|/\.aws)", re.I),
}

_MODEL = {"det": None, "family": None}


def _load():
    if _MODEL["det"] is None:
        _MODEL["det"] = joblib.load(os.path.join(C.MODEL_DIR, "content_detector.joblib"))
        fp = os.path.join(C.MODEL_DIR, "family_head.joblib")
        _MODEL["family"] = joblib.load(fp) if os.path.exists(fp) else None
    return _MODEL["det"], _MODEL["family"]


def _explain(field_a: str) -> list[str]:
    return [name for name, pat in _EXPLAIN_RULES.items() if pat.search(field_a)]


def predict(method: str = "", path: str = "", query: str = "", body: str = "",
            headers=None, fpr_budget: float = None) -> dict:
    det, family = _load()
    req = {"url_path_raw": path, "url_query": query, "request_body": body, "headers": headers}
    pt = float(det.p_text([req])[0])
    pn = float(det.p_numeric([req])[0])
    pc = float(det.p_content([req], [pt], [pn])[0])
    thr = det.threshold(fpr_budget or C.PRIMARY_FPR)
    is_attack = pc >= thr

    fam, fam_conf = None, 0.0
    if is_attack and family is not None:
        from .normalize import assemble_fields
        fa = assemble_fields(url_path_raw=path, url_query=query, request_body=body, headers=headers)["field_a"]
        X = family["vectorizer"].transform([fa])
        proba = family["clf"].predict_proba(X)[0]
        j = int(proba.argmax())
        fam = str(family["clf"].classes_[j]); fam_conf = float(proba[j])

    from .normalize import assemble_fields
    fa = assemble_fields(url_path_raw=path, url_query=query, request_body=body, headers=headers)["field_a"]
    return {
        "is_attack": bool(is_attack),
        "p_attack": pc,
        "family": fam,
        "family_confidence": fam_conf,
        "explanation": _explain(fa),
        "channel": {"p_text": pt, "p_numeric": pn},
    }


if __name__ == "__main__":
    import json
    for ex in [
        dict(method="GET", path="/users/v1/admin' OR 1=1-- -", query="", body=""),
        dict(method="GET", path="/search", query="q=<script>alert(1)</script>", body=""),
        dict(method="GET", path="/products", query="id=5", body="", headers={"User-Agent": "Mozilla/5.0"}),
        dict(method="GET", path="/blog/etc%252Fpasswd", query="", body=""),
    ]:
        print(json.dumps({"in": ex, "out": predict(**ex)}, indent=2))
