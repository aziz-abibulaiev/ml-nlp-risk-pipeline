import os
import re2 as re
import re as pyre   
import json
import math
import logging
import joblib
import uvicorn
import urllib.parse, html
import time
import asyncio

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","service":"nlp-inference","msg":"%(message)s"}',
)
log = logging.getLogger("nlp-inference")

NLP_THRESHOLD_RAW = os.getenv("NLP_THRESHOLD", "").strip()

PATH_TFIDF = os.getenv("NLP_TFIDF_LR_PATH", "/models/nlp/nlp_tfidf_lr.joblib")
PATH_CHAR  = os.getenv("NLP_CHAR_SGD_PATH", "/models/nlp/nlp_char_sgd.joblib")
META_TFIDF = Path(PATH_TFIDF).with_suffix(".meta.json")
META_CHAR  = Path(PATH_CHAR).with_suffix(".meta.json")

RULES_FILE = Path(os.getenv("NLP_RULES_FILE", "/app/rules/base_rules.json"))
RULES: Dict[str, Any] = {}
COMPILED_RULES: Dict[str, List] = {} 

# Prometheus metrics
NLP_SCORE_REQS = Counter("nlp_score_requests_total", "Total NLP /score requests", ["status"])
NLP_SCORE_LAT  = Histogram(
    "nlp_score_latency_seconds", "Latency of NLP /score",
    buckets=(0.002,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1,2,5)
)
NLP_MODEL_LAT  = Histogram(
    "nlp_model_infer_latency_seconds", "Per-model NLP inference latency", ["kind"],
    buckets=(0.001,0.003,0.007,0.015,0.03,0.06,0.12,0.25,0.5,1)
)
RULE_HITS = Counter(
    "nlp_rule_hits_total",
    "Rule-engine hits by attack and kind (strong/partial)",
    ["attack", "kind"]
)

TOPIC_MAP: Dict[str, str] = {
    "sql_injection": "SQL Injection",
    "nosql_injection": "NoSQL Injection",
    "ldap_injection": "LDAP Injection",
    "cmd_injection": "Command Injection",
    "xss": "Cross-Site Scripting (XSS)",
    "csrf": "Cross-Site Request Forgery",
    "ssrf": "Server-Side Request Forgery",
    "ssrf_sensitive": "Cloud Metadata SSRF",
    "auth_bruteforce": "Authentication Brute-Force",
    "credential_stuffing": "Credential Stuffing",
    "idor": "Broken Access Control (IDOR)",
    "misconfiguration": "Security Misconfiguration",
    "data_exposure": "Sensitive Data Exposure",
    "dos": "Denial of Service",
    "websocket": "WebSocket Abuse",
    "api_key_exposure": "API Key Exposure",
    "business_logic": "Business Logic Abuse",
}

NLP_MODEL_ERRS = Counter("nlp_model_infer_errors_total", "Per-model NLP inference errors", ["kind"])

NLP_MODEL_EXISTS = Gauge("nlp_model_file_exists", "NLP model file exists (0/1)", ["kind"])
NLP_MODEL_LOADED = Gauge("nlp_model_loaded", "NLP model loaded in memory (0/1)", ["kind"])


MODELS: Dict[str, Any] = {
    "tfidf_lr": None,
    "char_sgd": None,
}
LOADED: Dict[str, bool] = {"tfidf_lr": False, "char_sgd": False}

SUMMARY_MIN = float(os.getenv("NLP_SUMMARY_MIN", "0.5"))

_RULES_MTIME: float = 0.0
_RULES_LAST_CHECK: float = 0.0
_RULES_CHECK_INTERVAL: float = 5.0

PER_ATTACK_MODELS: Dict[str, Dict[str, Any]] = {
    "tfidf_lr": {},
    "char_sgd": {},
}
PER_ATTACK_META: Dict[str, Dict[str, Any]] = {
    "tfidf_lr": {},
    "char_sgd": {},
}

SEMAPHORE_LIMIT = int(os.getenv("SEMAPHORE_LIMIT", "20"))

_RE_MAX_REGEX_INPUT_LEN = 5000

# Helpers
def _normalize_text(s: str) -> str:
    if not s:
        return ""
    try:
        s = urllib.parse.unquote_plus(s)
    except Exception:
        pass
    s = html.unescape(s)
    s = s.replace("\\'", "'").replace('\\"', '"').replace("\\u0027", "'").replace("\\u003d", "=")
    return " ".join(s.split()).lower()

def _is_unsupported_by_re2(pat: str) -> bool:
    if not pat:
        return False
    # backreference like \1, \2, etc.
    if re.search(r"\\\d", pat):
        return True
    # named backref or backref-like: (?P<name>...), (?P=name)
    if re.search(r"\(\?P<", pat) or re.search(r"\(\?P=", pat):
        return True
    # lookaround constructs (positive/negative lookahead/lookbehind)
    if "(?=" in pat or "(?!" in pat or "(?<=" in pat or "(?<!" in pat:
        return True
    # conditional constructs or other PCRE-only tokens
    if "(?(" in pat:
        return True
    # very long/complex alternations could still be fine with re2, but we keep simple heuristics
    return False

def _load_rules():
    global RULES, COMPILED_RULES
    try:
        if not RULES_FILE.exists():
            log.warning("rules file not found: %s", str(RULES_FILE))
            RULES = {}
            COMPILED_RULES = {}
            return

        with RULES_FILE.open("r", encoding="utf-8") as f:
            RULES = json.load(f)

        COMPILED_RULES = {}
        ap = RULES.get("attack_patterns", {})
        for attack_type, group in ap.items():
            compiled = {"strong": [], "tokens": []}
            compiled["min_hits"] = int(group.get("min_hits", 1))
            compiled["divisor"]  = float(group.get("partial_norm_divisor", 3))
            
            for rx in group.get("strong", []) or []:
                try:
                    rx = str(rx)
                    if _is_unsupported_by_re2(rx):
                        compiled["strong"].append({"unsafe_pattern": rx})
                        log.info("rules: unsupported_by_re2 attack=%s pattern=%s", attack_type, rx[:200])
                    else:
                        compiled["strong"].append(re.compile(rx))
                except Exception:
                    log.exception("compile_rule_failed attack=%s rx=%s", attack_type, rx)
                    compiled["strong"].append({"unsafe_pattern": rx})

            tokens = group.get("partial_tokens", None)
            if tokens is None:
                tokens = group.get("tokens", []) or []
            compiled["tokens"] = [str(t).lower() for t in tokens]

            COMPILED_RULES[attack_type] = compiled

        try:
            cfg = RULES.get("config", {}) if RULES else {}
            re_rec = int(cfg.get("recommended_re_max_regex_input_len", 0) or 0)
            if re_rec > 0:
                global _RE_MAX_REGEX_INPUT_LEN
                _RE_MAX_REGEX_INPUT_LEN = re_rec
        except Exception:
            pass

        try:
            global _RULES_MTIME
            _RULES_MTIME = RULES_FILE.stat().st_mtime
        except Exception:
            _RULES_MTIME = 0.0

        log.info("loaded rules kinds=%s (re_max_len=%s)", list(COMPILED_RULES.keys()), _RE_MAX_REGEX_INPUT_LEN)
    except Exception:
        log.exception("load_rules_failed")
        RULES = {}
        COMPILED_RULES = {}

def _try_reload_rules():
    global _RULES_LAST_CHECK, _RULES_MTIME
    now = time.time()
    if now - _RULES_LAST_CHECK < _RULES_CHECK_INTERVAL:
        return
    _RULES_LAST_CHECK = now
    try:
        st = RULES_FILE.stat()
        if st.st_mtime != _RULES_MTIME:
            log.info("rules_file_changed mtime_old=%.0f mtime_new=%.0f", _RULES_MTIME, st.st_mtime)
            _load_rules()
    except FileNotFoundError:
        log.warning("rules_file_missing path=%s", str(RULES_FILE))
    except Exception:
        log.exception("rules_hot_reload_failed")

def _load_recommended_threshold() -> Optional[float]:
    recs = []
    for p in (META_TFIDF, META_CHAR):
        try:
            if p.exists():
                with p.open("r") as f:
                    meta = json.load(f)
                rt = meta.get("recommended_threshold", None)
                if isinstance(rt, (int, float)):
                    recs.append(float(rt))
        except Exception:
            log.exception('nlp_meta_read_failed path="%s"', str(p))
    if not recs:
        return None
    return max(recs)

def _effective_threshold() -> float:
    if NLP_THRESHOLD_RAW:
        try:
            return float(NLP_THRESHOLD_RAW)
        except Exception:
            log.warning('bad_env_threshold value="%s"', NLP_THRESHOLD_RAW)
    rt = _load_recommended_threshold()
    return float(rt) if isinstance(rt, (float, int)) else 0.60

def _summarize_per_models(per: List[Dict[str, Any]], min_score: float = SUMMARY_MIN) -> List[Dict[str, Any]]:
    if not per:
        return []

    best_pa = {"tfidf_lr": None, "char_sgd": None}
    any_items = {"tfidf_lr": None, "char_sgd": None}

    for item in per:
        kind = str(item.get("kind", ""))
        s = float(item.get("score", 0.0))

        # ANY models
        if kind in ("tfidf_lr", "char_sgd"):
            if any_items[kind] is None or s > any_items[kind]["score"]:
                any_items[kind] = {"kind": kind, "score": s}
            continue

        # per-attack models
        if "__" in kind:
            fam, _atk = kind.split("__", 1)
            if fam in ("tfidf_lr", "char_sgd") and s >= min_score:
                prev = best_pa[fam]
                if prev is None or s > prev["score"]:
                    best_pa[fam] = {"kind": kind, "score": s}

    out = []
    if best_pa["tfidf_lr"]:
        out.append(best_pa["tfidf_lr"])
    if best_pa["char_sgd"]:
        out.append(best_pa["char_sgd"])
    if any_items["char_sgd"]:
        out.append(any_items["char_sgd"])
    if any_items["tfidf_lr"]:
        out.append(any_items["tfidf_lr"])

    # sort by score descending and remove duplicates
    out = sorted(out, key=lambda x: float(x["score"]), reverse=True)
    seen = set()
    compact = []
    for it in out:
        if it["kind"] in seen:
            continue
        seen.add(it["kind"])
        compact.append({"kind": it["kind"], "score": float(it["score"])})
    return compact

def _count_contains(blob: str, patterns: List[str]) -> int:
    c = 0
    for p in patterns:
        if p in blob:
            c += 1
    return c

def _load_one_model(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    obj = joblib.load(p)

    pipe = obj.get("pipeline")
    if pipe is not None:
        return {"pipeline": pipe}

    vec = obj.get("vectorizer") or obj.get("vect")
    clf = obj.get("clf") or obj.get("model")
    if vec is None or clf is None:
        return None
    return {"vectorizer": vec, "clf": clf}

def _load_any_models():
    # TFIDF ANY
    m = _load_one_model(PATH_TFIDF)
    MODELS["tfidf_lr"] = m
    NLP_MODEL_EXISTS.labels("tfidf_lr").set(1 if Path(PATH_TFIDF).exists() else 0)
    NLP_MODEL_LOADED.labels("tfidf_lr").set(1 if m else 0)
    log.info('nlp_model_loaded kind="tfidf_lr" path="%s"', PATH_TFIDF)

    # CHAR ANY
    m = _load_one_model(PATH_CHAR)
    MODELS["char_sgd"] = m
    NLP_MODEL_EXISTS.labels("char_sgd").set(1 if Path(PATH_CHAR).exists() else 0)
    NLP_MODEL_LOADED.labels("char_sgd").set(1 if m else 0)
    log.info('nlp_model_loaded kind="char_sgd" path="%s"', PATH_CHAR)

def _meta_threshold_for(path: str, default: float = 0.6) -> float:
    meta_p = Path(path).with_suffix(".meta.json")
    if not meta_p.exists():
        return default
    try:
        meta = json.loads(meta_p.read_text())
        rt = meta.get("recommended_threshold", None)
        return float(rt) if isinstance(rt, (int, float)) else default
    except Exception:
        return default

def _discover_per_attack_from_meta() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for meta_path in (META_TFIDF, META_CHAR):
        try:
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            pa = meta.get("per_attack") or {}
            for atk, m in pa.items():
                cur = out.setdefault(atk, {})
                if "tfidf_lr" in m:
                    cur["tfidf_lr"] = m["tfidf_lr"]
                if "char_sgd" in m:
                    cur["char_sgd"] = m["char_sgd"]
        except Exception:
            log.exception("per_attack_meta_read_failed path=%s", str(meta_path))
    return out

def _discover_per_attack_by_glob() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    base = Path("/models/nlp")
    for p in base.glob("nlp_tfidf_lr__*.joblib"):
        atk = p.stem.split("__", 1)[1]
        out.setdefault(atk, {})["tfidf_lr"] = str(p)
    for p in base.glob("nlp_char_sgd__*.joblib"):
        atk = p.stem.split("__", 1)[1]
        out.setdefault(atk, {})["char_sgd"] = str(p)
    return out

def _load_per_attack_models() -> None:
    index = _discover_per_attack_from_meta()
    if not index:
        index = _discover_per_attack_by_glob()

    total = 0
    for atk, mapping in index.items():
        if "tfidf_lr" in mapping:
            m = _load_one_model(mapping["tfidf_lr"])
            if m:
                PER_ATTACK_MODELS["tfidf_lr"][atk] = m
                PER_ATTACK_META["tfidf_lr"][atk] = {"threshold": _meta_threshold_for(mapping["tfidf_lr"], 0.6)}
                NLP_MODEL_EXISTS.labels(f"tfidf_lr__{atk}").set(1)
                NLP_MODEL_LOADED.labels(f"tfidf_lr__{atk}").set(1)
                total += 1
        if "char_sgd" in mapping:
            m = _load_one_model(mapping["char_sgd"])
            if m:
                PER_ATTACK_MODELS["char_sgd"][atk] = m
                PER_ATTACK_META["char_sgd"][atk] = {"threshold": _meta_threshold_for(mapping["char_sgd"], 0.6)}
                NLP_MODEL_EXISTS.labels(f"char_sgd__{atk}").set(1)
                NLP_MODEL_LOADED.labels(f"char_sgd__{atk}").set(1)
                total += 1
    log.info("per_attack_models_loaded total=%d kinds=%s", total, list(set([k for k in PER_ATTACK_MODELS["tfidf_lr"].keys()] + [k for k in PER_ATTACK_MODELS["char_sgd"].keys()])))

def extract_topics(signals: Dict[str, float], min_score: float = 0.5, top_k: int = 5) -> List[Dict[str, float]]:
    """Extract top attack topics from signals using max(strong, partial) per attack type."""
    scored = []
    for k, human in TOPIC_MAP.items():
        s = max(float(signals.get(f"{k}_strong", 0.0)), float(signals.get(f"{k}_partial", 0.0)))
        if s >= min_score:
            scored.append({"topic": human, "score": round(s, 4), "key": k})
    scored.sort(key=lambda x: x["score"], reverse=True)
    out = [{"topic": x["topic"], "score": x["score"]} for x in scored[:top_k]]
    return out

def detect_attack_signals(tf: Dict[str, Any], blob: str) -> Tuple[Dict[str, float], float]:
    signals: Dict[str, float] = {}
    if not isinstance(blob, str):
        blob = str(blob or "")
    lb = blob.lower()

    if not COMPILED_RULES:
        fallback_signals = {}
        kw_hits = 0
        for tok in ("select","union","onerror=","document.cookie"):
            if tok in lb:
                kw_hits += 1
        base = min(1.0, kw_hits / 3.0)
        return {}, base

    strong_any = 0.0
    partial_any = 0.0
    kw_total = 0
    payload_len = float(min(len(lb), 30000))

    for atk, spec in COMPILED_RULES.items():
        s_strong = 0.0
        s_partial = 0.0

        tokens = spec.get("tokens", []) or []
        if tokens:
            cnt = _count_contains(lb, tokens)
            min_hits = int(spec.get("min_hits", 1))
            divisor  = float(spec.get("divisor", 3))
            if cnt >= min_hits:
                s_partial = min(1.0, cnt / max(1.0, divisor))
            else:
                s_partial = 0.0
        for patt in spec.get("strong", []):
            try:
                if isinstance(patt, dict) and patt.get("unsafe_pattern"):
                    pstr = patt["unsafe_pattern"]
                    if len(lb) <= _RE_MAX_REGEX_INPUT_LEN:
                        try:
                            r = pyre.compile(pstr)
                            if r.search(lb):
                                s_strong = 1.0
                                break
                        except Exception:
                            continue
                    else:
                        continue
                else:
                    if hasattr(patt, "search"):
                        if len(lb) <= _RE_MAX_REGEX_INPUT_LEN:
                            if patt.search(lb):
                                s_strong = 1.0
                                break
                        else:
                            try:
                                plain = getattr(patt, "pattern", "")
                                words = pyre.findall(r"[A-Za-z0-9_]{3,}", plain)
                                if any(w.lower() in lb for w in words):
                                    s_partial = max(s_partial, 0.5)
                            except Exception:
                                pass
            except Exception:
                log.exception("regex_search_failed attack=%s", atk)
                continue

        signals[f"{atk}_strong"] = float(s_strong)
        signals[f"{atk}_partial"] = float(s_partial)
        try:
            if s_strong > 0.0:
                RULE_HITS.labels(atk, "strong").inc()
            if s_partial > 0.0:
                RULE_HITS.labels(atk, "partial").inc()
        except Exception:
            pass
        strong_any = max(strong_any, s_strong)
        partial_any = max(partial_any, s_partial)
        kw_total += _count_contains(lb, tokens)

    misc = RULES.get("misc", {})
    auth_tokens = [t.lower() for t in misc.get("auth_tokens", [])]
    misconf_tokens = [t.lower() for t in misc.get("misconf_tokens", [])]
    dos_tokens = [t.lower() for t in misc.get("dos_tokens", [])]

    cred_hits = _count_contains(lb, auth_tokens) if auth_tokens else 0
    denom_auth = max(1, len(auth_tokens) // 2) if auth_tokens else 1
    signals["auth_bruteforce_hint"] = float(min(1.0, cred_hits / denom_auth))

    signals["credential_stuffing_hint"] = 1.0 if ("password" in lb and ("'" in lb or '"' in lb) and "admin" in lb) else 0.0

    misconf_hits = _count_contains(lb, misconf_tokens) if misconf_tokens else 0
    denom_misconf = max(1, len(misconf_tokens) // 2) if misconf_tokens else 1
    signals["misconfig_hint"] = float(min(1.0, misconf_hits / denom_misconf))

    dos_hits = _count_contains(lb, dos_tokens) if dos_tokens else 0
    denom_dos = max(1, len(dos_tokens) // 2) if dos_tokens else 1
    signals["dos_hint_text"] = float(min(1.0, dos_hits / denom_dos))

    # aggregations
    signals["keyword_hits"] = float(kw_total)
    signals["payload_len"] = float(len(lb))
    signals["regex_skipped_due_to_len"] = float(1.0 if len(lb) > _RE_MAX_REGEX_INPUT_LEN else 0.0)

    signals["any_strong"] = float(strong_any)
    signals["any_partial"] = float(partial_any)
    
    if strong_any >= 1.0:
        base = min(1.0, 0.7 + 0.3 * strong_any)
    else:
        kw_norm = min(1.0, kw_total / 5.0)
        plen_norm = min(1.0, payload_len / 15000.0)
        base = min(1.0, 0.35 * partial_any + 0.25 * kw_norm + 0.1 * plen_norm)

    if (signals.get("api_key_exposed_hint", 0.0) or signals.get("api_key_strong", 0.0)) and partial_any > 0.0:
        base = min(1.0, base + 0.1)

    return signals, float(base)

def _compact_signals(signals: Dict[str, Any]) -> Dict[str, Any]:
    EXCLUDE = {"any_strong", "any_partial", "payload_len", "endpoint", "method"}
    out: Dict[str, Any] = {}
    for k, v in signals.items():
        if k in EXCLUDE:
            continue
        if isinstance(v, (int, float)):
            if v > 0:
                out[k] = round(float(v), 6)
        else:
            if v:
                out[k] = v
    return out

def _safe_get(tf: Dict[str, Any], key: str) -> str:
    v = tf.get(key, "")
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

def rules_scoring(tf: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    # getting raw fields
    payload = _safe_get(tf, "payload")
    headers = _safe_get(tf, "headers")
    endpoint = _safe_get(tf, "endpoint")
    error_text = _safe_get(tf, "error_text")
    method = _safe_get(tf, "method")

    # Combining and normalizing
    raw_blob = " ".join([payload, headers, endpoint, error_text, method])[:20000]
    blob = _normalize_text(raw_blob)

    # Detecting signals for all attack types
    signals, base_score = detect_attack_signals(tf, blob)

    signals["payload_len"] = float(len(payload))

    return base_score, _compact_signals(signals)

def suspicious_char_ratio(s: str) -> float:
    if not s:
        return 0.0
    bad = set("'\"<>%()\\=;")
    return sum(ch in bad for ch in s) / max(1, len(s))

def _score_per_attack(payload: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_entries: List[Dict[str, Any]] = []
    topic_scores: Dict[str, float] = {}

    text = payload or ""
    # TFIDF per-attack
    for atk, m in PER_ATTACK_MODELS["tfidf_lr"].items():
        try:
            if "pipeline" in m:
                proba = float(m["pipeline"].predict_proba([text])[0, 1])
            else:
                vec = m["vectorizer"]
                clf = m["clf"]
                X = vec.transform([text])
                proba = float(clf.predict_proba(X)[0, 1])
            per_entries.append({"kind": f"tfidf_lr__{atk}", "score": proba})
            thr = float(PER_ATTACK_META["tfidf_lr"].get(atk, {}).get("threshold", 0.6))
            if proba >= thr:
                topic_scores[atk] = max(topic_scores.get(atk, 0.0), proba)
        except Exception:
            NLP_MODEL_ERRS.labels("tfidf_lr__per_attack").inc()

    # CHAR per-attack
    for atk, m in PER_ATTACK_MODELS["char_sgd"].items():
        try:
            if "pipeline" in m:
                proba = float(m["pipeline"].predict_proba([text])[0, 1])
            else:
                vec = m["vectorizer"]
                clf = m["clf"]
                X = vec.transform([text])
                proba = float(clf.predict_proba(X)[0, 1])
            per_entries.append({"kind": f"char_sgd__{atk}", "score": proba})
            thr = float(PER_ATTACK_META["char_sgd"].get(atk, {}).get("threshold", 0.6))
            if proba >= thr:
                topic_scores[atk] = max(topic_scores.get(atk, 0.0), proba)
        except Exception:
            NLP_MODEL_ERRS.labels("char_sgd__per_attack").inc()

    # Preparing topics
    topics_pa: List[Dict[str, Any]] = []
    if topic_scores:
        items = sorted(topic_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        for atk, sc in items:
            topics_pa.append({"topic": TOPIC_MAP.get(atk, atk), "score": round(float(sc), 4)})
    return per_entries, topics_pa

def _score_one_body(body: Dict[str, Any]) -> Dict[str, Any]:
    try:
        tf = (body.get("text_fields") or {})
        payload = (tf.get("payload") or "")
        log_kind = (tf.get("log_kind") or "").lower()

        use_tfidf = LOADED.get("tfidf_lr", False) and (log_kind in ("access_log", "cloudtrail") or len(payload) >= 10)
        use_char  = LOADED.get("char_sgd", False) and (suspicious_char_ratio(payload) > 0.15 or len(payload) > 200)

        per = []

        # rules scoring
        with NLP_MODEL_LAT.labels("rules").time():
            base, sig = rules_scoring(tf)
        per.append({"kind": "rules", "score": base, "signals": sig})

        if use_tfidf:
            try:
                with NLP_MODEL_LAT.labels("tfidf_lr").time():
                    model = MODELS["tfidf_lr"]
                    if "pipeline" in model:
                        proba = float(model["pipeline"].predict_proba([payload])[0, 1])
                    else:
                        vec = model["vectorizer"]
                        X = vec.transform([payload])
                        proba = float(model["clf"].predict_proba(X)[0, 1])
                per.append({"kind": "tfidf_lr", "score": proba})
            except Exception:
                NLP_MODEL_ERRS.labels("tfidf_lr").inc()

        if use_char:
            try:
                with NLP_MODEL_LAT.labels("char_sgd").time():
                    model = MODELS["char_sgd"]
                    if "pipeline" in model:
                        proba = float(model["pipeline"].predict_proba([payload])[0, 1])
                    else:
                        vec = model["vectorizer"]
                        Xc = vec.transform([payload])
                        proba = float(model["clf"].predict_proba(Xc)[0, 1])
                per.append({"kind": "char_sgd", "score": proba})
            except Exception:
                NLP_MODEL_ERRS.labels("char_sgd").inc()
        pa_entries, topics_pa = _score_per_attack(payload)
        if pa_entries:
            per.extend(pa_entries)
            
        per_cut = _summarize_per_models(per, min_score=SUMMARY_MIN)
        nlp_score = max((p["score"] for p in per), default=0.0)
        thr_eff = _effective_threshold()
        topics = topics_pa if topics_pa else extract_topics(sig, min_score=0.5, top_k=5)

        resp = {
            "hash_id": body.get("hash_id"),
            "per_model": per_cut,
            "nlp_score": nlp_score,
            "threshold": thr_eff,
            "threshold_hit": nlp_score >= thr_eff,
            "topics": topics,
        }
        return JSONResponse(resp)
    except Exception:
        return {
            "hash_id": body.get("hash_id"),
            "per_model": [{"kind": "rules", "score": 0.0}],
            "nlp_score": 0.0,
            "threshold": _effective_threshold(),
            "threshold_hit": False,
            "used_models": [],
            "meta": {"error": "exception_fallback"},
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    for name, path in (("tfidf_lr", PATH_TFIDF), ("char_sgd", PATH_CHAR)):
        exists = Path(path).exists()
        NLP_MODEL_EXISTS.labels(name).set(1.0 if exists else 0.0)
        try:
            if exists:
                m = _load_one_model(path)
                if m:
                    MODELS[name] = m
                    LOADED[name] = True
                    NLP_MODEL_LOADED.labels(name).set(1.0)
                    log.info('nlp_model_loaded kind="%s" path="%s"', name, path)
                else:
                    LOADED[name] = False
                    NLP_MODEL_LOADED.labels(name).set(0.0)
            else:
                LOADED[name] = False
                NLP_MODEL_LOADED.labels(name).set(0.0)
        except Exception:
            LOADED[name] = False
            NLP_MODEL_LOADED.labels(name).set(0.0)
            log.exception("nlp_model_load_failed kind=%s", name)
    _load_any_models()
    _load_rules()
    _load_per_attack_models()
    yield

app = FastAPI(title="nlp-inference", version="1.2.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    exists = {"tfidf_lr": Path(PATH_TFIDF).exists(), "char_sgd": Path(PATH_CHAR).exists()}
    for k, v in exists.items():
        NLP_MODEL_EXISTS.labels(k).set(1.0 if v else 0.0)
        NLP_MODEL_LOADED.labels(k).set(1.0 if LOADED.get(k) else 0.0)

    thr_eff = _effective_threshold()
    return {
        "exists": exists,
        "loaded": LOADED,
        "threshold": thr_eff,
        "threshold_meta": _load_recommended_threshold(),
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/models")
def nlp_models():
    def _file_info(p: Path):
        try:
            st = p.stat()
            return {"size_bytes": int(st.st_size), "mtime": st.st_mtime}
        except Exception:
            return {"size_bytes": None, "mtime": None}

    items = {}
    for kind, job_path in (("tfidf_lr", PATH_TFIDF), ("char_sgd", PATH_CHAR)):
        p = Path(job_path)
        meta_p = p.with_suffix(".meta.json")
        meta = None
        try:
            if meta_p.exists():
                with meta_p.open("r") as f:
                    meta = json.load(f)
        except Exception:
            log.exception('nlp_meta_read_failed kind="%s" path="%s"', kind, str(meta_p))
        items[kind] = {
            "path": job_path,
            "exists": p.exists(),
            "loaded": LOADED.get(kind, False),
            "file": _file_info(p) if p.exists() else {"size_bytes": None, "mtime": None},
            "meta_path": str(meta_p),
            "meta": meta,
        }

    return {
        "effective_threshold": _effective_threshold(),
        "threshold_meta": _load_recommended_threshold(),
        "models": items,
    }


@app.post("/score")
def score(body: Dict[str,Any], x_trace_id: Optional[str] = Header(default=None, convert_underscores=False)):
    trace = x_trace_id or "-"
    _try_reload_rules()
    with NLP_SCORE_LAT.time():
        try:
            tf = (body.get("text_fields") or {})
            payload = (tf.get("payload") or "")
            log_kind = (tf.get("log_kind") or "").lower()

            # auto-switching models
            use_tfidf = LOADED.get("tfidf_lr", False) and (log_kind in ("access_log","cloudtrail") or len(payload) >= 10)
            use_char  = LOADED.get("char_sgd", False) and (suspicious_char_ratio(payload) > 0.15 or len(payload) > 200)

            per = []

            # rules scoring
            with NLP_MODEL_LAT.labels("rules").time():
                base, sig = rules_scoring(tf)
            per.append({"kind":"rules","score":base, "signals": sig})
            if use_tfidf:
                try:
                    with NLP_MODEL_LAT.labels("tfidf_lr").time():
                      vec = MODELS["tfidf_lr"]["vectorizer"]
                      X = vec.transform([payload])
                      proba = float(MODELS["tfidf_lr"]["clf"].predict_proba(X)[0,1])
                    per.append({"kind":"tfidf_lr","score":proba})
                except Exception:
                    NLP_MODEL_ERRS.labels("tfidf_lr").inc()
                    log.exception("tfidf_lr_infer_failed")

            if use_char:
                try:
                    with NLP_MODEL_LAT.labels("char_sgd").time():
                      vec = MODELS["char_sgd"]["vectorizer"]
                      Xc = vec.transform([payload])
                      proba = float(MODELS["char_sgd"]["clf"].predict_proba(Xc)[0, 1])
                    per.append({"kind":"char_sgd","score":proba})
                except Exception:
                    NLP_MODEL_ERRS.labels("char_sgd").inc()
                    log.exception("char_sgd_infer_failed")

            pa_entries, topics_pa = _score_per_attack(payload)
            if pa_entries:
                per.extend(pa_entries)

            topics = topics_pa if topics_pa else extract_topics(sig, min_score=0.5, top_k=5)
            try:
                m = MODELS.get("tfidf_lr")
                if m:
                    if "pipeline" in m:
                        proba = float(m["pipeline"].predict_proba([payload])[0, 1])
                    else:
                        vec = m["vectorizer"]; clf = m["clf"]
                        X = vec.transform([payload])
                        proba = float(clf.predict_proba(X)[0, 1])
                    per.append({"kind": "tfidf_lr", "score": proba})
            except Exception:
                NLP_MODEL_ERRS.labels("tfidf_lr").inc()
                log.exception("tfidf_lr_infer_failed")

            try:
                m = MODELS.get("char_sgd")
                if m:
                    if "pipeline" in m:
                        proba = float(m["pipeline"].predict_proba([payload])[0, 1])
                    else:
                        vec = m["vectorizer"]; clf = m["clf"]
                        Xc = vec.transform([payload])
                        proba = float(clf.predict_proba(Xc)[0, 1])
                    per.append({"kind": "char_sgd", "score": proba})
            except Exception:
                NLP_MODEL_ERRS.labels("char_sgd").inc()
                log.exception("char_sgd_infer_failed")
            nlp_score = max(p["score"] for p in per) if per else 0.0
            thr_eff = _effective_threshold()
            per_cut = _summarize_per_models(per, min_score=SUMMARY_MIN)
            resp = {
                "hash_id": body.get("hash_id"),
                "per_model": per_cut,
                "nlp_score": nlp_score,
                "threshold": thr_eff,
                "threshold_hit": nlp_score >= thr_eff,
                "topics": topics,
            }
            NLP_SCORE_REQS.labels("ok").inc()
            log.info('nlp_result trace_id="%s" hash="%s" score=%.4f', trace, resp["hash_id"], nlp_score)
            return JSONResponse(resp)

        except Exception as e:
            NLP_SCORE_REQS.labels("error").inc()
            log.exception(f'nlp_score_failed trace_id="{trace}" err="{e}"')
            return JSONResponse({
                "hash_id": body.get("hash_id"),
                "per_model": [{"kind": "rules", "score": 0.0}],
                "nlp_score": 0.0,
                "threshold": _effective_threshold(),
                "threshold_hit": False,
                "meta": {"error": "exception_fallback"},
            })

@app.post("/score_batch")
async def score_batch(
    bodies: List[Dict[str, Any]],
    x_trace_id: Optional[str] = Header(default=None, convert_underscores=False),
):
    trace = x_trace_id or "-"
    _try_reload_rules()

    items = bodies[:1000]

    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)

    async def _run_one(b: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            return await asyncio.to_thread(_score_one_body, b)

    results = await asyncio.gather(*(_run_one(b) for b in items))
    return JSONResponse(results)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))