import argparse
import collections
import csv
import json
import logging
import math
import os
import re
import string
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd

try:
    import kagglehub
except Exception:
    kagglehub = None


log = logging.getLogger(__name__)

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
REQUEST_LINE_RE = re.compile(r"^\s*([A-Za-z]+)\s+(\S+)")
HEADER_LINE_RE = re.compile(r"^\s*([A-Za-z0-9\-]+)\s*:\s*(.*?)\s*$")
ENCODED_TOKEN_RE = re.compile(r"%[0-9A-Fa-f]{2}")

DEFAULT_OUT = os.getenv("ML_TRAIN_FILE", "/data/ml_train.csv")

USE_KAGGLE_HTTPPARAMS = os.getenv("USE_KAGGLE_HTTPPARAMS", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
HTTPPARAMS_KAGGLE_HANDLE = os.getenv(
    "HTTPPARAMS_KAGGLE_HANDLE",
    "evg3n1j/httpparamsdataset",
)
HTTPPARAMS_KAGGLE_FILE = os.getenv(
    "HTTPPARAMS_ML_TRAIN_FILE",
    "payload_train.csv",
)
HTTPPARAMS_LOCAL_PATH = os.getenv(
    "HTTPPARAMS_ML_LOCAL_PATH",
    "./data/HttpParamsDataset_payload_train.csv",
)
HTTPPARAMS_TEXT_COL = os.getenv("HTTPPARAMS_TEXT_COL", "payload")
HTTPPARAMS_LABEL_COL = os.getenv("HTTPPARAMS_LABEL_COL", "label")
HTTPPARAMS_ATTACK_COL = os.getenv("HTTPPARAMS_ATTACK_COL", "attack_type")

ML_TRAIN_BENIGN_ONLY = os.getenv("ML_TRAIN_BENIGN_ONLY", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ML_DUP_GROUP_ROUND_DECIMALS = int(os.getenv("ML_DUP_GROUP_ROUND_DECIMALS", "6"))
ML_DATASET_SHUFFLE_SEED = int(os.getenv("ML_DATASET_SHUFFLE_SEED", "42"))
ML_NUM_MAX_DUP_PER_PATTERN_BENIGN = int(os.getenv("ML_NUM_MAX_DUP_PER_PATTERN_BENIGN", "2000"))
ML_NUM_MAX_DUP_PER_PATTERN_ATTACK = int(os.getenv("ML_NUM_MAX_DUP_PER_PATTERN_ATTACK", "0"))
ML_TRAIN_MAX_DUP_PER_PATTERN_BENIGN = int(
    os.getenv("ML_TRAIN_MAX_DUP_PER_PATTERN_BENIGN", str(ML_NUM_MAX_DUP_PER_PATTERN_BENIGN))
)
ML_TRAIN_MAX_DUP_PER_PATTERN_ATTACK = int(
    os.getenv("ML_TRAIN_MAX_DUP_PER_PATTERN_ATTACK", str(ML_NUM_MAX_DUP_PER_PATTERN_ATTACK))
)


def _find_col(df: pd.DataFrame, primary: str, fallbacks: List[str]) -> Optional[str]:
    candidates = [primary] + fallbacks
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _resolve_csv_from_root(root: Path, file_name: str) -> Path:
    direct = root / file_name
    if direct.exists():
        return direct

    wanted = file_name.strip().lower()
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.name.lower() == wanted:
            return candidate
    raise FileNotFoundError(f"[httpparams-ml] CSV not found: {file_name} in {root}")


def load_httpparams_df() -> pd.DataFrame:
    if USE_KAGGLE_HTTPPARAMS:
        if kagglehub is None:
            raise RuntimeError(
                "USE_KAGGLE_HTTPPARAMS=1 but kagglehub is not installed."
            )
        log.info(
            "[httpparams-ml] downloading %s (file=%s)",
            HTTPPARAMS_KAGGLE_HANDLE,
            HTTPPARAMS_KAGGLE_FILE,
        )
        root = Path(kagglehub.dataset_download(HTTPPARAMS_KAGGLE_HANDLE))
        csv_path = _resolve_csv_from_root(root, HTTPPARAMS_KAGGLE_FILE)
    else:
        csv_path = Path(HTTPPARAMS_LOCAL_PATH)

    if not csv_path.exists():
        raise FileNotFoundError(f"[httpparams-ml] CSV not found at: {csv_path}")

    log.info("[httpparams-ml] loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    log.info("[httpparams-ml] loaded shape=%s", df.shape)
    return df


def _map_label_strict(v, attack_v=None) -> Optional[int]:
    """
    Returns:
      - 0 for benign labels,
      - 1 for attack labels,
      - None when label cannot be determined confidently.
    """
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("norm", "benign", "normal", "0"):
            return 0
        if s in ("anom", "anomaly", "attack", "malicious", "1"):
            return 1
        if s in ("sqli", "sql_injection", "sql", "xss", "cmdi", "cmd_injection", "path-traversal", "path_traversal"):
            return 1
    else:
        try:
            iv = int(v)
            return 0 if iv == 0 else 1
        except Exception:
            pass

    # fallback to attack/category if available
    if isinstance(attack_v, str):
        a = attack_v.strip().lower()
        if a in ("norm", "benign", "normal", "none", "0"):
            return 0
        if a in ("anom", "anomaly", "attack", "malicious", "generic", "1"):
            return 1
        if a:
            return 1

    return None


def char_entropy(s: str) -> float:
    if not s:
        return 0.0
    cnt = collections.Counter(s)
    n = len(s)
    return float(-sum((c / n) * math.log2(c / n) for c in cnt.values()))


def suspicious_char_ratio_from_string(s: str) -> float:
    if not s:
        return 0.0
    sus = sum(ch in SUSPICIOUS_CHARS for ch in s)
    return float(sus / max(1, len(s)))


def digit_ratio(s: str) -> float:
    if not s:
        return 0.0
    digs = sum(ch.isdigit() for ch in s)
    return float(digs / max(1, len(s)))


def upper_ratio(s: str) -> float:
    if not s:
        return 0.0
    ups = sum(ch.isupper() for ch in s)
    return float(ups / max(1, len(s)))


def punct_ratio(s: str) -> float:
    if not s:
        return 0.0
    p = sum(ch in string.punctuation for ch in s)
    return float(p / max(1, len(s)))


def pct_encoded_ratio(s: str) -> float:
    if not s:
        return 0.0
    units = len(ENCODED_TOKEN_RE.findall(s)) + s.count("+")
    return float(units / max(1, len(s)))


def _looks_like_query(raw: str) -> bool:
    if "=" not in raw:
        return False
    if raw.startswith("{") or raw.startswith("["):
        return False
    if " " in raw and "http" not in raw:
        return False
    return True


def _split_head_body(payload: str) -> tuple[str, str]:
    text = str(payload or "")
    # Some corpora keep escaped CR/LF as plain text ("\\r\\n").
    if "\\r\\n" in text or "\\n" in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    if "\r\n\r\n" in text:
        a, b = text.split("\r\n\r\n", 1)
        return a, b
    if "\n\n" in text:
        a, b = text.split("\n\n", 1)
        return a, b
    return text, ""


def _extract_request_parts(payload: str) -> tuple[str, str, dict, str]:
    """
    Returns:
      method, target, headers_dict, body
    """
    head, body = _split_head_body(payload)
    lines = [ln.strip() for ln in head.replace("\r", "").split("\n") if ln.strip()]
    method = "UNKNOWN"
    target = "/"
    headers: dict[str, str] = {}

    if lines:
        m = REQUEST_LINE_RE.match(lines[0])
        if m:
            method = m.group(1).upper()
            target = m.group(2)
            for ln in lines[1:]:
                hm = HEADER_LINE_RE.match(ln)
                if hm:
                    headers[hm.group(1).lower()] = hm.group(2)
        else:
            raw = lines[0]
            if raw.startswith("http://") or raw.startswith("https://"):
                target = raw
            elif raw.startswith("/"):
                target = raw
            elif _looks_like_query(raw):
                target = "/?" + raw

            for ln in lines[1:]:
                hm = HEADER_LINE_RE.match(ln)
                if hm:
                    headers[hm.group(1).lower()] = hm.group(2)

    # If there was no clear request-line and no headers, treat full payload as value.
    if method == "UNKNOWN" and target == "/" and not headers:
        raw = str(payload or "").strip()
        if raw.startswith("http://") or raw.startswith("https://"):
            target = raw
        elif raw.startswith("/"):
            target = raw
        elif _looks_like_query(raw):
            target = "/?" + raw
        body = raw

    return method, target, headers, body


def _is_json_text(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    if not ((s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))):
        return 0
    try:
        json.loads(s)
        return 1
    except Exception:
        return 0


def featurize_payload(payload: str) -> dict:
    text = str(payload or "")
    method, target, headers, body = _extract_request_parts(text)

    if target.startswith("http://") or target.startswith("https://"):
        parsed = urlparse(target)
    else:
        if not target.startswith("/"):
            target = "/" + target
        parsed = urlparse("http://local" + target)

    path = parsed.path or "/"
    query = parse_qs(parsed.query, keep_blank_values=True)
    query_param_count = len(query)
    query_str = parsed.query or ""

    endpoint = path + (("?" + parsed.query) if parsed.query else "")
    method_code = METHOD_MAP.get(method, 0)
    payload_size = len(text.encode("utf-8"))
    headers_str = ";".join(f"{k}:{v}" for k, v in headers.items())
    ct = headers.get("content-type", "")
    is_json = 1 if ("json" in str(ct).lower()) else _is_json_text(body or text)
    query_values = [str(v or "") for vals in query.values() for v in vals]
    query_lengths = [len(v) for v in query_values]
    query_n = len(query_values)
    query_empty_n = sum(1 for v in query_values if not v)
    header_values = [str(v or "") for v in headers.values()]
    header_total_len = sum(len(v) for v in header_values)
    header_max_len = max((len(v) for v in header_values), default=0)
    encoded_text = f"{endpoint} {body or text}"

    return {
        "endpoint_len": float(len(path)),
        "header_count": float(len(headers)),
        "payload_size": float(payload_size),
        "headers_entropy": float(char_entropy(headers_str)),
        "query_param_count": float(query_param_count),
        "path_depth": float(sum(1 for seg in path.split("/") if seg)),
        "suspicious_char_ratio": float(suspicious_char_ratio_from_string(endpoint + " " + text + " " + headers_str)),
        "is_json_payload": float(is_json),
        "http_method_code": float(method_code),
        "query_len": float(len(query_str)),
        "query_entropy": float(char_entropy(query_str)),
        "query_value_len_mean": float(sum(query_lengths) / query_n) if query_n else 0.0,
        "query_value_len_max": float(max(query_lengths)) if query_lengths else 0.0,
        "empty_query_value_ratio": float(query_empty_n / query_n) if query_n else 0.0,
        "path_entropy": float(char_entropy(path)),
        "path_digit_ratio": float(digit_ratio(path)),
        "path_upper_ratio": float(upper_ratio(path)),
        "body_len": float(len(body)),
        "body_entropy": float(char_entropy(body)),
        "body_digit_ratio": float(digit_ratio(body)),
        "body_punct_ratio": float(punct_ratio(body)),
        "pct_encoded_ratio": float(pct_encoded_ratio(encoded_text)),
        "header_value_total_len": float(header_total_len),
        "header_value_max_len": float(header_max_len),
    }


def _log_pattern_stats(df: pd.DataFrame, tag: str) -> None:
    if df.empty:
        log.info("[%s] empty dataframe", tag)
        return

    x = df[FEATURE_COLS].copy()
    for c in FEATURE_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce").round(ML_DUP_GROUP_ROUND_DECIMALS)
    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    log.info(
        "[%s] rows=%s labels=%s unique_patterns=%s",
        tag,
        len(df),
        y.value_counts().to_dict(),
        len(x.drop_duplicates()),
    )
    for lbl in sorted(y.unique().tolist()):
        m = y == int(lbl)
        xx = x.loc[m]
        if xx.empty:
            continue
        vc = xx.astype(str).agg("|".join, axis=1).value_counts().head(3).tolist()
        dup_ratio = 1.0 - (len(xx.drop_duplicates()) / float(len(xx)))
        log.info(
            "[%s] label=%s rows=%s unique_patterns=%s dup_ratio=%.4f top3_counts=%s",
            tag,
            int(lbl),
            len(xx),
            len(xx.drop_duplicates()),
            dup_ratio,
            vc,
        )


def _cap_pattern_repeats_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["label"] = pd.to_numeric(work["label"], errors="coerce").fillna(0).astype(int)

    work = work.sample(frac=1.0, random_state=ML_DATASET_SHUFFLE_SEED).reset_index(drop=True)

    keys = work[FEATURE_COLS].copy()
    for c in FEATURE_COLS:
        keys[c] = pd.to_numeric(keys[c], errors="coerce").round(ML_DUP_GROUP_ROUND_DECIMALS)
    keys["label"] = work["label"].astype(int)

    rank = keys.groupby(FEATURE_COLS + ["label"], sort=False).cumcount()
    caps = work["label"].map(
        lambda z: ML_TRAIN_MAX_DUP_PER_PATTERN_BENIGN if int(z) == 0 else ML_TRAIN_MAX_DUP_PER_PATTERN_ATTACK
    )
    keep = (caps <= 0) | (rank < caps)
    out = work.loc[keep].reset_index(drop=True)

    log.info(
        "[httpparams-ml] pattern cap applied: rows_before=%s rows_after=%s dropped=%s",
        len(work),
        len(out),
        int(len(work) - len(out)),
    )
    return out


def build_ml_train_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    text_col = _find_col(
        df_raw,
        HTTPPARAMS_TEXT_COL,
        ["payload", "value", "param_value", "param", "text"],
    )
    if not text_col:
        raise ValueError(
            f"[httpparams-ml] payload column not found. df.columns={list(df_raw.columns)}"
        )

    label_col = _find_col(
        df_raw,
        HTTPPARAMS_LABEL_COL,
        ["label", "class", "y", "target"],
    )
    if not label_col:
        raise ValueError(
            f"[httpparams-ml] label column not found. df.columns={list(df_raw.columns)}"
        )

    attack_col = _find_col(
        df_raw,
        HTTPPARAMS_ATTACK_COL,
        ["attack_type", "attack", "type", "category"],
    )

    log.info(
        "[httpparams-ml] using columns: payload=%s label=%s attack_type=%s",
        text_col,
        label_col,
        attack_col,
    )

    work = pd.DataFrame({"payload": df_raw[text_col].astype(str)})
    if attack_col:
        work["attack_type"] = df_raw[attack_col].astype(str)
    else:
        work["attack_type"] = ""

    work["label"] = [
        _map_label_strict(v, a)
        for v, a in zip(df_raw[label_col].tolist(), work["attack_type"].tolist())
    ]

    before_labeled = len(work)
    work = work[work["label"].notna()].copy()
    dropped_unknown = before_labeled - len(work)
    if dropped_unknown > 0:
        log.info("[httpparams-ml] dropped rows with unknown label mapping: %s", dropped_unknown)
    work["label"] = work["label"].astype(int)
    work["payload"] = work["payload"].str.strip()
    work = work[work["payload"] != ""].reset_index(drop=True)

    if ML_TRAIN_BENIGN_ONLY:
        before = len(work)
        work = work[work["label"] == 0].reset_index(drop=True)
        log.info(
            "[httpparams-ml] benign-only train enabled: kept %s/%s rows",
            len(work),
            before,
        )

    if work.empty:
        raise SystemExit("[httpparams-ml] train set is empty after label mapping/filtering")

    feats = work["payload"].apply(featurize_payload)
    out = pd.DataFrame(feats.tolist())
    out["label"] = work["label"].astype(int).values
    out[SOURCE_COL] = SOURCE_HTTPPARAMS

    # Domain-fit diagnostics: how close this source looks to request-like runtime events.
    n = max(1, len(out))
    method_known_ratio = float((out["http_method_code"] > 0).sum()) / n
    has_headers_ratio = float((out["header_count"] > 0).sum()) / n
    has_query_ratio = float((out["query_param_count"] > 0).sum()) / n
    json_ratio = float((out["is_json_payload"] > 0).sum()) / n
    log.info(
        "[httpparams-ml] request-like stats: method_known=%.4f has_headers=%.4f has_query=%.4f json_payload=%.4f",
        method_known_ratio,
        has_headers_ratio,
        has_query_ratio,
        json_ratio,
    )
    if method_known_ratio < 0.01 and has_headers_ratio < 0.01 and has_query_ratio < 0.01:
        log.warning(
            "[httpparams-ml] source looks payload-only (very low request context). "
            "Use as supplemental benign train, not as the only numeric train source."
        )

    _log_pattern_stats(out, "httpparams-ml/before_cap")
    out = _cap_pattern_repeats_df(out)
    _log_pattern_stats(out, "httpparams-ml/after_cap")
    return out


def merge_with_existing(new_df: pd.DataFrame, out_path: Path, mode: str) -> pd.DataFrame:
    if mode == "override":
        return new_df

    if not out_path.exists():
        log.info("[httpparams-ml] append requested but file missing -> creating new file")
        return new_df

    old = pd.read_csv(out_path, low_memory=False)
    for col in FEATURE_COLS + ["label"]:
        if col not in old.columns:
            old[col] = 0
        if col not in new_df.columns:
            new_df[col] = 0
    merged = pd.concat(
        [old[FEATURE_COLS + ["label"]], new_df[FEATURE_COLS + ["label"]]],
        ignore_index=True,
    )
    merged = merged.drop_duplicates(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build numeric ML train dataset from HttpParamsDataset payload_train.csv."
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--mode",
        choices=["override", "append"],
        default="override",
        help="override: replace file, append: merge with existing train CSV",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("[httpparams-ml] start out=%s mode=%s", out_path, args.mode)

    df_raw = load_httpparams_df()
    df_new = build_ml_train_df(df_raw)
    df_final = merge_with_existing(df_new, out_path, args.mode)

    cols = FEATURE_COLS + ["label"]
    df_final = df_final[cols]
    df_final.to_csv(out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    label_counts = pd.to_numeric(df_final["label"], errors="coerce").fillna(0).astype(int).value_counts().to_dict()
    log.info(
        "[httpparams-ml] written rows=%s labels=%s path=%s",
        len(df_final),
        label_counts,
        out_path,
    )


if __name__ == "__main__":
    main()
