import os
import re
import logging
import collections
import argparse
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from ml_http_features import FEATURE_COLS, METHOD_MAP, clean_text, featurize_http_request, normalize_url

log = logging.getLogger(__name__)

SOURCE_COL = "source"
SOURCE_MAHENDRA = "mahendra"
SOURCE_CSIC = "csic"
SOURCE_ATTACKLAB = "attacklab"

ML_DUP_GROUP_ROUND_DECIMALS = int(os.getenv("ML_DUP_GROUP_ROUND_DECIMALS", "6"))
ML_DATASET_SHUFFLE_SEED = int(os.getenv("ML_DATASET_SHUFFLE_SEED", "42"))
ML_NUM_MAX_DUP_PER_PATTERN_BENIGN = int(os.getenv("ML_NUM_MAX_DUP_PER_PATTERN_BENIGN", "2000"))
ML_NUM_MAX_DUP_PER_PATTERN_ATTACK = int(os.getenv("ML_NUM_MAX_DUP_PER_PATTERN_ATTACK", "0"))
ML_VAL_MAX_DUP_PER_PATTERN_BENIGN = int(os.getenv("ML_VAL_MAX_DUP_PER_PATTERN_BENIGN", "0"))
ML_VAL_MAX_DUP_PER_PATTERN_ATTACK = int(os.getenv("ML_VAL_MAX_DUP_PER_PATTERN_ATTACK", "0"))
ML_ENFORCE_VAL_TEST_DISJOINT = os.getenv("ML_ENFORCE_VAL_TEST_DISJOINT", "true").strip().lower() in (
    "1", "true", "yes", "on"
)

# -------------------------------------------------------------------
# Output paths
# -------------------------------------------------------------------

# Main ML train file (benign only)
TRAIN_OUT_PATH = os.getenv("ML_NUM_FILE", "./data/ml_numeric.csv")

# Validation / test file (attacks / malicious traffic)
VALID_OUT_PATH = os.getenv("ML_VALIDATION_FILE", "./data/ml_validation.csv")
TEST_OUT_PATH = os.getenv("ML_TEST_FILE", "./data/ml_test.csv")

# -------------------------------------------------------------------
# Kaggle / local config
# -------------------------------------------------------------------
# 1) Attacklab (optional extra validation)
USE_KAGGLE_ATTACKLAB = os.getenv("USE_KAGGLE_ATTACKLAB", "1") in (
    "1",
    "true",
    "True",
    "yes",
    "on",
)
ATTACKLAB_KAGGLE_FILE = os.getenv(
    "ATTACKLAB_KAGGLE_FILE",
    "attack-simulation-alert.csv",
)
LOCAL_ATTACKLAB_PATH = os.getenv("ATTACKLAB_LOCAL_PATH", "./data/attacklab.csv")
ATTACKLAB_MAX_ROWS = int(os.getenv("ATTACKLAB_MAX_ROWS", "500000"))
ATTACKLAB_VAL_ATTACK_ROWS = int(os.getenv("ATTACKLAB_VAL_ATTACK_ROWS", "20000"))
ATTACKLAB_VAL_ATTACK_OFFSET = int(os.getenv("ATTACKLAB_VAL_ATTACK_OFFSET", "0"))
ATTACKLAB_TEST_ATTACK_ROWS = int(os.getenv("ATTACKLAB_TEST_ATTACK_ROWS", "20000"))
ATTACKLAB_TEST_ATTACK_OFFSET = int(os.getenv("ATTACKLAB_TEST_ATTACK_OFFSET", "20000"))
ATTACKLAB_HTTP_ONLY = os.getenv("ATTACKLAB_HTTP_ONLY", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ATTACKLAB_STRATIFIED_SELECTION = os.getenv("ATTACKLAB_STRATIFIED_SELECTION", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ATTACKLAB_MAX_DUP_PER_PATTERN_SPLIT = int(os.getenv("ATTACKLAB_MAX_DUP_PER_PATTERN_SPLIT", "3"))
ATTACKLAB_POOL_MULTIPLIER = int(os.getenv("ATTACKLAB_POOL_MULTIPLIER", "80"))
ATTACKLAB_POOL_MIN_ROWS = int(os.getenv("ATTACKLAB_POOL_MIN_ROWS", "50000"))
ATTACKLAB_RELAX_DUP_CAP_FILL = os.getenv("ATTACKLAB_RELAX_DUP_CAP_FILL", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ATTACKLAB_UNIQUE_PATTERNS_ONLY = os.getenv("ATTACKLAB_UNIQUE_PATTERNS_ONLY", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# 2) Labeled web server logs (mahendradata)
USE_KAGGLE_LABELED = os.getenv("USE_KAGGLE_LABELED", "1") in (
    "1",
    "true",
    "True",
    "yes",
    "on",
)
LABELED_KAGGLE_FILE = os.getenv(
    "LABELED_KAGGLE_FILE",
    "labeled.csv",  # file name inside the Kaggle dataset
)
LOCAL_LABELED_PATH = os.getenv(
    "LABELED_LOCAL_PATH",
    "./data/web_server_access_logs_labeled.csv",
)

LABELED_MAX_ROWS = int(os.getenv("LABELED_MAX_ROWS", "500000"))
MAHENDRA_TRAIN_ROWS = int(os.getenv("MAHENDRA_TRAIN_ROWS", "500000"))
CSIC_TRAIN_BENIGN_ROWS = int(os.getenv("CSIC_TRAIN_BENIGN_ROWS", "50000"))
CSIC_TRAIN_BENIGN_OFFSET = int(os.getenv("CSIC_TRAIN_BENIGN_OFFSET", "0"))
TRAIN_ENFORCE_DISJOINT_WITH_VAL_TEST = os.getenv("TRAIN_ENFORCE_DISJOINT_WITH_VAL_TEST", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
TRAIN_DISJOINT_MODE = os.getenv("TRAIN_DISJOINT_MODE", "pattern").strip().lower()
TRAIN_SOURCE_BALANCE_ENABLED = os.getenv("TRAIN_SOURCE_BALANCE_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
TRAIN_SOURCE_MAX_RATIO = float(os.getenv("TRAIN_SOURCE_MAX_RATIO", "1.0"))
TRAIN_SOURCE_BALANCE_WARN_MIN_ROWS = int(os.getenv("TRAIN_SOURCE_BALANCE_WARN_MIN_ROWS", "10000"))
USE_KAGGLE_CSIC = os.getenv("USE_KAGGLE_CSIC", "1") in ("1", "true", "True", "yes", "on")
CSIC_KAGGLE_FILE = os.getenv("CSIC_KAGGLE_FILE", "csic_database.csv")
CSIC_LOCAL_PATH = os.getenv("CSIC_LOCAL_PATH", "./data/csic_database.csv")
# -------------------------------------------------------------------
# Shared helpers / constants
# -------------------------------------------------------------------

# HTTP headers present in Attacklab CSV
ATTACKLAB_HEADER_COLS = [
    "http.user_agent",
    "http.cookie",
    "http.host",
    "http.referer",
    "http.content_type",
    "http.content_length_header",
    "http.authorization",
    "http.connection",
    "http.location",
]

CSIC_HEADER_COLS = [
    "User-Agent",
    "Pragma",
    "Cache-Control",
    "Accept",
    "Accept-encoding",
    "Accept-charset",
    "language",
    "host",
    "cookie",
    "content-type",
    "connection",
]


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


def _cap_pattern_repeats_df(
    df: pd.DataFrame,
    cap_by_label: dict[int, int],
    tag: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["label"] = pd.to_numeric(work["label"], errors="coerce").fillna(0).astype(int)
    if all(int(cap_by_label.get(lbl, 0)) <= 0 for lbl in work["label"].unique().tolist()):
        return work

    # Shuffle once to avoid position bias when truncating repeated groups.
    work = work.sample(frac=1.0, random_state=ML_DATASET_SHUFFLE_SEED).reset_index(drop=True)

    keys = work[FEATURE_COLS].copy()
    for c in FEATURE_COLS:
        keys[c] = pd.to_numeric(keys[c], errors="coerce").round(ML_DUP_GROUP_ROUND_DECIMALS)
    keys["label"] = work["label"].astype(int)

    rank = keys.groupby(FEATURE_COLS + ["label"], sort=False).cumcount()
    caps = work["label"].map(lambda z: int(cap_by_label.get(int(z), 0)))
    keep = (caps <= 0) | (rank < caps)
    out = work.loc[keep].reset_index(drop=True)

    log.info(
        "[%s] pattern cap applied: rows_before=%s rows_after=%s dropped=%s cap_by_label=%s",
        tag,
        len(work),
        len(out),
        int(len(work) - len(out)),
        cap_by_label,
    )
    return out


def _postprocess_dataset_file(path: str, cap_by_label: dict[int, int], tag: str) -> None:
    if not os.path.exists(path):
        log.info("[%s] skip postprocess; file not found: %s", tag, path)
        return
    df = pd.read_csv(path, low_memory=False)
    if "label" not in df.columns:
        log.info("[%s] skip postprocess; no label column: %s", tag, path)
        return
    _log_pattern_stats(df, f"{tag}/before")
    out = _cap_pattern_repeats_df(df, cap_by_label=cap_by_label, tag=tag)
    _log_pattern_stats(out, f"{tag}/after")
    out.to_csv(path, index=False)
    log.info("[%s] saved postprocessed file: %s", tag, path)


# -------------------------------------------------------------------
# Attacklab: path, parsing, featurization (extra validation)
# -------------------------------------------------------------------

def get_attacklab_csv_path() -> str:
    """
    Resolve path to Attacklab CSV (validation). Kaggle by default, local if configured.
    """
    if USE_KAGGLE_ATTACKLAB:
        if kagglehub is None:
            raise RuntimeError("USE_KAGGLE_ATTACKLAB=1, but kagglehub is not installed")
        if not ATTACKLAB_KAGGLE_FILE:
            raise RuntimeError(
                "ATTACKLAB_KAGGLE_FILE must be set (relative path in dataset)"
            )

        root = kagglehub.dataset_download("moradrawashdeh/attack-simulation-lab")
        csv_path = os.path.join(root, ATTACKLAB_KAGGLE_FILE)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find {ATTACKLAB_KAGGLE_FILE} in {root}")
        return csv_path

    if not os.path.exists(LOCAL_ATTACKLAB_PATH):
        raise FileNotFoundError(
            f"ATTACKLAB_LOCAL_PATH does not exist: {LOCAL_ATTACKLAB_PATH}"
        )
    return LOCAL_ATTACKLAB_PATH


def label_from_attacklab_alert(alert_value):
    """
    Map alert to label:
      - "benign" -> 0
      - any other non-empty string -> 1
      - invalid / empty -> None
    """
    if not isinstance(alert_value, str):
        return None
    s = alert_value.strip().lower()
    if not s:
        return None
    return 0 if s == "benign" else 1


def _as_clean_text(v) -> str:
    if pd.isna(v):
        return ""
    return clean_text(v)


def _is_attacklab_http_row(row) -> bool:
    proto = _as_clean_text(row.get("frame.protocols")).lower()
    method = _as_clean_text(row.get("http.request.method")).upper()
    uri = _as_clean_text(row.get("http.request.uri"))
    full_uri = _as_clean_text(row.get("http.request.full_uri"))
    host = _as_clean_text(row.get("http.host"))

    has_http_proto = "http" in proto
    has_http_fields = bool(method or uri or full_uri or host)
    if not (has_http_proto or has_http_fields):
        return False

    # Drop generic transport-only records that collapse features to a constant.
    if not (uri or full_uri or method in METHOD_MAP or host):
        return False
    return True


def _attacklab_signature(row) -> str:
    """
    Build a coarse attack signature for stratified AttackLab sampling.
    """
    alert = _as_clean_text(row.get("alert")).lower() or "unknown"
    proto_raw = _as_clean_text(row.get("frame.protocols")).lower()
    proto_tokens = [t for t in re.split(r"[:/]", proto_raw) if t]
    proto_tokens = [
        t for t in proto_tokens
        if t not in {"eth", "ethertype", "ip", "ipv6", "tcp", "udp", "icmp", "arp"}
    ]
    proto_sig = proto_tokens[-1] if proto_tokens else "other"

    method = _as_clean_text(row.get("http.request.method")).upper() or "NA"
    raw_uri = _as_clean_text(row.get("http.request.uri")) or _as_clean_text(row.get("http.request.full_uri"))
    if raw_uri.startswith("http://") or raw_uri.startswith("https://"):
        parsed = urlparse(raw_uri)
    else:
        u = raw_uri if raw_uri.startswith("/") else f"/{raw_uri}" if raw_uri else "/"
        parsed = urlparse(f"http://x{u}")
    path = parsed.path or "/"
    segs = [s for s in path.split("/") if s]
    path_sig = segs[0].lower() if segs else "root"
    has_query = 1 if (parsed.query or ("?" in raw_uri)) else 0

    ctype = _as_clean_text(row.get("http.content_type")).lower()
    if ";" in ctype:
        ctype = ctype.split(";", 1)[0].strip()
    if not ctype:
        ctype = "none"
    return f"{alert}|{proto_sig}|{method}|{path_sig}|q{has_query}|ct={ctype}"


def build_url_from_attacklab_row(row) -> str:
    """Reconstruct URL from Attacklab row."""
    host = _as_clean_text(row.get("http.host"))
    uri = _as_clean_text(row.get("http.request.uri"))
    if not uri:
        uri = _as_clean_text(row.get("http.request.full_uri"))

    if uri.startswith("http://") or uri.startswith("https://"):
        return uri or "/"

    if host:
        if not uri:
            return f"http://{host}/"
        if uri.startswith("/"):
            return f"http://{host}{uri}"
        return f"http://{host}/{uri}"

    return uri or "/"


def featurize_attacklab_row(row) -> dict:
    """
    Extract numeric features from one Attacklab row for validation.
    """
    url = build_url_from_attacklab_row(row)
    headers_values = [_as_clean_text(row.get(col)) for col in ATTACKLAB_HEADER_COLS]
    body_text = (
        _as_clean_text(row.get("http.file_data"))
        or _as_clean_text(row.get("http.request.body"))
        or _as_clean_text(row.get("http.content_length_header"))
    )
    feats = featurize_http_request(
        url=url,
        method=row.get("http.request.method"),
        header_values=headers_values,
        body_text=body_text,
        content_type=row.get("http.content_type"),
        suspicious_extra_text=_as_clean_text(row.get("alert")),
    )

    lbl = label_from_attacklab_alert(row.get("alert"))

    return {
        **feats,
        "label": lbl,
        SOURCE_COL: SOURCE_ATTACKLAB,
    }


def _row_dict_to_pattern_key(rec: dict) -> str:
    vals = []
    for c in FEATURE_COLS:
        v = pd.to_numeric(pd.Series([rec.get(c)]), errors="coerce").iloc[0]
        if pd.isna(v):
            vals.append("nan")
        else:
            vals.append(str(round(float(v), ML_DUP_GROUP_ROUND_DECIMALS)))
    return "|".join(vals)


def _sample_attacklab_split_stratified(
    pool_df: pd.DataFrame,
    rows_req: int,
    offset: int,
    upper_bound: int | None,
    forbidden_pattern_keys: set[str],
    used_row_ids: set[int],
    random_state: int,
    tag: str,
) -> tuple[pd.DataFrame, set[str], set[int]]:
    if rows_req <= 0 or pool_df.empty:
        return pd.DataFrame(columns=(FEATURE_COLS + ["label", SOURCE_COL])), set(), used_row_ids

    cand = pool_df[pool_df["_eligible_idx"] >= int(max(0, offset))].copy()
    if upper_bound is not None:
        cand = cand[cand["_eligible_idx"] < int(upper_bound)]
    if cand.empty:
        log.warning("[%s] no candidates after offset=%s upper_bound=%s", tag, offset, upper_bound)
        return pd.DataFrame(columns=(FEATURE_COLS + ["label", SOURCE_COL])), set(), used_row_ids

    if used_row_ids:
        cand = cand[~cand["_row_id"].isin(used_row_ids)]
    if forbidden_pattern_keys:
        cand = cand[~cand["_pattern_key"].isin(forbidden_pattern_keys)]
    if cand.empty:
        log.warning("[%s] no candidates after disjoint filtering", tag)
        return pd.DataFrame(columns=(FEATURE_COLS + ["label", SOURCE_COL])), set(), used_row_ids

    if ATTACKLAB_STRATIFIED_SELECTION:
        cand = cand.sample(frac=1.0, random_state=int(random_state)).reset_index(drop=True)
    else:
        cand = cand.sort_values(by="_eligible_idx", kind="mergesort").reset_index(drop=True)

    by_sig = collections.defaultdict(list)
    for i, rec in cand.iterrows():
        by_sig[str(rec["_attack_sig"])].append(int(i))

    sig_order = pd.Series(list(by_sig.keys())).sample(frac=1.0, random_state=int(random_state)).tolist()
    sig_order = [str(s) for s in sig_order]

    selected_idx: list[int] = []
    selected_patterns = collections.Counter()

    # Round-robin over signatures: avoids first-N bias from dominant scenarios.
    while len(selected_idx) < rows_req and sig_order:
        progressed = False
        next_order = []
        for sig in sig_order:
            pool = by_sig.get(sig) or []
            pick = None
            while pool:
                idx = pool.pop()
                key = str(cand.at[idx, "_pattern_key"])
                if ATTACKLAB_MAX_DUP_PER_PATTERN_SPLIT > 0 and selected_patterns[key] >= ATTACKLAB_MAX_DUP_PER_PATTERN_SPLIT:
                    continue
                pick = idx
                break
            by_sig[sig] = pool

            if pick is not None:
                selected_idx.append(int(pick))
                key = str(cand.at[pick, "_pattern_key"])
                selected_patterns[key] += 1
                progressed = True
                if len(selected_idx) >= rows_req:
                    break

            if pool:
                next_order.append(sig)

        sig_order = next_order
        if not progressed:
            break

    # Optional fallback: relax per-pattern cap if strict selection under-filled.
    if len(selected_idx) < rows_req and ATTACKLAB_RELAX_DUP_CAP_FILL:
        already = set(selected_idx)
        for i in range(len(cand)):
            if i in already:
                continue
            selected_idx.append(i)
            already.add(i)
            if len(selected_idx) >= rows_req:
                break

    if len(selected_idx) < rows_req and not ATTACKLAB_RELAX_DUP_CAP_FILL:
        log.warning(
            "[%s] strict diversity mode kept pattern cap (requested=%s selected=%s cap=%s)",
            tag,
            int(rows_req),
            int(len(selected_idx)),
            int(ATTACKLAB_MAX_DUP_PER_PATTERN_SPLIT),
        )

    out = cand.iloc[selected_idx].copy()
    out_patterns = set(out["_pattern_key"].tolist())
    used_row_ids = set(used_row_ids) | set(out["_row_id"].astype(int).tolist())
    out = out[FEATURE_COLS + ["label", SOURCE_COL]].reset_index(drop=True)
    return out, out_patterns, used_row_ids


def _log_attacklab_diversity(df: pd.DataFrame, tag: str) -> None:
    if df.empty:
        log.info("[%s] empty split", tag)
        return
    keys = _pattern_key_series(df)
    uniq = int(keys.nunique())
    rows = int(len(df))
    dup_ratio = 1.0 - (float(uniq) / float(max(1, rows)))
    log.info("[%s] rows=%s unique_patterns=%s dup_ratio=%.4f", tag, rows, uniq, dup_ratio)


def _append_or_write(df: pd.DataFrame, path: str, override: bool, tag: str) -> None:
    if df.empty:
        log.info("[%s] no rows to write: %s", tag, path)
        return
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    if override or not os.path.exists(path):
        df.to_csv(path, index=False)
        log.info("[%s] wrote %s rows -> %s (override=%s)", tag, len(df), path, override)
        return

    head = pd.read_csv(path, nrows=0)
    missing = [c for c in df.columns if c not in head.columns]
    if missing:
        raise ValueError(f"[{tag}] existing {path} is missing columns: {missing}")
    df = df[head.columns]
    df.to_csv(path, mode="a", index=False, header=False)
    log.info("[%s] appended %s rows -> %s", tag, len(df), path)


def _pattern_key_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)
    x = df[FEATURE_COLS].copy()
    for c in FEATURE_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce").round(ML_DUP_GROUP_ROUND_DECIMALS)
    return x.astype(str).agg("|".join, axis=1)


def _row_key_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)
    cols = [c for c in (FEATURE_COLS + ["label", SOURCE_COL]) if c in df.columns]
    x = df[cols].copy()
    for c in FEATURE_COLS:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").round(ML_DUP_GROUP_ROUND_DECIMALS)
    if "label" in x.columns:
        x["label"] = pd.to_numeric(x["label"], errors="coerce").fillna(0).astype(int)
    return x.astype(str).agg("|".join, axis=1)


def _enforce_val_test_pattern_disjoint(prefer: str = "validation") -> dict:
    """
    Ensure no feature-pattern overlap between validation and test.
    prefer='validation' keeps validation intact and drops overlaps from test.
    """
    if (not os.path.exists(VALID_OUT_PATH)) or (not os.path.exists(TEST_OUT_PATH)):
        return {"skipped": "missing_file"}

    df_val = pd.read_csv(VALID_OUT_PATH, low_memory=False)
    df_test = pd.read_csv(TEST_OUT_PATH, low_memory=False)
    if df_val.empty or df_test.empty:
        return {"skipped": "empty_file"}

    key_val = _pattern_key_series(df_val)
    key_test = _pattern_key_series(df_test)
    overlap = set(key_val.tolist()) & set(key_test.tolist())
    if not overlap:
        return {"overlap_patterns": 0, "dropped_rows": 0}

    if prefer.strip().lower() == "test":
        keep_val = ~key_val.isin(overlap)
        dropped = int((~keep_val).sum())
        df_val = df_val.loc[keep_val].reset_index(drop=True)
        df_val.to_csv(VALID_OUT_PATH, index=False)
        return {"overlap_patterns": int(len(overlap)), "dropped_rows": dropped, "dropped_from": "validation"}

    keep_test = ~key_test.isin(overlap)
    dropped = int((~keep_test).sum())
    df_test = df_test.loc[keep_test].reset_index(drop=True)
    df_test.to_csv(TEST_OUT_PATH, index=False)
    return {"overlap_patterns": int(len(overlap)), "dropped_rows": dropped, "dropped_from": "test"}


def build_attacklab_splits(
    override_validation: bool = False,
    override_test: bool = False,
) -> None:
    """Append diversified AttackLab attack rows into calibration/test files."""
    val_rows_req = max(0, int(ATTACKLAB_VAL_ATTACK_ROWS))
    val_off = max(0, int(ATTACKLAB_VAL_ATTACK_OFFSET))
    test_rows_req = max(0, int(ATTACKLAB_TEST_ATTACK_ROWS))
    test_off = max(0, int(ATTACKLAB_TEST_ATTACK_OFFSET))
    cols = FEATURE_COLS + ["label", SOURCE_COL]

    # Fast path: no AttackLab rows requested for either split.
    if val_rows_req <= 0 and test_rows_req <= 0:
        log.info(
            "[attacklab] skip: ATTACKLAB_VAL_ATTACK_ROWS=%s and ATTACKLAB_TEST_ATTACK_ROWS=%s",
            int(val_rows_req),
            int(test_rows_req),
        )
        return

    csv_path = get_attacklab_csv_path()
    chunk_size = int(os.getenv("ATTACKLAB_CHUNKSIZE", "50000"))
    max_rows = ATTACKLAB_MAX_ROWS

    # Respect already written CSIC splits so new AttackLab patterns stay disjoint
    # from the opposite split.
    existing_val_keys = set()
    existing_test_keys = set()
    if (not override_validation) and os.path.exists(VALID_OUT_PATH):
        df_prev_val = pd.read_csv(VALID_OUT_PATH, low_memory=False)
        existing_val_keys = set(_pattern_key_series(df_prev_val).tolist())
    if (not override_test) and os.path.exists(TEST_OUT_PATH):
        df_prev_test = pd.read_csv(TEST_OUT_PATH, low_memory=False)
        existing_test_keys = set(_pattern_key_series(df_prev_test).tolist())

    desired_total = max(0, int(val_rows_req + test_rows_req))
    pool_target = max(
        ATTACKLAB_POOL_MIN_ROWS,
        desired_total * max(1, ATTACKLAB_POOL_MULTIPLIER),
        int(max(val_off + val_rows_req, test_off + test_rows_req)),
    )
    pool_target = min(pool_target, int(max_rows))

    pool_records: list[dict] = []
    eligible_seen = 0
    row_id = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=True):
        for _, r in chunk.iterrows():
            lbl = label_from_attacklab_alert(r.get("alert"))
            if lbl != 1:
                continue
            if ATTACKLAB_HTTP_ONLY and (not _is_attacklab_http_row(r)):
                continue

            rec = featurize_attacklab_row(r)
            if rec.get("label") != 1:
                continue

            eligible_seen += 1
            if eligible_seen > max_rows:
                break

            rec["_pattern_key"] = _row_dict_to_pattern_key(rec)
            rec["_attack_sig"] = _attacklab_signature(r)
            rec["_eligible_idx"] = int(eligible_seen - 1)
            rec["_row_id"] = int(row_id)
            row_id += 1
            pool_records.append(rec)

            if len(pool_records) >= pool_target:
                break

        if eligible_seen >= max_rows or len(pool_records) >= pool_target:
            break

    if not pool_records:
        log.warning("[attacklab] no eligible attack rows after filtering")
        val_df = pd.DataFrame(columns=cols)
        test_df = pd.DataFrame(columns=cols)
    else:
        pool_df = pd.DataFrame(pool_records)
        if ATTACKLAB_UNIQUE_PATTERNS_ONLY:
            rows_before = int(len(pool_df))
            pool_df = (
                pool_df.sample(frac=1.0, random_state=ML_DATASET_SHUFFLE_SEED + 7)
                .drop_duplicates(subset=["_pattern_key"], keep="first")
                .reset_index(drop=True)
            )
            log.info(
                "[attacklab] unique_patterns_only enabled: rows_before=%s rows_after=%s",
                rows_before,
                int(len(pool_df)),
            )
        log.info(
            "[attacklab] candidate_pool rows=%s eligible_seen=%s uniq_patterns=%s uniq_signatures=%s http_only=%s",
            len(pool_df),
            eligible_seen,
            int(pool_df["_pattern_key"].nunique()),
            int(pool_df["_attack_sig"].nunique()),
            ATTACKLAB_HTTP_ONLY,
        )
        log.info(
            "[attacklab] top_signatures=%s",
            pool_df["_attack_sig"].value_counts().head(10).to_dict(),
        )

        used_row_ids: set[int] = set()
        val_forbidden = set(existing_test_keys)
        val_upper_bound = int(test_off) if int(test_off) > int(val_off) else None
        val_df, val_keys_local, used_row_ids = _sample_attacklab_split_stratified(
            pool_df=pool_df,
            rows_req=val_rows_req,
            offset=val_off,
            upper_bound=val_upper_bound,
            forbidden_pattern_keys=val_forbidden,
            used_row_ids=used_row_ids,
            random_state=ML_DATASET_SHUFFLE_SEED + 11,
            tag="attacklab/val",
        )

        test_forbidden = set(existing_val_keys) | set(val_keys_local)
        test_df, test_keys_local, used_row_ids = _sample_attacklab_split_stratified(
            pool_df=pool_df,
            rows_req=test_rows_req,
            offset=test_off,
            upper_bound=None,
            forbidden_pattern_keys=test_forbidden,
            used_row_ids=used_row_ids,
            random_state=ML_DATASET_SHUFFLE_SEED + 29,
            tag="attacklab/test",
        )

        overlap_local = set(val_keys_local) & set(test_keys_local)
        if overlap_local:
            key_test = _pattern_key_series(test_df)
            drop_mask = key_test.isin(overlap_local)
            dropped = int(drop_mask.sum())
            test_df = test_df.loc[~drop_mask].reset_index(drop=True)
            log.warning(
                "[attacklab] local val/test overlap patterns=%s dropped_from_test=%s",
                len(overlap_local),
                dropped,
            )

    if val_rows_req > 0 and val_df.shape[0] < val_rows_req:
        log.warning(
            "[attacklab] val requested=%s attack rows, got=%s",
            val_rows_req,
            int(val_df.shape[0]),
        )
    if test_rows_req > 0 and test_df.shape[0] < test_rows_req:
        log.warning(
            "[attacklab] test requested=%s attack rows, got=%s",
            test_rows_req,
            int(test_df.shape[0]),
        )

    _log_attacklab_diversity(val_df, "attacklab/val_selected")
    _log_attacklab_diversity(test_df, "attacklab/test_selected")

    _append_or_write(val_df, VALID_OUT_PATH, override_validation, "attacklab->ml_validation")
    _append_or_write(test_df, TEST_OUT_PATH, override_test, "attacklab->ml_test")


# -------------------------------------------------------------------
# Labeled web logs (mahendradata): path, featurization, train+validation
# -------------------------------------------------------------------

def get_labeled_csv_path() -> str:
    """
    Resolve path to web-server-access-logs-labeled CSV
    (mahendradata/web-server-access-logs-labeled).
    """
    if USE_KAGGLE_LABELED:
        if kagglehub is None:
            raise RuntimeError("USE_KAGGLE_LABELED=1, but kagglehub is not installed")
        if not LABELED_KAGGLE_FILE:
            raise RuntimeError(
                "LABELED_KAGGLE_FILE must be set (file name inside the Kaggle dataset)"
            )

        root = kagglehub.dataset_download(
            "mahendradata/web-server-access-logs-labeled"
        )
        csv_path = os.path.join(root, LABELED_KAGGLE_FILE)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Could not find {LABELED_KAGGLE_FILE} in {root}"
            )
        return csv_path

    if not os.path.exists(LOCAL_LABELED_PATH):
        raise FileNotFoundError(
            f"LABELED_LOCAL_PATH does not exist: {LOCAL_LABELED_PATH}"
        )
    return LOCAL_LABELED_PATH


def load_csic_df() -> pd.DataFrame:
    if USE_KAGGLE_CSIC:
        if not CSIC_KAGGLE_FILE:
            raise RuntimeError("CSIC_KAGGLE_FILE must be set (relative path in dataset)")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "ispangler/csic-2010-web-application-attacks",
            CSIC_KAGGLE_FILE,
        )
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("Failed to load CSIC dataset via Kaggle adapter")
        return df

    if not os.path.exists(CSIC_LOCAL_PATH):
        raise FileNotFoundError(f"CSIC_LOCAL_PATH does not exist: {CSIC_LOCAL_PATH}")
    return pd.read_csv(CSIC_LOCAL_PATH, low_memory=False)


def featurize_csic_row(row) -> dict:
    raw_url = str(row.get("URL", "") or "").strip()
    if raw_url:
        raw_url = raw_url.split()[0]
    url = normalize_url(raw_url, host=row.get("host"), default_host="csic.local")
    header_values = [row.get(c) for c in CSIC_HEADER_COLS]
    feats = featurize_http_request(
        url=url,
        method=row.get("Method"),
        header_values=header_values,
        body_text=row.get("content", ""),
        content_type=row.get("content-type"),
    )
    cls = row.get("classification")
    try:
        label = 0 if int(cls) == 0 else 1
    except Exception:
        label = None
    return {**feats, "label": label, SOURCE_COL: SOURCE_CSIC}


def featurize_labeled_row(row) -> dict:
    """
    Extract numeric features from a row of the labeled web logs dataset
    (mahendradata). Real labels are preserved and filtered by caller.

    Expected columns:
      - ip, time, method, url, protocol, status, size,
        referrer, user_agent, extra, no, label, type
    """
    raw_url = row.get("url") or "/"
    url = normalize_url(raw_url, default_host="mahendra.local")
    ref = _as_clean_text(row.get("referrer"))
    ua = _as_clean_text(row.get("user_agent"))
    body_text = _as_clean_text(row.get("extra"))
    feats = featurize_http_request(
        url=url,
        method=row.get("method"),
        header_values=[ref, ua],
        body_text=body_text,
        content_type=None,
    )

    # Preserve real label from dataset:
    #   0 / benign -> 0
    #   1 / non-benign -> 1
    label = None
    raw_label = row.get("label")
    if isinstance(raw_label, str):
        s = raw_label.strip().lower()
        if s in ("0", "benign", "normal"):
            label = 0
        elif s in ("1", "attack", "anomaly", "malicious"):
            label = 1
    elif pd.notna(raw_label):
        try:
            label = 0 if int(float(raw_label)) == 0 else 1
        except Exception:
            label = None

    if label is None:
        t = row.get("type")
        if isinstance(t, str):
            t = t.strip().lower()
            if t == "benign":
                label = 0
            elif t:
                label = 1

    return {
        **feats,
        "label": label,
        SOURCE_COL: SOURCE_MAHENDRA,
    }


def _collect_mahendra_benign_df() -> pd.DataFrame:
    target = max(0, int(MAHENDRA_TRAIN_ROWS))
    cols = FEATURE_COLS + ["label", SOURCE_COL]
    if target <= 0:
        log.info("[train-mix] MAHENDRA_TRAIN_ROWS=%s -> skip mahendra collection", target)
        return pd.DataFrame(columns=cols)

    csv_path = get_labeled_csv_path()
    chunk_size = int(os.getenv("LABELED_CHUNKSIZE", "100000"))
    rows = []
    collected = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=True):
        batch = [featurize_labeled_row(r) for _, r in chunk.iterrows()]
        df = pd.DataFrame(batch)[cols]
        df = df[df["label"] == 0]
        if df.empty:
            continue
        need = target - collected
        if need <= 0:
            break
        take = df.iloc[:need].copy()
        rows.append(take)
        collected += int(take.shape[0])
        if collected >= target:
            break
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
    log.info("[train-mix] collected mahendra benign rows=%s", int(out.shape[0]))
    return out


def _collect_csic_benign_df(
    forbidden_keys: set[str] | None = None,
    disjoint_mode: str = "pattern",
) -> pd.DataFrame:
    target = max(0, int(CSIC_TRAIN_BENIGN_ROWS))
    if target <= 0:
        return pd.DataFrame(columns=FEATURE_COLS + ["label", SOURCE_COL])
    df_raw = load_csic_df()
    batch = [featurize_csic_row(r) for _, r in df_raw.iterrows()]
    out = pd.DataFrame(batch)
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out = out[out["label"] == 0].reset_index(drop=True)

    # Keep train disjoint from val/test by filtering the full CSIC benign pool
    # before sampling. mode=pattern is strict anti-template-leakage;
    # mode=row keeps more rows and removes only exact row overlap.
    mode = (disjoint_mode or "pattern").strip().lower()
    if forbidden_keys:
        key_all = _row_key_series(out) if mode == "row" else _pattern_key_series(out)
        keep = ~key_all.isin(forbidden_keys)
        dropped = int((~keep).sum())
        out = out.loc[keep].reset_index(drop=True)
        log.info(
            "[train-mix] csic benign disjoint filter mode=%s kept=%s dropped_overlap=%s",
            mode,
            int(out.shape[0]),
            dropped,
        )

    # Deterministic shuffle to avoid order/offset bias from raw source ordering.
    if not out.empty:
        out = out.sample(frac=1.0, random_state=ML_DATASET_SHUFFLE_SEED).reset_index(drop=True)

    available_after_disjoint = int(len(out))

    off = max(0, int(CSIC_TRAIN_BENIGN_OFFSET))
    if off >= available_after_disjoint:
        log.warning("[train-mix] csic benign offset=%s out of range=%s", off, available_after_disjoint)
        return pd.DataFrame(columns=FEATURE_COLS + ["label", SOURCE_COL])
    out = out.iloc[off : off + target].copy()
    log.info(
        "[train-mix] collected csic benign rows=%s (offset=%s target=%s available_after_disjoint=%s)",
        int(out.shape[0]),
        off,
        target,
        available_after_disjoint,
    )
    return out


def _forbidden_train_disjoint_keys(mode: str = "pattern") -> set[str]:
    keys: set[str] = set()
    if not TRAIN_ENFORCE_DISJOINT_WITH_VAL_TEST:
        return keys
    mode = (mode or "pattern").strip().lower()
    for path in (VALID_OUT_PATH, TEST_OUT_PATH):
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            continue
        if mode == "off":
            continue
        if mode == "row":
            keys |= set(_row_key_series(df).tolist())
        else:
            keys |= set(_pattern_key_series(df).tolist())
    return keys


def _rebalance_train_sources(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebalance benign train rows by source so one domain does not dominate.
    Keeps at most TRAIN_SOURCE_MAX_RATIO between largest and smallest source
    among {mahendra, csic}. With ratio=1.0 this becomes 1:1 balancing.
    """
    if (not TRAIN_SOURCE_BALANCE_ENABLED) or train_df.empty or SOURCE_COL not in train_df.columns:
        return train_df
    if "label" not in train_df.columns:
        return train_df

    work = train_df.copy()
    work["label"] = pd.to_numeric(work["label"], errors="coerce").fillna(0).astype(int)
    benign = work[work["label"] == 0].copy()
    if benign.empty:
        return train_df

    src_counts = benign[SOURCE_COL].astype(str).value_counts().to_dict()
    relevant_sources = [s for s in (SOURCE_MAHENDRA, SOURCE_CSIC) if int(src_counts.get(s, 0)) > 0]
    if len(relevant_sources) < 2:
        log.warning(
            "[train-mix] source balancing skipped: available_sources=%s counts=%s",
            relevant_sources,
            src_counts,
        )
        return train_df

    minority = min(int(src_counts[s]) for s in relevant_sources)
    ratio = float(TRAIN_SOURCE_MAX_RATIO) if np.isfinite(TRAIN_SOURCE_MAX_RATIO) and TRAIN_SOURCE_MAX_RATIO > 0 else 1.0
    max_per_source = max(1, int(np.floor(minority * ratio)))

    balanced_parts: list[pd.DataFrame] = []
    for src in relevant_sources:
        part = benign[benign[SOURCE_COL].astype(str) == src]
        if len(part) > max_per_source:
            part = part.sample(n=max_per_source, random_state=ML_DATASET_SHUFFLE_SEED)
        balanced_parts.append(part)

    # Keep any extra benign sources (if present) capped to max_per_source as well.
    for src, cnt in src_counts.items():
        if src in relevant_sources:
            continue
        part = benign[benign[SOURCE_COL].astype(str) == str(src)]
        if len(part) > max_per_source:
            part = part.sample(n=max_per_source, random_state=ML_DATASET_SHUFFLE_SEED)
        balanced_parts.append(part)

    benign_bal = pd.concat(balanced_parts, ignore_index=True) if balanced_parts else benign.iloc[0:0].copy()
    non_benign = work[work["label"] != 0].copy()
    out = pd.concat([benign_bal, non_benign], ignore_index=True)
    out = out.sample(frac=1.0, random_state=ML_DATASET_SHUFFLE_SEED).reset_index(drop=True)

    log.info(
        "[train-mix] source balancing applied ratio<=%.3f rows_before=%s rows_after=%s src_before=%s src_after=%s",
        ratio,
        int(train_df.shape[0]),
        int(out.shape[0]),
        src_counts,
        out[out["label"] == 0][SOURCE_COL].astype(str).value_counts().to_dict(),
    )
    if int(out.shape[0]) < int(TRAIN_SOURCE_BALANCE_WARN_MIN_ROWS):
        log.warning(
            "[train-mix] balanced train is small (rows=%s < warn_min=%s). "
            "Increase CSIC_TRAIN_BENIGN_ROWS or reduce CSIC val/test benign slices.",
            int(out.shape[0]),
            int(TRAIN_SOURCE_BALANCE_WARN_MIN_ROWS),
        )
    return out


def build_labeled_train_benign(override: bool = False):
    """
    Build BENIGN train mix from Mahendra + CSIC with optional disjoint filtering
    against current validation/test splits.
    """
    os.makedirs(os.path.dirname(TRAIN_OUT_PATH) or ".", exist_ok=True)
    if os.path.exists(TRAIN_OUT_PATH) and (not override):
        log.info("[train-mix] train file already exists, skip (use --override-train to rebuild): %s", TRAIN_OUT_PATH)
        return

    disjoint_mode = (TRAIN_DISJOINT_MODE or "pattern").strip().lower()
    if disjoint_mode not in ("pattern", "row", "off"):
        log.warning("[train-mix] unknown TRAIN_DISJOINT_MODE=%s, fallback to pattern", disjoint_mode)
        disjoint_mode = "pattern"
    forbidden = _forbidden_train_disjoint_keys(mode=disjoint_mode)

    mah_df = _collect_mahendra_benign_df()
    csic_df = _collect_csic_benign_df(
        forbidden_keys=forbidden,
        disjoint_mode=disjoint_mode,
    )
    train_df = pd.concat([mah_df, csic_df], ignore_index=True)
    if train_df.empty:
        raise RuntimeError("[train-mix] no benign rows were collected for ml_train")

    if forbidden:
        keys = _row_key_series(train_df) if disjoint_mode == "row" else _pattern_key_series(train_df)
        keep = ~keys.isin(forbidden)
        dropped = int((~keep).sum())
        train_df = train_df.loc[keep].reset_index(drop=True)
        log.info(
            "[train-mix] dropped rows overlapping with val/test (mode=%s): %s",
            disjoint_mode,
            dropped,
        )

    train_df = _rebalance_train_sources(train_df)

    train_df = train_df.sample(frac=1.0, random_state=ML_DATASET_SHUFFLE_SEED).reset_index(drop=True)
    train_df.to_csv(TRAIN_OUT_PATH, index=False)
    log.info(
        "[train-mix] DONE rows=%s labels=%s sources=%s -> %s",
        int(train_df.shape[0]),
        train_df["label"].value_counts().to_dict(),
        train_df[SOURCE_COL].value_counts().to_dict(),
        TRAIN_OUT_PATH,
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build ML numeric features from labeled web logs "
            "(mahendradata) and optionally Attacklab."
        ),
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help=(
            "Legacy shortcut: overwrite all outputs touched by this script "
            "(train/validation/test according to enabled stages)."
        ),
    )
    parser.add_argument(
        "--override-train",
        action="store_true",
        help="Overwrite train output file before building benign train mix.",
    )
    parser.add_argument(
        "--override-validation",
        action="store_true",
        help="Overwrite validation output file before appending AttackLab attacks.",
    )
    parser.add_argument(
        "--override-test",
        action="store_true",
        help="Overwrite test output file before appending AttackLab attacks.",
    )
    parser.add_argument(
        "--no-attacklab",
        action="store_true",
        help="If set, skip appending AttackLab attacks to validation/test files.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Build benign train mix only (Mahendra + CSIC benign).",
    )
    parser.add_argument(
        "--attacklab-only",
        action="store_true",
        help="Append AttackLab attack splits only (do not build Mahendra train).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.train_only and args.attacklab_only:
        raise SystemExit("Cannot use --train-only and --attacklab-only together")

    override_train = bool(args.override or args.override_train)
    override_validation = bool(args.override or args.override_validation)
    override_test = bool(args.override or args.override_test)

    if not args.attacklab_only:
        log.info(
            "=== Building benign train mix (mahendra + csic, label=0 only) "
            "(override_train=%s) ===",
            override_train,
        )
        build_labeled_train_benign(override=override_train)

    if not args.no_attacklab and not args.train_only:
        log.info(
            "=== Appending AttackLab attack-only rows into validation/test "
            "(override_validation=%s override_test=%s) ===",
            override_validation,
            override_test,
        )
        build_attacklab_splits(
            override_validation=override_validation,
            override_test=override_test,
        )
    elif args.no_attacklab:
        log.info("Skipping AttackLab append because --no-attacklab was provided.")

    _postprocess_dataset_file(
        TRAIN_OUT_PATH,
        cap_by_label={
            0: ML_NUM_MAX_DUP_PER_PATTERN_BENIGN,
            1: ML_NUM_MAX_DUP_PER_PATTERN_ATTACK,
        },
        tag="train/ml_numeric",
    )
    if ML_VAL_MAX_DUP_PER_PATTERN_BENIGN > 0 or ML_VAL_MAX_DUP_PER_PATTERN_ATTACK > 0:
        _postprocess_dataset_file(
            VALID_OUT_PATH,
            cap_by_label={
                0: ML_VAL_MAX_DUP_PER_PATTERN_BENIGN,
                1: ML_VAL_MAX_DUP_PER_PATTERN_ATTACK,
            },
            tag="validation/ml_validation",
        )
        _postprocess_dataset_file(
            TEST_OUT_PATH,
            cap_by_label={
                0: ML_VAL_MAX_DUP_PER_PATTERN_BENIGN,
                1: ML_VAL_MAX_DUP_PER_PATTERN_ATTACK,
            },
            tag="test/ml_test",
        )
    else:
        log.info("[attacklab] de-dup caps disabled for validation/test (both caps <= 0)")

    if ML_ENFORCE_VAL_TEST_DISJOINT and (not args.train_only):
        disjoint_info = _enforce_val_test_pattern_disjoint(prefer="validation")
        log.info("[attacklab] enforced val/test disjoint: %s", disjoint_info)


if __name__ == "__main__":
    main()
