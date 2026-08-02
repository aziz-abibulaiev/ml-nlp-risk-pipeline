"""Build ML numeric features and NLP text artifacts from SR-BH 2020.

SR-BH 2020 (Sureda Riera et al., Computers & Security 2022, doi:10.1016/j.cose.2022.102788,
Harvard Dataverse doi:10.7910/DVN/OGOIXX, CC0 1.0) is 907,815 real HTTP requests captured
over 12 days in July 2020 by an internet-exposed WordPress honeypot running
ModSecurity 2.9.2 + CRS 3.3.0 in detection-only mode.

Ingestion only: nothing here trains, refits, recalibrates or overwrites a model, and no
existing file under data/ or models/ is touched.

=============================================================================
WARNING - THE CAPEC CLASS LABELS ARE NOT RELIABLE GROUND TRUTH
=============================================================================
The binary label (`000 - Normal` == 1 -> benign, else attack) is trustworthy: it was
manually and semi-automatically reviewed by the dataset authors.

The 13 CAPEC class columns are NOT. Measured on this file: only 42.7% of the rows
labelled `66 - SQL Injection` contain any SQL-injection marker (union select, or 1=1,
information_schema, ...). The remaining 57.3% are template injection ({{ ... }}),
path traversal (etc/passwd, c:\\) or SSI injection (<!--#EXEC cmd="ls /"-->) that
CRS 3.3.0 misattributed to SQLi.

The `capec_*` columns are carried through for provenance and exploratory work only.
They must not be used as multi-class ground truth without an explicit caveat.
The binary label is fine.
=============================================================================

Feature parity is a hard requirement: FEATURE_COLS and the extraction helpers are
imported from ml_http_features, never reimplemented, and the emitted column order is
asserted against models/ml/if_model.meta.json:feature_order before anything is written.

See docs/srbh2020_mapping.md for the field-mapping decisions and their consequences.
"""

import argparse
import json
import logging
import os
import re
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_http_features import FEATURE_COLS, featurize_http_request, normalize_url  # noqa: E402

log = logging.getLogger(__name__)

SOURCE_COL = "source"
SOURCE_SRBH = "srbh2020"
SEED = 42

SRBH_INPUT_PATH = os.getenv(
    "SRBH_INPUT_CSV",
    os.path.expanduser(
        "~/Desktop/навчання/Аспирантура/дисертація/джерела/datasets/data_capec_multilabel.csv"
    ),
)
SRBH_EXPECTED_MD5 = "173ec515308bdce5aec19cfd5b792596"
SRBH_OUT_DIR = os.getenv("SRBH_OUT_DIR", "./data/srbh2020")

FEATURE_ORDER_META = os.getenv("ML_IF_META", "./models/ml/if_model.meta.json")
FEATURE_ORDER_SIG = os.getenv("ML_FEATURE_ORDER_SIG", "./models/feature_order.sig")

CHUNK_ROWS = int(os.getenv("SRBH_CHUNK_ROWS", "50000"))
LOG_EVERY = int(os.getenv("SRBH_LOG_EVERY", "100000"))
SAMPLE_ROWS = int(os.getenv("SRBH_SAMPLE_ROWS", "50000"))

NORMAL_COL = "000 - Normal"
METHOD_COL = "request_http_method"
URI_COL = "request_http_request"
BODY_COL = "request_body"
CONTENT_TYPE_COL = "request_content_type"

# The 11 request header columns SR-BH exposes, in file order.
HEADER_COLS = [
    "request_user_agent",
    "request_referer",
    "request_host",
    "request_origin",
    "request_cookie",
    "request_content_type",
    "request_accept",
    "request_accept_language",
    "request_accept_encoding",
    "request_do_not_track",
    "request_connection",
]

# Wire-format header names, so text_headers reads like a real header blob rather than
# like SR-BH column identifiers.
HEADER_WIRE_NAMES = {
    "request_user_agent": "user-agent",
    "request_referer": "referer",
    "request_host": "host",
    "request_origin": "origin",
    "request_cookie": "cookie",
    "request_content_type": "content-type",
    "request_accept": "accept",
    "request_accept_language": "accept-language",
    "request_accept_encoding": "accept-encoding",
    "request_do_not_track": "dnt",
    "request_connection": "connection",
}

CAPEC_COLS = [
    "272 - Protocol Manipulation",
    "242 - Code Injection",
    "88 - OS Command Injection",
    "126 - Path Traversal",
    "66 - SQL Injection",
    "16 - Dictionary-based Password Attack",
    "310 - Scanning for Vulnerable Software",
    "153 - Input Data Manipulation",
    "248 - Command Injection",
    "274 - HTTP Verb Tampering",
    "194 - Fake the Source of Data",
    "34 - HTTP Response Splitting",
    "33 - HTTP Request Smuggling",
]

INT_FEATURES = {
    "endpoint_len",
    "header_count",
    "payload_size",
    "query_param_count",
    "path_depth",
    "is_json_payload",
    "http_method_code",
    "query_len",
    "body_len",
    "header_value_total_len",
    "header_value_max_len",
}

PCT_TOKEN_RE = re.compile(r"%[0-9A-Fa-f]{2}")


def capec_slug(col: str) -> str:
    s = col.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return "capec_" + s.strip("_")


CAPEC_OUT_COLS = [capec_slug(c) for c in CAPEC_COLS]


def assert_feature_parity() -> None:
    """Fail loudly if the emitted order diverges from what the trained models expect."""
    with open(FEATURE_ORDER_META, "r", encoding="utf-8") as fh:
        meta_order = json.load(fh)["feature_order"]
    if list(meta_order) != list(FEATURE_COLS):
        raise RuntimeError(
            f"feature order mismatch: {FEATURE_ORDER_META} has {meta_order}, "
            f"ml_http_features.FEATURE_COLS has {FEATURE_COLS}"
        )
    if len(FEATURE_COLS) != 24:
        raise RuntimeError(f"expected 24 features, got {len(FEATURE_COLS)}")
    if os.path.exists(FEATURE_ORDER_SIG):
        import hashlib

        with open(FEATURE_ORDER_SIG, "r", encoding="utf-8") as fh:
            recorded = fh.read().strip()
        # Same scheme as jobs/retrain/main.py:5872 and services/ml_infer/main.py::_feature_sig.
        computed = hashlib.sha1(",".join(FEATURE_COLS).encode("utf-8")).hexdigest()[:8]
        if computed != recorded:
            raise RuntimeError(
                f"feature_order signature mismatch: {FEATURE_ORDER_SIG} has {recorded}, computed {computed}"
            )
    log.info("[srbh] feature parity OK: %d features, order matches %s", len(FEATURE_COLS), FEATURE_ORDER_META)


def verify_md5(path: str, expected: str) -> None:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    got = h.hexdigest()
    if got != expected:
        raise RuntimeError(f"MD5 mismatch for {path}: expected {expected}, got {got}")
    log.info("[srbh] MD5 verified: %s", got)


def build_features_schema() -> pa.Schema:
    fields = []
    for c in FEATURE_COLS:
        fields.append(pa.field(c, pa.int64() if c in INT_FEATURES else pa.float64()))
    fields.append(pa.field("row_id", pa.int64()))
    fields.append(pa.field("label", pa.int64()))
    fields.append(pa.field(SOURCE_COL, pa.string()))
    for c in CAPEC_OUT_COLS:
        fields.append(pa.field(c, pa.int64()))
    return pa.schema(fields)


TEXT_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.int64()),
        pa.field("text_uri", pa.string()),
        pa.field("text_body", pa.string()),
        pa.field("text_headers", pa.string()),
        pa.field("label", pa.int64()),
    ]
)


def process_chunk(chunk: pd.DataFrame, row_offset: int, stats: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    from urllib.parse import unquote_plus

    feat_rows = []
    text_rows = []

    uris = chunk[URI_COL].tolist()
    bodies = chunk[BODY_COL].tolist()
    methods = chunk[METHOD_COL].tolist()
    ctypes = chunk[CONTENT_TYPE_COL].tolist()
    hosts = chunk["request_host"].tolist()
    header_cols = [chunk[c].tolist() for c in HEADER_COLS]
    header_names = [HEADER_WIRE_NAMES[c] for c in HEADER_COLS]
    normals = chunk[NORMAL_COL].tolist()
    capec_cols = [chunk[c].tolist() for c in CAPEC_COLS]

    for i in range(len(chunk)):
        raw_uri = uris[i]
        raw_body = bodies[i]
        method = methods[i]
        ctype = ctypes[i]

        header_pairs = [
            (header_names[j], col[i]) for j, col in enumerate(header_cols) if col[i]
        ]
        header_values = [v for _, v in header_pairs]
        host = hosts[i]

        try:
            url = normalize_url(raw_uri, host=host, default_host="srbh.local")
            feats = featurize_http_request(
                url=url,
                method=method,
                header_values=header_values,
                body_text=raw_body,
                content_type=ctype,
            )
        except ValueError:
            # Malformed authority (unbalanced IPv6 brackets, bad port) -> retry without host.
            stats["malformed_url_fallbacks"] += 1
            try:
                url = normalize_url(raw_uri, host="", default_host="srbh.local")
                feats = featurize_http_request(
                    url=url,
                    method=method,
                    header_values=header_values,
                    body_text=raw_body,
                    content_type=ctype,
                )
            except ValueError:
                stats["unparseable_uri"] += 1
                feats = featurize_http_request(
                    url="/",
                    method=method,
                    header_values=header_values,
                    body_text=raw_body,
                    content_type=ctype,
                )

        normal = int(normals[i])
        capec_vals = [int(col[i]) for col in capec_cols]
        capec_sum = sum(capec_vals)
        label = 0 if normal == 1 else 1

        if normal == 1 and capec_sum > 0:
            stats["rows_both_normal_and_attack"] += 1
        if label == 1 and capec_sum == 0:
            stats["rows_with_no_capec_and_attack"] += 1
        if capec_sum >= 2:
            stats["multilabel_counts"][capec_sum] = stats["multilabel_counts"].get(capec_sum, 0) + 1
        for name, v in zip(CAPEC_OUT_COLS, capec_vals):
            if v:
                stats["capec_totals"][name] += 1

        if raw_body:
            stats["rows_with_nonempty_request_body"] += 1
        if "?" in raw_uri:
            stats["rows_with_query_string_in_uri"] += 1
        if PCT_TOKEN_RE.search(raw_uri):
            stats["rows_with_pct_encoding_in_uri"] += 1
            if PCT_TOKEN_RE.search(unquote_plus(raw_uri)):
                stats["rows_double_encoded_in_uri"] += 1

        row_id = row_offset + i
        feats["row_id"] = row_id
        feats["label"] = label
        feats[SOURCE_COL] = SOURCE_SRBH
        for name, v in zip(CAPEC_OUT_COLS, capec_vals):
            feats[name] = v
        feat_rows.append(feats)

        text_rows.append(
            {
                "row_id": row_id,
                "text_uri": raw_uri,
                "text_body": raw_body,
                # Matches services/preprocess/main.py::_normalize_headers, which is what
                # services/nlp_infer/main.py receives as the `headers` text field.
                "text_headers": ";".join(f"{k}:{v}" for k, v in header_pairs),
                "label": label,
            }
        )

    feat_df = pd.DataFrame(feat_rows, columns=FEATURE_COLS + ["row_id", "label", SOURCE_COL] + CAPEC_OUT_COLS)
    text_df = pd.DataFrame(text_rows, columns=["row_id", "text_uri", "text_body", "text_headers", "label"])
    return feat_df, text_df


def main(argv=None) -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=SRBH_INPUT_PATH)
    parser.add_argument("--out-dir", default=SRBH_OUT_DIR)
    parser.add_argument("--skip-md5", action="store_true", help="skip the MD5 integrity check")
    args = parser.parse_args(argv)

    assert_feature_parity()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"SR-BH input not found: {args.input}")
    if args.skip_md5:
        log.warning("[srbh] MD5 verification skipped by request")
    else:
        verify_md5(args.input, SRBH_EXPECTED_MD5)

    os.makedirs(args.out_dir, exist_ok=True)
    features_path = os.path.join(args.out_dir, "srbh_features.parquet")
    text_path = os.path.join(args.out_dir, "srbh_text.parquet")
    sample_path = os.path.join(args.out_dir, "srbh_sample_50k.parquet")
    stats_path = os.path.join(args.out_dir, "srbh_stats.json")

    stats = {
        "total_rows": 0,
        "label_0_normal": 0,
        "label_1_attack": 0,
        "rows_with_no_capec_and_attack": 0,
        "rows_both_normal_and_attack": 0,
        "multilabel_counts": {},
        "capec_totals": {c: 0 for c in CAPEC_OUT_COLS},
        "rows_with_nonempty_request_body": 0,
        "rows_with_query_string_in_uri": 0,
        "rows_with_pct_encoding_in_uri": 0,
        "rows_double_encoded_in_uri": 0,
        "malformed_url_fallbacks": 0,
        "unparseable_uri": 0,
    }

    feat_schema = build_features_schema()
    usecols = [METHOD_COL, URI_COL, BODY_COL] + HEADER_COLS + [NORMAL_COL] + CAPEC_COLS
    usecols = list(dict.fromkeys(usecols))

    t0 = time.time()
    row_offset = 0
    next_log = LOG_EVERY

    with pq.ParquetWriter(features_path, feat_schema, compression="zstd") as fw, pq.ParquetWriter(
        text_path, TEXT_SCHEMA, compression="zstd"
    ) as tw:
        reader = pd.read_csv(
            args.input,
            usecols=usecols,
            dtype=str,
            na_filter=False,
            chunksize=CHUNK_ROWS,
            engine="c",
        )
        for chunk in reader:
            feat_df, text_df = process_chunk(chunk, row_offset, stats)
            for c in FEATURE_COLS:
                feat_df[c] = feat_df[c].astype("int64" if c in INT_FEATURES else "float64")
            fw.write_table(pa.Table.from_pandas(feat_df, schema=feat_schema, preserve_index=False))
            tw.write_table(pa.Table.from_pandas(text_df, schema=TEXT_SCHEMA, preserve_index=False))

            n = len(feat_df)
            row_offset += n
            stats["total_rows"] += n
            stats["label_1_attack"] += int(feat_df["label"].sum())
            if row_offset >= next_log:
                log.info("[srbh] processed %s rows (%.1fs)", row_offset, time.time() - t0)
                next_log += LOG_EVERY

    stats["label_0_normal"] = stats["total_rows"] - stats["label_1_attack"]
    log.info("[srbh] ingestion done: %s rows in %.1fs", stats["total_rows"], time.time() - t0)

    feats = pd.read_parquet(features_path, columns=FEATURE_COLS + ["label"])
    emitted_order = [c for c in feats.columns if c in set(FEATURE_COLS)]
    if emitted_order != FEATURE_COLS:
        raise RuntimeError(f"emitted feature order drifted: {emitted_order}")

    nan_counts = feats[FEATURE_COLS].isna().sum()
    if int(nan_counts.sum()) != 0:
        raise RuntimeError(f"NaN present in feature matrix: {nan_counts[nan_counts > 0].to_dict()}")
    import numpy as np

    float_cols = [c for c in FEATURE_COLS if c not in INT_FEATURES]
    inf_count = int(np.isinf(feats[float_cols].to_numpy()).sum())
    if inf_count != 0:
        raise RuntimeError(f"infinite values present in feature matrix: {inf_count}")

    rng_frac = SAMPLE_ROWS / float(stats["total_rows"])
    sample_idx = (
        feats.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=int(round(len(g) * rng_frac)), random_state=SEED), include_groups=False)
        .index
    )
    sample_idx = pd.Index(sample_idx)
    deficit = SAMPLE_ROWS - len(sample_idx)
    if deficit != 0:
        remaining = feats.index.difference(sample_idx)
        if deficit > 0:
            extra = pd.Series(remaining).sample(n=deficit, random_state=SEED).to_numpy()
            sample_idx = sample_idx.append(pd.Index(extra))
        else:
            sample_idx = pd.Index(pd.Series(sample_idx).sample(n=SAMPLE_ROWS, random_state=SEED).to_numpy())

    full = pd.read_parquet(features_path)
    sample = full.loc[sorted(sample_idx)].reset_index(drop=True)
    sample.to_parquet(sample_path, index=False, compression="zstd")

    full_ratio = stats["label_1_attack"] / stats["total_rows"]
    sample_ratio = float(sample["label"].mean())
    if len(sample) != SAMPLE_ROWS:
        raise RuntimeError(f"sample has {len(sample)} rows, expected {SAMPLE_ROWS}")
    if abs(sample_ratio - full_ratio) > 0.005:
        raise RuntimeError(
            f"sample label ratio {sample_ratio:.6f} deviates from full {full_ratio:.6f} by more than 0.5 pp"
        )

    out_stats = {
        "dataset": "SR-BH 2020",
        "source_file": os.path.abspath(args.input),
        "md5": SRBH_EXPECTED_MD5,
        "seed": SEED,
        "total_rows": stats["total_rows"],
        "label_0_normal": stats["label_0_normal"],
        "label_0_normal_pct": round(100.0 * stats["label_0_normal"] / stats["total_rows"], 3),
        "label_1_attack": stats["label_1_attack"],
        "label_1_attack_pct": round(100.0 * stats["label_1_attack"] / stats["total_rows"], 3),
        "rows_with_no_capec_and_attack": stats["rows_with_no_capec_and_attack"],
        "rows_both_normal_and_attack": stats["rows_both_normal_and_attack"],
        "multilabel_exactly_2": stats["multilabel_counts"].get(2, 0),
        "multilabel_exactly_3": stats["multilabel_counts"].get(3, 0),
        "multilabel_exactly_4": stats["multilabel_counts"].get(4, 0),
        "capec_totals": {
            orig: stats["capec_totals"][slug] for orig, slug in zip(CAPEC_COLS, CAPEC_OUT_COLS)
        },
        "rows_with_nonempty_request_body": stats["rows_with_nonempty_request_body"],
        "rows_with_nonempty_request_body_pct": round(
            100.0 * stats["rows_with_nonempty_request_body"] / stats["total_rows"], 1
        ),
        "rows_with_query_string_in_uri": stats["rows_with_query_string_in_uri"],
        "rows_with_query_string_in_uri_pct": round(
            100.0 * stats["rows_with_query_string_in_uri"] / stats["total_rows"], 1
        ),
        "rows_with_pct_encoding_in_uri": stats["rows_with_pct_encoding_in_uri"],
        "rows_double_encoded_in_uri": stats["rows_double_encoded_in_uri"],
        "malformed_url_fallbacks": stats["malformed_url_fallbacks"],
        "unparseable_uri": stats["unparseable_uri"],
        "feature_nan_count": 0,
        "feature_inf_count": 0,
        "feature_order": list(FEATURE_COLS),
        "sample_rows": int(len(sample)),
        "sample_label_1_ratio": round(sample_ratio, 6),
        "full_label_1_ratio": round(full_ratio, 6),
    }
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(out_stats, fh, indent=2, ensure_ascii=False)

    print(json.dumps(out_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
