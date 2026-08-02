"""Smoke test for the SR-BH 2020 ingestion path.

Runs the extractor over ~1,000 SR-BH-shaped rows and asserts feature-order parity with
the trained models plus absence of NaN/inf. Uses the real dataset head when the file is
present, and falls back to a deterministic synthetic fixture covering the awkward cases
(double encoding, absent headers, unknown verbs, oversized fields) otherwise, so the
assertions run either way.

    pytest tests/srbh_ingestion_smoke.py
    python tests/srbh_ingestion_smoke.py
"""

import json
import os
import random
import sys
import urllib.parse

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "jobs", "retrain"))

from build_ml_numeric_from_srbh import (  # noqa: E402
    CAPEC_COLS,
    CAPEC_OUT_COLS,
    HEADER_COLS,
    HEADER_WIRE_NAMES,
    INT_FEATURES,
    NORMAL_COL,
    SRBH_INPUT_PATH,
    assert_feature_parity,
    process_chunk,
)
from ml_http_features import FEATURE_COLS  # noqa: E402

N_ROWS = 1000
SEED = 42

INPUT_COLS = [
    "request_http_method",
    "request_http_request",
    "request_body",
] + HEADER_COLS + [NORMAL_COL] + CAPEC_COLS


def _synthetic_chunk() -> pd.DataFrame:
    rng = random.Random(SEED)
    targets = [
        "/",
        "/blog/index.php/2020/04/04/voluptatum-reprehenderit/",
        "/blog/xmlrpc.php?rsd",
        "/blog/etc%252Fpasswd/my-account/edit-profile/%7B%7B%20attachment.thumbnail.url%20%7D%7D",
        "/%3C%21--%23EXEC+cmd%3D%22ls+%2F%22--%3E/index.php",
        "/index.php?id=1' OR 1=1--&q=" + "A" * 4000,
        "http://test-site.com/absolute-form",
        "/wp-login.php?redirect_to=%2F%2Fevil%2Ecom",
        "not-a-leading-slash",
        "/a/b/c/d/e/f?x=&y=&z=",
    ]
    methods = ["GET", "POST", "HEAD", "OPTIONS", "TRACE", "FROBNICATE", ""]
    rows = []
    for i in range(N_ROWS):
        rec = {c: "" for c in INPUT_COLS}
        rec["request_http_method"] = rng.choice(methods)
        rec["request_http_request"] = rng.choice(targets)
        rec["request_body"] = "" if i % 8 else json.dumps({"user": "a" * (i % 50), "pw": "x"})
        rec["request_host"] = "test-site.com"
        if i % 3:
            rec["request_user_agent"] = "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/%d.0" % (i % 90)
        if i % 5 == 0:
            rec["request_cookie"] = "wordpress_logged_in=" + "c" * (i % 200)
        if i % 4 == 0:
            rec["request_content_type"] = rng.choice(["application/json", "application/x-www-form-urlencoded"])
        rec["request_accept"] = "*/*"
        rec["request_connection"] = "keep-alive"
        attack = i % 3 == 0
        rec[NORMAL_COL] = "0" if attack else "1"
        if attack:
            rec[CAPEC_COLS[i % len(CAPEC_COLS)]] = "1"
        for c in CAPEC_COLS:
            rec[c] = rec[c] or "0"
        rows.append(rec)
    return pd.DataFrame(rows, columns=INPUT_COLS)


def load_chunk() -> pd.DataFrame:
    path = os.getenv("SRBH_INPUT_CSV", SRBH_INPUT_PATH)
    if os.path.exists(path):
        return pd.read_csv(path, usecols=INPUT_COLS, dtype=str, na_filter=False, nrows=N_ROWS)[INPUT_COLS]
    return _synthetic_chunk()


def _empty_stats() -> dict:
    return {
        "rows_both_normal_and_attack": 0,
        "rows_with_no_capec_and_attack": 0,
        "multilabel_counts": {},
        "capec_totals": {c: 0 for c in CAPEC_OUT_COLS},
        "rows_with_nonempty_request_body": 0,
        "rows_with_query_string_in_uri": 0,
        "rows_with_pct_encoding_in_uri": 0,
        "rows_double_encoded_in_uri": 0,
        "malformed_url_fallbacks": 0,
        "unparseable_uri": 0,
    }


def test_feature_order_parity():
    assert_feature_parity()


def test_extractor_over_1000_rows():
    chunk = load_chunk()
    assert len(chunk) == N_ROWS

    feats, text = process_chunk(chunk, 0, _empty_stats())

    assert len(feats) == N_ROWS
    assert len(text) == N_ROWS

    emitted = [c for c in feats.columns if c in set(FEATURE_COLS)]
    assert emitted == FEATURE_COLS, f"feature order drifted: {emitted}"
    assert list(feats.columns)[: len(FEATURE_COLS)] == FEATURE_COLS

    block = feats[FEATURE_COLS]
    assert int(block.isna().sum().sum()) == 0, "NaN in feature matrix"
    assert not np.isinf(block.to_numpy(dtype="float64")).any(), "inf in feature matrix"

    for c in FEATURE_COLS:
        v = pd.to_numeric(feats[c])
        if c in INT_FEATURES:
            assert (v == v.astype("int64")).all(), f"{c} is not integral"
            assert (v >= 0).all(), f"{c} is negative"

    assert set(feats["label"].unique()) <= {0, 1}
    assert (feats["source"] == "srbh2020").all()
    assert list(text["row_id"]) == list(feats["row_id"])

    # header_count must reflect headers actually present, never a constant 11.
    # (Not asserting variance here: the real dataset's first 1000 rows are homogeneous.)
    assert feats["header_count"].max() <= len(HEADER_COLS)
    assert not (feats["header_count"] == len(HEADER_COLS)).all(), "header_count collapsed to constant 11"


def test_empty_headers_are_omitted():
    """The mapping decision: absent headers must not inflate header_count."""
    base = {c: "" for c in INPUT_COLS}
    base["request_http_method"] = "GET"
    base["request_http_request"] = "/x"
    base[NORMAL_COL] = "1"
    for c in CAPEC_COLS:
        base[c] = "0"

    sparse = dict(base, request_host="h", request_accept="*/*")
    full = dict(base, **{c: "v" for c in HEADER_COLS})
    chunk = pd.DataFrame([sparse, full], columns=INPUT_COLS)

    feats, text = process_chunk(chunk, 0, _empty_stats())
    assert feats["header_count"].tolist() == [2, len(HEADER_COLS)]
    assert feats["header_value_total_len"].tolist() == [len("h") + len("*/*"), len(HEADER_COLS)]
    assert text["text_headers"].iloc[0] == "host:h;accept:*/*"


def test_text_headers_format_matches_preprocess():
    """Must reproduce services/preprocess/main.py::_normalize_headers byte for byte.

    Note the format is lossy: header values legitimately contain ';' (User-Agent) and ':'
    (Referer), so the blob cannot be split back into pairs. That is the pipeline's
    existing wire format and is matched deliberately, not fixed here.
    """
    chunk = load_chunk()
    _, text = process_chunk(chunk, 0, _empty_stats())

    expected = []
    for _, row in chunk.iterrows():
        pairs = [(HEADER_WIRE_NAMES[c], row[c]) for c in HEADER_COLS if row[c]]
        expected.append(";".join(f"{k}:{v}" for k, v in pairs))
    assert list(text["text_headers"]) == expected

    blobs = [b for b in text["text_headers"] if b]
    assert blobs, "no header blobs produced"
    for blob in blobs[:50]:
        assert blob.split(";", 1)[0].split(":", 1)[0] in HEADER_WIRE_NAMES.values()


def test_raw_text_is_not_pre_decoded():
    """The NLP channel decodes at inference; the artifact must keep raw text."""
    chunk = load_chunk()
    _, text = process_chunk(chunk, 0, _empty_stats())
    assert list(text["text_uri"]) == list(chunk["request_http_request"])
    assert list(text["text_body"]) == list(chunk["request_body"])
    encoded = [u for u in text["text_uri"] if "%" in u]
    if encoded:
        assert any(u != urllib.parse.unquote_plus(u) for u in encoded)


if __name__ == "__main__":
    test_feature_order_parity()
    test_extractor_over_1000_rows()
    test_empty_headers_are_omitted()
    test_text_headers_format_matches_preprocess()
    test_raw_text_is_not_pre_decoded()
    print("srbh_ingestion_smoke: all assertions passed")
