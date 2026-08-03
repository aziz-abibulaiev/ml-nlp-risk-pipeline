import argparse
import logging
import os

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from ml_http_features import FEATURE_COLS, featurize_http_request, normalize_url

log = logging.getLogger(__name__)

SOURCE_COL = "source"
SOURCE_CSIC = "csic"

ML_DUP_GROUP_ROUND_DECIMALS = int(os.getenv("ML_DUP_GROUP_ROUND_DECIMALS", "6"))
ML_DATASET_SHUFFLE_SEED = int(os.getenv("ML_DATASET_SHUFFLE_SEED", "42"))
ML_VAL_MAX_DUP_PER_PATTERN_BENIGN = int(os.getenv("ML_VAL_MAX_DUP_PER_PATTERN_BENIGN", "0"))
ML_VAL_MAX_DUP_PER_PATTERN_ATTACK = int(os.getenv("ML_VAL_MAX_DUP_PER_PATTERN_ATTACK", "0"))

# Path to save the output ML numeric features CSV (benign → train)
# NOTE: CSIC DOES NOT TOUCH TRAIN_OUT_PATH ANYMORE
TRAIN_OUT_PATH = os.getenv("ML_NUM_FILE", "./data/ml_numeric.csv")

# Path to save validation/test rows (benign + attacks → validation)
VAL_OUT_PATH = os.getenv("ML_VALIDATION_FILE", "./data/ml_validation.csv")
TEST_OUT_PATH = os.getenv("ML_TEST_FILE", "./data/ml_test.csv")
SUPERVISED_TRAIN_OUT_PATH = os.getenv("ML_SUPERVISED_TRAIN_FILE", "./data/ml_supervised_train.csv")

# Optional extra benign-only split for OCSVM training.
# This split is filtered to be pattern-disjoint from validation/test.
OCSVM_EXTRA_BENIGN_FILE = os.getenv("OCSVM_EXTRA_BENIGN_FILE", "").strip()
CSIC_OCSVM_BENIGN_ROWS = int(os.getenv("CSIC_OCSVM_BENIGN_ROWS", "0"))
CSIC_OCSVM_BENIGN_OFFSET = int(os.getenv("CSIC_OCSVM_BENIGN_OFFSET", "0"))

# CSIC class-wise split configuration for fixed calibration/test files.
# Values are "requested"; if dataset has fewer rows, script takes what exists.
CSIC_VAL_BENIGN_ROWS = int(os.getenv("CSIC_VAL_BENIGN_ROWS", "10000"))
CSIC_VAL_BENIGN_OFFSET = int(os.getenv("CSIC_VAL_BENIGN_OFFSET", "0"))
CSIC_VAL_ATTACK_ROWS = int(os.getenv("CSIC_VAL_ATTACK_ROWS", "0"))
CSIC_VAL_ATTACK_OFFSET = int(os.getenv("CSIC_VAL_ATTACK_OFFSET", "0"))

CSIC_TEST_BENIGN_ROWS = int(os.getenv("CSIC_TEST_BENIGN_ROWS", "20000"))
CSIC_TEST_BENIGN_OFFSET = int(os.getenv("CSIC_TEST_BENIGN_OFFSET", "10000"))
CSIC_TEST_ATTACK_ROWS = int(os.getenv("CSIC_TEST_ATTACK_ROWS", "20000"))
CSIC_TEST_ATTACK_OFFSET = int(os.getenv("CSIC_TEST_ATTACK_OFFSET", "0"))

ML_ENFORCE_VAL_TEST_DISJOINT = os.getenv("ML_ENFORCE_VAL_TEST_DISJOINT", "true").strip().lower() in (
    "1", "true", "yes", "on"
)
CSIC_SUPERVISED_EXCLUDE_VAL_TEST = os.getenv("CSIC_SUPERVISED_EXCLUDE_VAL_TEST", "true").strip().lower() in (
    "1", "true", "yes", "on"
)
CSIC_SUPERVISED_EXCLUDE_MODE = os.getenv("CSIC_SUPERVISED_EXCLUDE_MODE", "pattern").strip().lower()
CSIC_SUPERVISED_REQUIRE_BOTH_CLASSES = os.getenv("CSIC_SUPERVISED_REQUIRE_BOTH_CLASSES", "true").strip().lower() in (
    "1", "true", "yes", "on"
)

# By default, pull datasets from Kaggle:
USE_KAGGLE = os.getenv("USE_KAGGLE_CSIC", "1") in ("1", "true", "True", "yes", "on")
KAGGLE_FILE_PATH = os.getenv(
    "CSIC_KAGGLE_FILE",
    "csic_database.csv",
)

# Local CSV is used only if USE_KAGGLE_CSIC=0 is explicitly set
LOCAL_CSIC_PATH = os.getenv("CSIC_LOCAL_PATH", "./data/csic_database.csv")

HEADER_COLS = [
    "User-Agent", "Pragma", "Cache-Control", "Accept",
    "Accept-encoding", "Accept-charset", "language",
    "host", "cookie", "content-type", "connection"
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


def load_csic_df() -> pd.DataFrame:
    """
    Load the CSIC dataset from Kaggle or local disk.
    """
    if USE_KAGGLE:
        if kagglehub is None:
            raise RuntimeError("USE_KAGGLE_CSIC=1, but kagglehub is not installed")
        if not KAGGLE_FILE_PATH:
            raise RuntimeError("CSIC_KAGGLE_FILE must be set (relative path in dataset)")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "ispangler/csic-2010-web-application-attacks",
            KAGGLE_FILE_PATH,
        )
        return df

    if not os.path.exists(LOCAL_CSIC_PATH):
        raise FileNotFoundError(f"CSIC_LOCAL_PATH does not exist: {LOCAL_CSIC_PATH}")
    return pd.read_csv(LOCAL_CSIC_PATH)


def featurize_row(row) -> dict:
    """
    Featurize a single row from the CSIC dataset.
    """
    raw_url = str(row.get("URL", "") or "").strip()
    if raw_url:
        raw_url = raw_url.split()[0]
    url = normalize_url(raw_url, host=row.get("host"), default_host="csic.local")
    headers_values = [row.get(col) for col in HEADER_COLS]
    feats = featurize_http_request(
        url=url,
        method=row.get("Method"),
        header_values=headers_values,
        body_text=row.get("content", ""),
        content_type=row.get("content-type"),
    )

    cls = row.get("classification", None)
    try:
        label_raw = int(cls)
    except Exception:
        label_raw = None

    # Normalize to 0/1:
    #   classification == 0 -> label=0 (benign)
    #   classification != 0 -> label=1 (attack)
    if label_raw is None:
        label = None
    else:
        label = 0 if label_raw == 0 else 1

    return {
        **feats,
        "label": label,
        SOURCE_COL: SOURCE_CSIC,
    }


def append_or_write(df: pd.DataFrame, path: str, override: bool, tag: str) -> None:
    """
    Write part of the dataset to CSV:
      - if override=True or file does not exist → write new file
      - else → append to existing file (with column check).
    """
    if df.empty:
        log.info("[csic→%s] no rows to write, skipping (%s)", tag, path)
        return

    # Make sure directory exists
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)

    if override or not os.path.exists(path):
        df.to_csv(path, index=False)
        log.info(
            "[csic→%s] saved %s rows -> %s (override=%s)",
            tag,
            df.shape[0],
            path,
            override,
        )
        return

    # Append mode with column compatibility check
    base_head = pd.read_csv(path, nrows=0)
    missing = [c for c in df.columns if c not in base_head.columns]
    if missing:
        raise ValueError(
            f"[csic→{tag}] existing {path} has no columns {missing}; "
            f"have={base_head.columns.tolist()}"
        )

    # Align column order
    df = df[base_head.columns]
    df.to_csv(path, mode="a", index=False, header=False)
    log.info(
        "[csic→%s] appended %s rows -> %s",
        tag,
        df.shape[0],
        path,
    )


def _take_slice(df: pd.DataFrame, offset: int, rows: int, tag: str) -> pd.DataFrame:
    """
    Take a safe contiguous slice [offset : offset+rows) from dataframe.
    If bounds exceed available rows, return truncated slice and log it.
    """
    if rows <= 0 or df.empty:
        return df.iloc[0:0].copy()

    n = int(df.shape[0])
    start = max(0, int(offset))
    if start >= n:
        log.warning("[%s] offset=%s >= available=%s, returning empty slice", tag, start, n)
        return df.iloc[0:0].copy()
    end = min(n, start + int(rows))
    out = df.iloc[start:end].copy()
    if out.shape[0] < int(rows):
        log.warning(
            "[%s] requested rows=%s from offset=%s, got=%s (available=%s)",
            tag,
            int(rows),
            start,
            int(out.shape[0]),
            n,
        )
    return out


def _pattern_key_series(df: pd.DataFrame) -> pd.Series:
    """
    Build comparable feature-pattern keys for overlap/disjoint checks.
    """
    if df.empty:
        return pd.Series(dtype=str)
    x = df[FEATURE_COLS].copy()
    for c in FEATURE_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce").round(ML_DUP_GROUP_ROUND_DECIMALS)
    return x.astype(str).agg("|".join, axis=1)


def _drop_overlapping_patterns_from_test(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tag: str = "csic",
) -> tuple[pd.DataFrame, dict]:
    """
    Enforce pattern disjointness by removing from test all rows whose feature-pattern
    already appears in validation.
    """
    if val_df.empty or test_df.empty:
        return test_df, {"overlap_patterns": 0, "dropped_test_rows": 0}

    val_keys = _pattern_key_series(val_df)
    test_keys = _pattern_key_series(test_df)
    overlap = set(val_keys.tolist()) & set(test_keys.tolist())
    if not overlap:
        return test_df, {"overlap_patterns": 0, "dropped_test_rows": 0}

    keep_mask = ~test_keys.isin(overlap)
    dropped = int((~keep_mask).sum())
    out = test_df.loc[keep_mask].reset_index(drop=True)
    log.warning(
        "[%s] enforced val/test disjoint: overlap_patterns=%s dropped_test_rows=%s",
        tag,
        len(overlap),
        dropped,
    )
    return out, {"overlap_patterns": int(len(overlap)), "dropped_test_rows": dropped}


def _drop_val_test_overlap_from_supervised_train(
    sup_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str = "pattern",
) -> tuple[pd.DataFrame, dict]:
    """
    Remove leakage from supervised train against validation/test.
    Mode:
      - pattern: drop rows whose FEATURE pattern appears in val/test (recommended)
      - row: drop exact full-row matches (FEATURE_COLS + label + source)
    """
    if sup_df.empty:
        return sup_df, {"mode": mode, "dropped_rows": 0}
    forbidden_df = pd.concat([val_df, test_df], ignore_index=True)
    if forbidden_df.empty:
        return sup_df, {"mode": mode, "dropped_rows": 0}

    mode_norm = str(mode or "pattern").strip().lower()
    if mode_norm == "row":
        row_cols = [c for c in (FEATURE_COLS + ["label", SOURCE_COL]) if c in sup_df.columns and c in forbidden_df.columns]
        sup_keys = sup_df[row_cols].astype(str).agg("|".join, axis=1)
        forb_keys = set(forbidden_df[row_cols].astype(str).agg("|".join, axis=1).tolist())
        keep_mask = ~sup_keys.isin(forb_keys)
        out = sup_df.loc[keep_mask].reset_index(drop=True)
        return out, {
            "mode": "row",
            "forbidden_rows": int(len(forb_keys)),
            "dropped_rows": int((~keep_mask).sum()),
        }

    sup_keys = _pattern_key_series(sup_df)
    forbidden_keys = set(_pattern_key_series(forbidden_df).tolist())
    keep_mask = ~sup_keys.isin(forbidden_keys)
    out = sup_df.loc[keep_mask].reset_index(drop=True)
    return out, {
        "mode": "pattern",
        "forbidden_patterns": int(len(forbidden_keys)),
        "dropped_rows": int((~keep_mask).sum()),
        "dropped_patterns": int((sup_keys[~keep_mask].nunique() if (~keep_mask).any() else 0)),
    }


def _refill_test_by_label_disjoint(
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    label_value: int,
    target_rows: int,
    tag: str,
) -> pd.DataFrame:
    """
    Refill test split for a single label using pool rows that do not overlap val/test patterns.
    """
    if target_rows <= 0:
        return test_df

    cur = test_df.loc[test_df["label"] == int(label_value)].copy()
    deficit = int(target_rows) - int(cur.shape[0])
    if deficit <= 0:
        return test_df

    val_keys = set(_pattern_key_series(val_df).tolist())
    test_keys = set(_pattern_key_series(test_df).tolist())
    forbidden = val_keys | test_keys

    pool_lbl = pool_df.loc[pool_df["label"] == int(label_value)].copy()
    if pool_lbl.empty:
        return test_df
    pool_keys = _pattern_key_series(pool_lbl)
    cand = pool_lbl.loc[~pool_keys.isin(forbidden)].copy()
    if cand.empty:
        log.warning(
            "[%s] refill label=%s failed: no non-overlap candidates (deficit=%s)",
            tag,
            int(label_value),
            deficit,
        )
        return test_df

    add = cand.iloc[:deficit].copy()
    out = pd.concat([test_df, add], ignore_index=True)
    log.info(
        "[%s] refill label=%s added=%s (deficit=%s)",
        tag,
        int(label_value),
        int(add.shape[0]),
        deficit,
    )
    return out


def main(argv=None) -> None:
    """
    Build ML numeric features from CSIC and write fixed class-wise slices into:
      - ml_validation.csv (VAL_OUT_PATH)
      - ml_test.csv       (TEST_OUT_PATH)
      - ml_supervised_train.csv (SUPERVISED_TRAIN_OUT_PATH, full mixed CSIC)

    TRAIN_OUT_PATH (ml_numeric.csv) is NOT modified here.

    If --override is passed, both output files are overwritten.
    Otherwise, data is appended.
    """
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--override",
        action="store_true",
        help="Overwrite existing ml_validation.csv, ml_test.csv and ml_supervised_train.csv instead of appending",
    )
    parser.add_argument(
        "--override-validation",
        action="store_true",
        help="Overwrite existing ml_validation.csv only",
    )
    parser.add_argument(
        "--override-test",
        action="store_true",
        help="Overwrite existing ml_test.csv only",
    )
    parser.add_argument(
        "--override-supervised-train",
        action="store_true",
        help="Overwrite existing ml_supervised_train.csv only",
    )
    args = parser.parse_args(argv)

    df = load_csic_df()
    rows = []
    for _, r in df.iterrows():
        feats = featurize_row(r)
        rows.append(feats)

    out_df = pd.DataFrame(rows)

    cols = FEATURE_COLS + ["label", SOURCE_COL]
    out_df = out_df[cols]

    # Drop rows where label is None/NaN
    out_df = out_df.dropna(subset=["label"])

    # Make sure label is 0/1 (it already is, but be explicit)
    out_df["label"] = out_df["label"].astype(int)
    out_df["label"] = (out_df["label"] != 0).astype(int)

    benign_df = out_df[out_df["label"] == 0]
    attack_df = out_df[out_df["label"] == 1]

    log.info(
        "[csic] total rows=%s, benign=%s, attacks=%s",
        out_df.shape[0],
        benign_df.shape[0],
        attack_df.shape[0],
    )

    val_benign_df = _take_slice(
        benign_df,
        offset=CSIC_VAL_BENIGN_OFFSET,
        rows=CSIC_VAL_BENIGN_ROWS,
        tag="csic/val_benign",
    )
    val_attack_df = _take_slice(
        attack_df,
        offset=CSIC_VAL_ATTACK_OFFSET,
        rows=CSIC_VAL_ATTACK_ROWS,
        tag="csic/val_attack",
    )
    val_df = pd.concat([val_benign_df, val_attack_df], ignore_index=True)

    test_benign_df = _take_slice(
        benign_df,
        offset=CSIC_TEST_BENIGN_OFFSET,
        rows=CSIC_TEST_BENIGN_ROWS,
        tag="csic/test_benign",
    )
    test_attack_df = _take_slice(
        attack_df,
        offset=CSIC_TEST_ATTACK_OFFSET,
        rows=CSIC_TEST_ATTACK_ROWS,
        tag="csic/test_attack",
    )
    test_df = pd.concat([test_benign_df, test_attack_df], ignore_index=True)

    disjoint_info = {}
    if ML_ENFORCE_VAL_TEST_DISJOINT:
        test_df, disjoint_info = _drop_overlapping_patterns_from_test(
            val_df=val_df,
            test_df=test_df,
            tag="csic",
        )
        # Try to recover requested test class counts while keeping disjointness.
        test_df = _refill_test_by_label_disjoint(
            test_df=test_df,
            val_df=val_df,
            pool_df=out_df,
            label_value=0,
            target_rows=CSIC_TEST_BENIGN_ROWS,
            tag="csic/test_refill",
        )
        test_df = _refill_test_by_label_disjoint(
            test_df=test_df,
            val_df=val_df,
            pool_df=out_df,
            label_value=1,
            target_rows=CSIC_TEST_ATTACK_ROWS,
            tag="csic/test_refill",
        )

    log.info(
        "[csic/split] val rows=%s labels=%s | test rows=%s labels=%s | disjoint=%s",
        int(val_df.shape[0]),
        val_df["label"].value_counts().to_dict() if not val_df.empty else {},
        int(test_df.shape[0]),
        test_df["label"].value_counts().to_dict() if not test_df.empty else {},
        disjoint_info,
    )

    override_val = bool(args.override or args.override_validation)
    override_test = bool(args.override or args.override_test)
    append_or_write(val_df, VAL_OUT_PATH, override_val, "ml_validation")
    append_or_write(test_df, TEST_OUT_PATH, override_test, "ml_test")

    # Supervised numeric train (mixed): full CSIC records, optionally excluding
    # all patterns/rows that appear in validation/test to avoid leakage.
    sup_train_df = out_df.copy()
    sup_exclude_info = {"enabled": bool(CSIC_SUPERVISED_EXCLUDE_VAL_TEST), "mode": str(CSIC_SUPERVISED_EXCLUDE_MODE)}
    if CSIC_SUPERVISED_EXCLUDE_VAL_TEST:
        sup_train_df, drop_info = _drop_val_test_overlap_from_supervised_train(
            sup_df=sup_train_df,
            val_df=val_df,
            test_df=test_df,
            mode=CSIC_SUPERVISED_EXCLUDE_MODE,
        )
        sup_exclude_info.update(drop_info or {})
    sup_label_counts = sup_train_df["label"].value_counts().to_dict() if (not sup_train_df.empty and "label" in sup_train_df.columns) else {}
    if CSIC_SUPERVISED_REQUIRE_BOTH_CLASSES:
        has_both = (0 in sup_label_counts) and (1 in sup_label_counts)
        if not has_both:
            raise RuntimeError(
                "[csic/supervised] after excluding val/test overlap, supervised train no longer has both classes; "
                f"labels={sup_label_counts}. Adjust CSIC_* split sizes or disable exclusion."
            )
    if not sup_train_df.empty:
        sup_train_df = sup_train_df.sample(frac=1.0, random_state=ML_DATASET_SHUFFLE_SEED).reset_index(drop=True)
    override_sup = bool(args.override or args.override_supervised_train)
    append_or_write(sup_train_df, SUPERVISED_TRAIN_OUT_PATH, override_sup, "ml_supervised_train")
    log.info(
        "[csic/supervised] rows=%s labels=%s path=%s exclude=%s",
        int(sup_train_df.shape[0]),
        sup_label_counts,
        SUPERVISED_TRAIN_OUT_PATH,
        sup_exclude_info,
    )

    # Optional CSIC benign-only extra split for OCSVM train (pattern-disjoint from val/test).
    if OCSVM_EXTRA_BENIGN_FILE and CSIC_OCSVM_BENIGN_ROWS > 0:
        target_extra_rows = int(CSIC_OCSVM_BENIGN_ROWS)
        extra_df = _take_slice(
            benign_df,
            offset=CSIC_OCSVM_BENIGN_OFFSET,
            rows=target_extra_rows,
            tag="csic/ocsvm_extra_benign",
        )
        if not extra_df.empty:
            used_keys = set(_pattern_key_series(pd.concat([val_df, test_df], ignore_index=True)).tolist())
            extra_keys = _pattern_key_series(extra_df)
            keep_extra = ~extra_keys.isin(used_keys)
            dropped_extra = int((~keep_extra).sum())
            extra_df = extra_df.loc[keep_extra].reset_index(drop=True)
            if dropped_extra > 0:
                log.info(
                    "[csic/ocsvm_extra] dropped rows overlapping with val/test patterns: %s",
                    dropped_extra,
                )

            # If requested quota is not met, backfill from the full benign pool
            # with any remaining non-overlap patterns.
            if extra_df.shape[0] < target_extra_rows:
                need = target_extra_rows - int(extra_df.shape[0])
                extra_forbidden = used_keys | set(_pattern_key_series(extra_df).tolist())
                all_benign_keys = _pattern_key_series(benign_df)
                cand = benign_df.loc[~all_benign_keys.isin(extra_forbidden)].copy()
                if not cand.empty:
                    add = cand.iloc[:need].copy()
                    extra_df = pd.concat([extra_df, add], ignore_index=True)
                    log.info(
                        "[csic/ocsvm_extra] backfilled rows from benign pool: +%s (target=%s)",
                        int(add.shape[0]),
                        target_extra_rows,
                    )

        if extra_df.empty and os.path.exists(OCSVM_EXTRA_BENIGN_FILE):
            os.remove(OCSVM_EXTRA_BENIGN_FILE)
            log.info(
                "[csic/ocsvm_extra] no rows after disjoint filtering; removed stale file: %s",
                OCSVM_EXTRA_BENIGN_FILE,
            )
        else:
            append_or_write(
                extra_df,
                OCSVM_EXTRA_BENIGN_FILE,
                override=True,
                tag="ocsvm_extra_benign",
            )

    # Optional de-dup cap for calibration/test data.
    if ML_VAL_MAX_DUP_PER_PATTERN_BENIGN > 0 or ML_VAL_MAX_DUP_PER_PATTERN_ATTACK > 0:
        _postprocess_dataset_file(
            VAL_OUT_PATH,
            cap_by_label={
                0: ML_VAL_MAX_DUP_PER_PATTERN_BENIGN,
                1: ML_VAL_MAX_DUP_PER_PATTERN_ATTACK,
            },
            tag="validation/ml_validation_final",
        )
        _postprocess_dataset_file(
            TEST_OUT_PATH,
            cap_by_label={
                0: ML_VAL_MAX_DUP_PER_PATTERN_BENIGN,
                1: ML_VAL_MAX_DUP_PER_PATTERN_ATTACK,
            },
            tag="test/ml_test_final",
        )
    else:
        log.info("[csic] de-dup caps disabled for validation/test (both caps <= 0)")


if __name__ == "__main__":
    main()
