"""Build deduplicated, disjoint SR-BH 2020 splits for cross-corpus experiments.

Consumes the TASK-01 artifacts (data/srbh2020/srbh_features.parquet, srbh_text.parquet)
and produces two split families under data/srbh2020/splits/:

Random family (stratified on the binary label, seed 42):

  srbh_train_benign       40 % of benign, benign-only     (novelty models)
  srbh_supervised_train   40 % of benign + 50 % of attacks
  srbh_calib              10 % of benign + 25 % of attacks (calibration + thresholds)
  srbh_test               10 % of benign + 25 % of attacks (touched once)

Temporal family (no shuffling; honeypot traffic is chronological, so a random split
leaks scan campaigns across the boundary):

  srbh_temporal_train_benign / _supervised_train : earliest ~70 % of rows
  srbh_temporal_calib                            : next ~10 %
  srbh_temporal_test                             : last  ~20 %

Hygiene, applied before any splitting:

  * global deduplication on the 24-feature vector (rounded to 6 decimals, the repo
    convention from build_ml_numeric_from_csic.ML_DUP_GROUP_ROUND_DECIMALS), keeping the
    first occurrence in file order — so a repeated request can never land on both sides
    of a boundary;
  * cross-label feature-vector collisions are counted and resolved the same way (first
    occurrence wins);
  * zero feature-vector overlap between every pair of splits is asserted, in both
    families, and the build fails loudly otherwise.

Timestamps are re-read from the source CSV by row_id (file order), since the TASK-01
feature artifact deliberately excludes them.

Outputs are Parquet plus srbh_splits_stats.json. Deterministic: two runs produce
byte-identical stats. Existing files under data/ and models/ are not touched.
"""

import argparse
import hashlib
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_http_features import FEATURE_COLS  # noqa: E402
from build_ml_numeric_from_srbh import SRBH_INPUT_PATH  # noqa: E402

log = logging.getLogger(__name__)

SEED = 42
ROUND_DECIMALS = 6

SRBH_DIR = os.getenv("SRBH_OUT_DIR", "./data/srbh2020")
SPLITS_DIR = os.path.join(SRBH_DIR, "splits")

RANDOM_FAMILY = {
    # split name -> (benign fraction, attack fraction)
    "srbh_train_benign": (0.40, 0.00),
    "srbh_supervised_train": (0.40, 0.50),
    "srbh_calib": (0.10, 0.25),
    "srbh_test": (0.10, 0.25),
}

TEMPORAL_TEST_FRAC = 0.20
TEMPORAL_CALIB_FRAC = 0.10

TS_FORMAT = "%d/%b/%Y:%H:%M:%S %z"


def _pattern_keys(df: pd.DataFrame) -> pd.Series:
    x = df[FEATURE_COLS].round(ROUND_DECIMALS)
    return x.astype(str).agg("|".join, axis=1)


def _rowid_hash(row_ids) -> str:
    b = ",".join(str(int(r)) for r in sorted(row_ids)).encode()
    return hashlib.sha256(b).hexdigest()[:16]


def _split_stats(name: str, df: pd.DataFrame) -> dict:
    n = int(len(df))
    n1 = int(df["label"].sum())
    return {
        "rows": n,
        "label_0": n - n1,
        "label_1": n1,
        "label_1_ratio": round(n1 / n, 6) if n else None,
        "row_id_set_sha256_16": _rowid_hash(df["row_id"].tolist()),
    }


def _assert_pairwise_disjoint(splits: dict[str, pd.DataFrame], family: str) -> None:
    keys = {name: set(_pattern_keys(df)) for name, df in splits.items()}
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = keys[a] & keys[b]
            if inter:
                raise RuntimeError(
                    f"[{family}] feature-vector overlap between {a} and {b}: {len(inter)} patterns"
                )
    log.info("[%s] pairwise feature-vector disjointness verified across %d splits", family, len(names))


def load_timestamps(input_csv: str, n_expected: int) -> pd.Series:
    ts_raw = pd.read_csv(input_csv, usecols=["timestamp"], dtype=str, na_filter=False)["timestamp"]
    if len(ts_raw) != n_expected:
        raise RuntimeError(f"timestamp rows {len(ts_raw)} != features rows {n_expected}")
    ts = pd.to_datetime(ts_raw, format=TS_FORMAT, errors="raise")
    return ts


def main(argv=None) -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=SRBH_INPUT_PATH, help="source CSV, for timestamps")
    args = parser.parse_args(argv)

    feats = pd.read_parquet(os.path.join(SRBH_DIR, "srbh_features.parquet"))
    text = pd.read_parquet(os.path.join(SRBH_DIR, "srbh_text.parquet"))
    if list(feats["row_id"]) != list(text["row_id"]):
        raise RuntimeError("row_id misalignment between srbh_features and srbh_text")
    n_total = len(feats)

    ts = load_timestamps(args.input_csv, n_total)
    monotonic = bool(ts.is_monotonic_increasing)
    log.info("[srbh-splits] loaded %s rows; timestamps monotonic in file order: %s", n_total, monotonic)

    # ---- global deduplication on the 24-feature vector, first occurrence wins ----
    keys = _pattern_keys(feats)
    dup_mask = keys.duplicated(keep="first")

    # Cross-label collisions: same vector, both labels, anywhere in the corpus.
    lbl_per_key = feats.groupby(keys)["label"].nunique()
    collision_keys = set(lbl_per_key[lbl_per_key > 1].index)
    collision_rows = int(keys.isin(collision_keys).sum())

    dup_by_label = {
        int(lbl): int(dup_mask[feats["label"] == lbl].sum()) for lbl in (0, 1)
    }
    rows_by_label = {int(lbl): int((feats["label"] == lbl).sum()) for lbl in (0, 1)}

    dedup = feats.loc[~dup_mask].reset_index(drop=True)
    dedup_ts = ts.loc[~dup_mask.values].reset_index(drop=True)
    log.info(
        "[srbh-splits] dedup: %s -> %s rows (dropped %s = %.2f%%); per-label dropped: benign %s/%s (%.2f%%), attack %s/%s (%.2f%%); cross-label collision patterns=%s rows=%s",
        n_total, len(dedup), int(dup_mask.sum()), 100.0 * dup_mask.sum() / n_total,
        dup_by_label[0], rows_by_label[0], 100.0 * dup_by_label[0] / rows_by_label[0],
        dup_by_label[1], rows_by_label[1], 100.0 * dup_by_label[1] / rows_by_label[1],
        len(collision_keys), collision_rows,
    )

    os.makedirs(SPLITS_DIR, exist_ok=True)
    text_by_rowid = text.set_index("row_id")

    def _write_split(name: str, df: pd.DataFrame) -> None:
        df = df.reset_index(drop=True)
        df.to_parquet(os.path.join(SPLITS_DIR, f"{name}.parquet"), index=False, compression="zstd")
        t = text_by_rowid.loc[df["row_id"]].reset_index()
        if list(t["row_id"]) != list(df["row_id"]):
            raise RuntimeError(f"[{name}] text/features row_id misalignment")
        t.to_parquet(os.path.join(SPLITS_DIR, f"{name}_text.parquet"), index=False, compression="zstd")

    stats: dict = {
        "seed": SEED,
        "round_decimals": ROUND_DECIMALS,
        "total_rows": n_total,
        "dedup_rows": int(len(dedup)),
        "duplicate_rows_removed": int(dup_mask.sum()),
        "duplicate_rate": round(float(dup_mask.sum()) / n_total, 6),
        "duplicate_rate_benign": round(dup_by_label[0] / rows_by_label[0], 6),
        "duplicate_rate_attack": round(dup_by_label[1] / rows_by_label[1], 6),
        "cross_label_collision_patterns": int(len(collision_keys)),
        "cross_label_collision_rows": collision_rows,
        "timestamps_monotonic_in_file_order": monotonic,
        "splits": {},
    }

    # ---- random family: stratified allocation by label, seed 42 ----
    rng = np.random.RandomState(SEED)
    benign_idx = dedup.index[dedup["label"] == 0].to_numpy()
    attack_idx = dedup.index[dedup["label"] == 1].to_numpy()
    rng.shuffle(benign_idx)
    rng.shuffle(attack_idx)

    def _take(pool: np.ndarray, cursor: int, frac: float, pool_total: int) -> tuple[np.ndarray, int]:
        k = int(round(frac * pool_total))
        return pool[cursor : cursor + k], cursor + k

    random_splits: dict[str, pd.DataFrame] = {}
    bc = ac = 0
    names = list(RANDOM_FAMILY)
    for i, name in enumerate(names):
        bfrac, afrac = RANDOM_FAMILY[name]
        if i == len(names) - 1:  # last split takes the remainder, avoiding rounding loss
            b_take, a_take = benign_idx[bc:], attack_idx[ac:]
            bc, ac = len(benign_idx), len(attack_idx)
        else:
            b_take, bc = _take(benign_idx, bc, bfrac, len(benign_idx))
            a_take, ac = _take(attack_idx, ac, afrac, len(attack_idx))
        part = dedup.loc[np.sort(np.concatenate([b_take, a_take]))]
        random_splits[name] = part

    _assert_pairwise_disjoint(random_splits, "random")
    for name, df in random_splits.items():
        _write_split(name, df)
        stats["splits"][name] = _split_stats(name, df)

    # ---- temporal family: chronological order, no shuffling ----
    order = np.argsort(dedup_ts.values, kind="stable")
    chron = dedup.iloc[order].reset_index(drop=True)
    chron_ts = dedup_ts.iloc[order].reset_index(drop=True)

    n = len(chron)
    n_test = int(round(TEMPORAL_TEST_FRAC * n))
    n_calib = int(round(TEMPORAL_CALIB_FRAC * n))
    n_train = n - n_test - n_calib

    train_win = chron.iloc[:n_train]
    calib_win = chron.iloc[n_train : n_train + n_calib]
    test_win = chron.iloc[n_train + n_calib :]

    temporal_splits = {
        "srbh_temporal_train_benign": train_win[train_win["label"] == 0],
        "srbh_temporal_supervised_train": train_win,
        "srbh_temporal_calib": calib_win,
        "srbh_temporal_test": test_win,
    }
    # train_benign is a subset of supervised_train by design; disjointness is asserted
    # between the three windows (train/calib/test), which is what leakage is about.
    _assert_pairwise_disjoint(
        {
            "srbh_temporal_supervised_train": train_win,
            "srbh_temporal_calib": calib_win,
            "srbh_temporal_test": test_win,
        },
        "temporal",
    )
    for name, df in temporal_splits.items():
        _write_split(name, df)
        stats["splits"][name] = _split_stats(name, df)

    stats["temporal"] = {
        "train_rows": int(n_train),
        "calib_rows": int(n_calib),
        "test_rows": int(n_test),
        "train_end": str(chron_ts.iloc[n_train - 1]),
        "calib_start": str(chron_ts.iloc[n_train]),
        "calib_end": str(chron_ts.iloc[n_train + n_calib - 1]),
        "test_start": str(chron_ts.iloc[n_train + n_calib]),
        "test_end": str(chron_ts.iloc[n - 1]),
        "note": "train_benign is the benign subset of supervised_train (same time window)",
    }

    out_path = os.path.join(SRBH_DIR, "srbh_splits_stats.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
