#!/usr/bin/env python3
"""
Data splits from the Kaggle datasets.

Writes the raw request fields of every split and manifest.json to data/splits/.
Usage: python scripts/evaluation/prepare_splits.py [--kaggle-root ~/.cache/kagglehub/datasets]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED_SPLIT = 42
SEED_MIXED = 20261110

ML_TRAIN_RANGE = (0, 500_000)
ML_VAL_RANGE = (500_000, 1_000_000)
MIXED_START = 5_000_000
N_MIXED_BENIGN = 25_000
N_ATTACKLAB_A = 250_000
N_ATTACKLAB_B = 10_000
N_CSIC_ATTACK_VAL = 10_000
N_CSIC_NORMAL_TEST = 15_000

def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_labeled_logs(path: Path) -> dict[str, pd.DataFrame]:
    """Stream the labeled logs once and keep the row ranges of the splits."""
    keep = {"train": [], "val": [], "mixed": []}
    excluded = {"train": {}, "val": {}, "mixed": {}}
    offset = 0
    cols = ["method", "url", "status", "referrer", "user_agent", "type"]
    for ch in pd.read_csv(path, usecols=cols, dtype=str, chunksize=1_000_000, keep_default_na=False):
        ch.insert(0, "row_id", np.arange(offset, offset + len(ch)))
        offset += len(ch)
        for name, lo, hi in (("train", *ML_TRAIN_RANGE), ("val", *ML_VAL_RANGE), ("mixed", MIXED_START, 1 << 62)):
            part = ch[(ch.row_id >= lo) & (ch.row_id < hi)]
            if part.empty:
                continue
            for t, n in part.type.value_counts().items():
                if t != "benign":
                    excluded[name][t] = excluded[name].get(t, 0) + int(n)
            keep[name].append(part[part.type == "benign"])
    out = {}
    for name, parts in keep.items():
        df = pd.concat(parts, ignore_index=True)
        out[name] = pd.DataFrame({
            "source": "wsal_labeled",
            "row_id": df.row_id,
            "method": df.method,
            "url": df.url,
            "user_agent": df.user_agent,
            "referrer": df.referrer,
            "status": df.status,
            "body": "",
            "label": 0,
        })
    out["_excluded"] = excluded
    out["_total_rows"] = offset
    return out


def load_attacklab(path: Path) -> pd.DataFrame:
    cols = ["frame.number", "http.request.method", "http.request.uri", "http.user_agent",
            "http.referer", "http.file_data", "alert"]
    header = pd.read_csv(path, nrows=0).columns
    cols = [c for c in cols if c in header]
    parts = []
    for ch in pd.read_csv(path, usecols=cols, dtype=str, chunksize=1_000_000, low_memory=False):
        ch = ch[ch["http.request.method"].notna() & (ch["alert"] == "suspicious")]
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    return pd.DataFrame({
        "source": "attacklab",
        "row_id": df["frame.number"].astype("int64"),
        "method": df["http.request.method"],
        "url": df["http.request.uri"].fillna(""),
        "user_agent": df.get("http.user_agent", pd.Series("", index=df.index)).fillna(""),
        "referrer": df.get("http.referer", pd.Series("", index=df.index)).fillna(""),
        "status": "",
        "body": df.get("http.file_data", pd.Series("", index=df.index)).fillna(""),
        "label": 1,
    })


def load_csic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    url = df["URL"].str.replace(r"\s+HTTP/\d\.\d$", "", regex=True)
    return pd.DataFrame({
        "source": "csic2010",
        "row_id": np.arange(len(df)),
        "method": df["Method"],
        "url": url,
        "user_agent": df["User-Agent"],
        "referrer": "",
        "status": "",
        "body": df["content"],
        "label": df["classification"].astype(int),
    })


def read_payload_objects(path: Path) -> list[dict]:
    """
    Read the payload file line by line. It is a pretty-printed JSON array with formatting
    defects (non-breaking spaces, missing commas, invalid escapes).
    """
    import re
    field = re.compile(r'^\s*"(\w+)"\s*:\s*"(.*)"\s*,?\s*$')
    objs, cur = [], None
    for line in path.read_text(encoding="utf-8").replace(" ", " ").splitlines():
        s = line.strip()
        if s.startswith("{"):
            cur = {}
        elif s.startswith("}"):
            if cur:
                objs.append(cur)
            cur = None
        elif cur is not None and (m := field.match(line)):
            try:
                cur[m.group(1)] = json.loads('"' + m.group(2) + '"')
            except json.JSONDecodeError:
                cur[m.group(1)] = m.group(2).replace('\\"', '"').replace("\\\\", "\\")
    return objs


def build_nlp(httpparams: Path, payloads: Path) -> dict[str, pd.DataFrame]:
    hp = pd.read_csv(httpparams, dtype=str, keep_default_na=False)
    hp = pd.DataFrame({"text": hp["payload"], "label": (hp["label"] != "norm").astype(int),
                       "attack_type": hp["attack_type"], "source": "httpparams"})
    pl = pd.DataFrame(read_payload_objects(payloads)).dropna(subset=["payload"])
    pl = pd.DataFrame({"text": pl["payload"].astype(str), "label": 1,
                       "attack_type": pl["id"].str.split("-").str[0], "source": "web_payloads"})
    df = pd.concat([hp, pl], ignore_index=True)
    n_before = len(df)
    df = df.assign(text=df["text"].astype(str))
    df = df[df.text.str.strip() != ""].copy()
    # payload-string de-duplication before splitting; conflicting labels resolved to attack
    df = df.sort_values("label", ascending=False, kind="stable").drop_duplicates("text", keep="first").copy()
    rng = np.random.default_rng(SEED_SPLIT)
    split = np.empty(len(df), dtype=object)
    for lab in (0, 1):
        idx = np.flatnonzero(df.label.to_numpy() == lab)
        rng.shuffle(idx)
        n_tr, n_va = int(round(0.70 * len(idx))), int(round(0.15 * len(idx)))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "val"
        split[idx[n_tr + n_va:]] = "test"
    df["split"] = split
    out = {s: df[df.split == s].drop(columns="split").reset_index(drop=True) for s in ("train", "val", "test")}
    out["_dedup"] = {"before": n_before, "after": len(df)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-root", default=str(Path.home() / ".cache/kagglehub/datasets"))
    ap.add_argument("--out", default="data/splits")
    a = ap.parse_args()
    root, out = Path(a.kaggle_root).expanduser(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    src = {
        "wsal_labeled": root / "mahendradata/web-server-access-logs-labeled/versions/1/labeled.csv",
        "attacklab": root / "moradrawashdeh/attack-simulation-lab/versions/1/attack-simulation-alert.csv",
        "csic2010": root / "ispangler/csic-2010-web-application-attacks/versions/1/csic_database.csv",
        "httpparams": root / "evg3n1j/httpparamsdataset/versions/1/payload_full.csv",
        "web_payloads": root / "cyberprince/web-application-payloads-dataset/versions/1/WEB_APPLICATION_PAYLOADS.jsonl",
    }
    manifest: dict = {"seeds": {"split": SEED_SPLIT, "mixed": SEED_MIXED}, "sources": {}, "counts": {}}
    for k, p in src.items():
        manifest["sources"][k] = {"path": str(p), "sha256": sha256(p)}

    logs = load_labeled_logs(src["wsal_labeled"])
    manifest["sources"]["wsal_labeled"]["total_rows"] = logs["_total_rows"]
    manifest["excluded_non_benign"] = logs["_excluded"]

    mixed = logs["mixed"].drop_duplicates(["method", "url", "user_agent"])
    manifest["counts"]["mixed_benign_unique_pool"] = len(mixed)
    mixed = mixed.sample(n=N_MIXED_BENIGN, random_state=SEED_MIXED).sort_values("row_id")

    al = load_attacklab(src["attacklab"])
    manifest["counts"]["attacklab_http_suspicious"] = len(al)
    al = al.sample(frac=1.0, random_state=SEED_SPLIT)
    al_a, al_b = al.iloc[:N_ATTACKLAB_A], al.iloc[N_ATTACKLAB_A:N_ATTACKLAB_A + N_ATTACKLAB_B]

    csic = load_csic(src["csic2010"]).sample(frac=1.0, random_state=SEED_SPLIT)
    csic_att = csic[csic.label == 1].iloc[:N_CSIC_ATTACK_VAL]
    csic_norm = csic[csic.label == 0].iloc[:N_CSIC_NORMAL_TEST]

    files = {
        "ml_train_benign": logs["train"],
        "ml_val_benign": logs["val"],
        "ml_val_attacks": pd.concat([al_a, csic_att], ignore_index=True),
        "ml_test": pd.concat([al_b, csic_norm], ignore_index=True),
        "mixed_benign": mixed,
    }
    nlp = build_nlp(src["httpparams"], src["web_payloads"])
    manifest["nlp_dedup"] = nlp.pop("_dedup")
    for s, df in nlp.items():
        files[f"nlp_{s}"] = df

    # leakage checks
    ids = {k: set(zip(v.source, v.row_id)) for k, v in files.items() if "row_id" in v}
    names = list(ids)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            assert not (ids[x] & ids[y]), f"row overlap between {x} and {y}"
    nlp_texts = set(pd.concat([nlp["train"], nlp["val"]]).text)
    assert not (nlp_texts & set(nlp["test"].text)), "NLP train/val vs test text overlap"
    assert not (nlp_texts & set(mixed.url)), "mixed benign URL found in NLP train/val"

    for k, df in files.items():
        df.to_csv(out / f"{k}.csv", index=False)
        manifest["counts"][k] = int(len(df))
        if "label" in df:
            manifest["counts"][k + "_by_label"] = {str(a): int(b) for a, b in df.label.value_counts().items()}
    # URL overlap is reported, not enforced: same website, different requests in time
    manifest["mixed_url_seen_in_ml_train"] = int(mixed.url.isin(set(files["ml_train_benign"].url)).sum())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
