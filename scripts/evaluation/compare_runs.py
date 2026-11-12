#!/usr/bin/env python3
"""
Compare the tables and per-event outputs of a run with a reference run.

Exit code 0 means identical tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ("table4.csv", "table4_distinct.csv", "table5.csv", "table5_by_group.csv")


SCORES = ("agg_ml", "agg_nlp", "final_score")


def fusion_outputs(run_dir: Path) -> dict[int, dict]:
    out = {}
    with open(run_dir / "events.jsonl") as f:
        for line in f:
            e = json.loads(line)
            out[e["seq"]] = (e.get("response") or {}).get("fusion") or {}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    run, ref = ROOT / "runs" / a.run, ROOT / "runs" / a.ref
    same = True
    print(f"== {a.run} vs {a.ref}")
    for name in TABLES:
        p, q = run / "tables" / name, ref / "tables" / name
        if not p.exists() or not q.exists():
            print(f"{name}: missing ({'run' if not p.exists() else 'reference'})")
            same = False
            continue
        x, y = pd.read_csv(p), pd.read_csv(q)
        if x.equals(y):
            print(f"{name}: identical")
            continue
        same = False
        print(f"{name}: DIFFERENT")
        if x.shape == y.shape and list(x.columns) == list(y.columns):
            diff = x.compare(y, result_names=("run", "reference"))
            print(diff.to_string())
        else:
            print(f"  shape {x.shape} vs {y.shape}")
    if (run / "events.jsonl").exists() and (ref / "events.jsonl").exists():
        fx, fy = fusion_outputs(run), fusion_outputs(ref)
        common = sorted(set(fx) & set(fy))
        print(f"events: {len(common)} matched")
        for k in SCORES:
            n = sum(1 for s in common if fx[s].get(k) != fy[s].get(k))
            print(f"  {k} differs for {n}")
        changed = [s for s in common if fx[s].get("risk_level") != fy[s].get("risk_level")]
        print(f"  risk level differs for {len(changed)}")
        for s in changed[:20]:
            print(f"    seq {s}: {fy[s].get('risk_level')} -> {fx[s].get('risk_level')}")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
