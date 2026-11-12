#!/usr/bin/env python3
"""
Send an evaluation stream through the running pipeline.

--set validation  validation sample, used only to select thresholds
--set mixed       benign traffic and smoke attacks in a fixed random order

Writes runs/<run-id>/events.jsonl and run_info.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.featurize import request_to_event  # noqa: E402

SEED_SPLIT = 42
SEED_MIXED = 20261110
N_VAL_EACH = 10_000


def rows_to_events(df: pd.DataFrame, kind: str) -> list[dict]:
    out = []
    for r in df.itertuples(index=False):
        ev = request_to_event(r.method, r.url, r.user_agent, r.referrer, r.status, r.body)
        out.append({"label": int(r.label), "source": r.source, "row_id": str(r.row_id),
                    "attack_group": "", "kind": kind, "event": ev})
    return out


def smoke_attack_events(loops: int = 5, seed: str = "20261110") -> list[dict]:
    spec = importlib.util.spec_from_file_location("smoke", ROOT / "tests/owasp_top_10_smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    captured: list[dict] = []

    def capture(session, url, detailed, token, event, attack_name, idx, pretty):
        captured.append(dict(event))
        return "", None

    smoke.send_request = capture
    smoke.say = lambda *_a, **_k: None
    argv = sys.argv
    sys.argv = ["smoke", "--loops", str(loops), "--seed", seed, "--no-pretty"]
    try:
        smoke.main()
    except SystemExit:
        pass
    finally:
        sys.argv = argv
    out = []
    for i, ev in enumerate(captured):
        grp = ev.pop("attack_group", "")
        ev.pop("trace_nonce", None)
        out.append({"label": 1, "source": "smoke", "row_id": str(i), "attack_group": grp,
                    "kind": "attack", "event": ev})
    return out


def build(set_name: str, data: Path) -> list[dict]:
    if set_name == "validation":
        vb = pd.read_csv(data / "ml_val_benign.csv", dtype=str, keep_default_na=False)
        va = pd.read_csv(data / "ml_val_attacks.csv", dtype=str, keep_default_na=False)
        vb = vb.sample(n=N_VAL_EACH, random_state=SEED_SPLIT)
        n_al = round(N_VAL_EACH * (va.source == "attacklab").sum() / len(va))
        va = pd.concat([
            va[va.source == "attacklab"].sample(n=n_al, random_state=SEED_SPLIT),
            va[va.source == "csic2010"].sample(n=N_VAL_EACH - n_al, random_state=SEED_SPLIT),
        ])
        items = rows_to_events(vb, "benign") + rows_to_events(va, "attack")
        random.Random(SEED_SPLIT).shuffle(items)
        return items
    if set_name == "mixed":
        mb = pd.read_csv(data / "mixed_benign.csv", dtype=str, keep_default_na=False)
        items = rows_to_events(mb, "benign") + smoke_attack_events()
        random.Random(SEED_MIXED).shuffle(items)
        return items
    raise SystemExit(set_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["validation", "mixed"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--url", default=f"http://localhost:{os.getenv('ORCH_PORT', '8000')}/process?detailed=1")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--data", default=str(ROOT / "data/splits"))
    ap.add_argument("--dry-run", action="store_true", help="build the stream and print counts only")
    a = ap.parse_args()

    items = build(a.set, Path(a.data))
    counts = pd.Series([f"{i['kind']}:{i['source']}" for i in items]).value_counts().to_dict()
    print(json.dumps({"n": len(items), "counts": counts}), flush=True)
    if a.dry_run:
        return

    run_dir = ROOT / "runs" / a.run_id
    if (run_dir / "events.jsonl").exists():
        raise SystemExit(f"{run_dir}/events.jsonl exists; runs are never overwritten")
    run_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ["ORCH_TOKEN"]
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD"], capture_output=True, text=True).stdout
    (run_dir / "run_info.json").write_text(json.dumps({
        "set": a.set, "run_id": a.run_id, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git, "worktree_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "n_events": len(items), "counts": counts, "concurrency": a.concurrency,
        "env": {k: os.getenv(k) for k in ("ML_THRESHOLD", "NLP_THRESHOLD", "FUSION_LEVELS",
                                          "FUSION_NORMAL_THR", "HYST_ENABLED", "MODELS_DIR",
                                          "ML_AGGREGATE", "FUSION_MODE", "FUSION_ALPHA", "FUSION_BETA",
                                          "IMPACT_POLICY_HOST", "COMPOSE_PROJECT_NAME")},
    }, indent=2))

    sess = requests.Session()
    headers = {"X-Orch-Token": token, "Content-Type": "application/json"}

    def send(i: int, it: dict) -> dict:
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                r = sess.post(a.url, json={"event": it["event"]}, headers=headers, timeout=60)
                lat = (time.perf_counter() - t0) * 1000
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
                return {"seq": i, **{k: it[k] for k in ("label", "source", "row_id", "attack_group", "kind")},
                        "http_status": r.status_code, "latency_ms": round(lat, 2), "response": body}
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                time.sleep(1 + attempt)
        return {"seq": i, **{k: it[k] for k in ("label", "source", "row_id", "attack_group", "kind")},
                "http_status": -1, "error": err}

    t0 = time.time()
    with open(run_dir / "events.jsonl", "w") as f:
        if a.concurrency <= 1:
            for i, it in enumerate(items):
                f.write(json.dumps(send(i, it), ensure_ascii=False) + "\n")
                if i % 1000 == 0:
                    print(f"{i}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        else:
            with cf.ThreadPoolExecutor(a.concurrency) as ex:
                futs = [ex.submit(send, i, it) for i, it in enumerate(items)]
                for n, fu in enumerate(cf.as_completed(futs)):
                    f.write(json.dumps(fu.result(), ensure_ascii=False) + "\n")
                    if n % 1000 == 0:
                        print(f"{n}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    info = json.loads((run_dir / "run_info.json").read_text())
    info.update({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_seconds": round(wall, 1)})
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"done {len(items)} events in {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
