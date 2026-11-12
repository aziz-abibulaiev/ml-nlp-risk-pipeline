#!/usr/bin/env python3
"""
Latency and throughput of the log-replay endpoint.

Replays the mixed stream through POST /replay_access_log, once detailed with concurrency 1
and once simple with concurrency 20. Writes runs/<run-id>/results.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stream import build  # noqa: E402

MODES = [("replay-detailed", True, 1), ("replay-simple-20", False, 20)]


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--url", default=f"http://localhost:{os.getenv('ORCH_PORT', '8000')}")
    ap.add_argument("--data", default=str(ROOT / "data/splits"))
    a = ap.parse_args()

    run_dir = ROOT / "runs" / a.run_id
    if (run_dir / "results.json").exists():
        raise SystemExit(f"{run_dir}/results.json exists; runs are never overwritten")
    run_dir.mkdir(parents=True, exist_ok=True)

    items = build("mixed", Path(a.data))
    src = run_dir / "replay_events.jsonl"
    with open(src, "w") as f:
        for it in items:
            f.write(json.dumps(it["event"], ensure_ascii=False) + "\n")
    orch = sh("docker", "compose", "ps", "-q", "orchestrator")
    sh("docker", "cp", str(src), f"{orch}:/tmp/replay_events.jsonl")

    git = sh("git", "rev-parse", "HEAD")
    diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout
    info = {"run_id": a.run_id, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_head": git, "worktree_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "n_events": len(items), "input_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "env": {k: os.getenv(k) for k in ("ML_THRESHOLD", "NLP_THRESHOLD", "FUSION_LEVELS",
                                              "FUSION_NORMAL_THR", "HYST_ENABLED", "MODELS_DIR",
                                              "ML_AGGREGATE", "FUSION_MODE", "FUSION_ALPHA", "FUSION_BETA",
                                              "IMPACT_POLICY_HOST", "COMPOSE_PROJECT_NAME")},
            "docker": {k: v for k, v in json.loads(sh("docker", "info", "--format", "{{json .}}")).items()
                       if k in ("NCPU", "MemTotal", "ServerVersion", "OperatingSystem", "Architecture")}}
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({"n": len(items)}), flush=True)

    headers = {"X-Orch-Token": os.environ["ORCH_TOKEN"], "Content-Type": "application/json"}
    results = {}
    for name, detailed, conc in MODES:
        sh("docker", "compose", "exec", "-T", "redis", "redis-cli", "FLUSHALL")
        time.sleep(5)
        t0 = time.perf_counter()
        r = requests.post(f"{a.url}/replay_access_log", params={"concurrency": conc},
                          json={"path": "/tmp/replay_events.jsonl", "kind": "event_jsonl", "detailed": detailed},
                          headers=headers, timeout=6 * 3600)
        client_s = time.perf_counter() - t0
        r.raise_for_status()
        body = r.json()
        body.pop("previews", None)
        res = {**body, "detailed": detailed, "client_wall_s": round(client_s, 3),
               "throughput_eps": round(body["events_sent"] / body["elapsed_seconds"], 1)}
        results[name] = res
        print(name, json.dumps(res), flush=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
