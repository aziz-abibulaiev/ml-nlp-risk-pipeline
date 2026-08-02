"""Campaign manifest writer/reader — the ONLY source of labels.

Every generation run (benign or attack) writes one sidecar manifest describing which tool
ran against which target from which src_ip during which time window. The labeller assigns
label / attack_family / attack_tool / label_source purely by matching a captured flow's
(src_ip, timestamp) into these windows. No captured payload is ever inspected.

A benign campaign is just a campaign with label=benign and attack_family/attack_tool null.
"""

import json
import os
from datetime import datetime, timezone


def now_iso() -> str:
    t = datetime.now(timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(t.microsecond / 1000):03d}Z"


def write_manifest(
    out_dir: str,
    campaign_id: str,
    app: str,
    label: str,
    src_ip: str,
    start_ts: str,
    end_ts: str,
    attack_family: str | None = None,
    attack_tool: str | None = None,
    extra: dict | None = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "campaign_id": campaign_id,
        "app": app,
        "label": label,
        "attack_family": attack_family,
        "attack_tool": attack_tool,
        "src_ip": src_ip,
        "start_ts": start_ts,
        "end_ts": end_ts,
    }
    if extra:
        manifest["extra"] = extra
    path = os.path.join(out_dir, f"{campaign_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def load_manifests(manifest_dir: str) -> list[dict]:
    out = []
    for name in sorted(os.listdir(manifest_dir)):
        if name.endswith(".json"):
            with open(os.path.join(manifest_dir, name), encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out
