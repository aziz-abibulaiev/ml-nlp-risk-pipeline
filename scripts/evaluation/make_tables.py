#!/usr/bin/env python3
"""
Detection tables from the per-event outputs of one or more runs.

Writes CSV, Markdown, XLSX and summary.json to runs/<run>/tables/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
LEVELS = ["normal", "low", "medium", "high", "critical"]
OPS = {"Medium+": {"medium", "high", "critical"}, "Low+": {"low", "medium", "high", "critical"}}


def cp_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def load(run: str) -> pd.DataFrame:
    rows = []
    for line in open(ROOT / "runs" / run / "events.jsonl"):
        e = json.loads(line)
        r = e.get("response") or {}
        fus = r.get("fusion") or {}
        rows.append({
            "seq": e["seq"], "label": e["label"], "source": e["source"], "row_id": e["row_id"],
            "attack_group": e.get("attack_group") or "", "http_status": e.get("http_status"),
            "latency_ms": e.get("latency_ms"), "hash_id": r.get("hash_id"),
            "content_key": hashlib.sha256(json.dumps({k: ((r.get("preprocess") or {}).get("text_fields") or {}).get(k)
                                                      for k in ("method", "endpoint", "payload", "headers")},
                                                     sort_keys=True).encode()).hexdigest()[:16],
            "ml_score": (r.get("ml") or {}).get("ml_score"), "nlp_score": (r.get("nlp") or {}).get("nlp_score"),
            "final_score": fus.get("final_score"), "risk_level": fus.get("risk_level"),
            "top_topic": ((fus.get("context") or {}).get("topics") or [{}])[0].get("topic") if (fus.get("context") or {}).get("topics") else "",
        })
    return pd.DataFrame(rows).sort_values("seq")


def table4(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for op, acc in OPS.items():
        pred = df.risk_level.isin(acc)
        tp = int((pred & (df.label == 1)).sum()); fp = int((pred & (df.label == 0)).sum())
        fn = int((~pred & (df.label == 1)).sum()); tn = int((~pred & (df.label == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
        fpr = fp / (fp + tn) if fp + tn else float("nan")
        rl, rh = cp_interval(tp, tp + fn); fl, fh = cp_interval(fp, fp + tn)
        out.append({"operating_point": op, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "Precision": prec, "Recall": rec, "F1": f1, "FPR": fpr,
                    "Recall_CI95": f"[{rl:.4f}, {rh:.4f}]", "FPR_CI95": f"[{fl:.4f}, {fh:.4f}]"})
    return pd.DataFrame(out)


def table4_distinct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Like table4, but every distinct request counts once. An attack counts as detected only if
    all its occurrences are, a benign request as flagged if any occurrence is.
    """
    out = []
    for op, acc in OPS.items():
        d = df.assign(pred=df.risk_level.isin(acc))
        att = d[d.label == 1].groupby("content_key").pred.all()
        ben = d[d.label == 0].groupby("content_key").pred.any()
        tp, fn = int(att.sum()), int((~att).sum()); fp, tn = int(ben.sum()), int((~ben).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        fpr = fp / (fp + tn) if fp + tn else float("nan")
        rl, rh = cp_interval(tp, tp + fn); fl, fh = cp_interval(fp, fp + tn)
        out.append({"operating_point": op, "distinct_attacks": tp + fn, "distinct_benign": fp + tn,
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn, "Recall": rec, "FPR": fpr,
                    "Recall_CI95": f"[{rl:.4f}, {rh:.4f}]", "FPR_CI95": f"[{fl:.4f}, {fh:.4f}]"})
    return pd.DataFrame(out)


def table5(df: pd.DataFrame, by: str) -> pd.DataFrame:
    t = pd.crosstab(df[by], df.risk_level).reindex(columns=LEVELS, fill_value=0)
    t["total"] = t.sum(axis=1)
    return t.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    a = ap.parse_args()
    for run in a.run:
        df = load(run)
        failed = int((df.http_status != 200).sum())
        ok = df.loc[df.http_status == 200].assign(**{"class": lambda x: x.label.map({0: "benign", 1: "attack"})})
        t4, t5 = table4(ok), table5(ok, "class")
        t4d = table4_distinct(ok)
        t5g = table5(ok[ok.label == 1], "attack_group")
        lat = ok.latency_ms.describe(percentiles=[0.5, 0.95, 0.99]).to_dict()
        info = json.loads((ROOT / "runs" / run / "run_info.json").read_text())
        summary = {"run": run, "n_events": int(len(df)), "failed_requests": failed,
                   "n_benign": int((ok.label == 0).sum()), "n_attack": int((ok.label == 1).sum()),
                   "latency_ms": {k: float(v) for k, v in lat.items()}, "run_info": info,
                   "table4": t4.to_dict(orient="records"),
                   "table4_distinct": t4d.to_dict(orient="records")}
        od = ROOT / "runs" / run / "tables"
        od.mkdir(exist_ok=True)
        t4.to_csv(od / "table4.csv", index=False); t4d.to_csv(od / "table4_distinct.csv", index=False); t5.to_csv(od / "table5.csv", index=False)
        t5g.to_csv(od / "table5_by_group.csv", index=False)
        ok.to_csv(od / "events_flat.csv", index=False)
        (od / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        with pd.ExcelWriter(od / "tables.xlsx") as xw:
            t4.to_excel(xw, sheet_name="table4", index=False)
            t4d.to_excel(xw, sheet_name="table4_distinct", index=False); t5.to_excel(xw, sheet_name="table5", index=False)
            t5g.to_excel(xw, sheet_name="table5_by_group", index=False); ok.to_excel(xw, sheet_name="events", index=False)
        md = [f"# Run {run}", "", f"Events: {len(df)} (failed requests: {failed})", "",
              "## Operating points", "", t4.to_markdown(index=False, floatfmt=".4f"), "",
              "## Operating points, distinct requests (each distinct method+endpoint+payload+headers once)", "", t4d.to_markdown(index=False, floatfmt=".4f"), "",
              "## Risk levels by class", "", t5.to_markdown(index=False), "",
              "## Risk levels by attack group", "", t5g.to_markdown(index=False), ""]
        (od / "tables.md").write_text("\n".join(md))
        print("\n".join(md))


if __name__ == "__main__":
    main()
