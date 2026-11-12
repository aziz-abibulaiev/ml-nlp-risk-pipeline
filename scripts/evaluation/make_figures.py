#!/usr/bin/env python3
"""
Figures from the table CSVs of a mixed run.

Writes PNG files to runs/<run>/tables/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# display order and names
CLASSES = [
    ("sql_injection", "SQL Injection"), ("nosql_injection", "NoSQL Injection"),
    ("ldap_injection", "LDAP Injection"), ("cmd_injection", "Command Injection"),
    ("xss", "Cross-Site Scripting"), ("ssrf", "SSRF"),
    ("ssrf_sensitive", "Cloud Metadata SSRF (IMDS)"), ("api_key_exposure", "API Key Exposure"),
    ("auth_bruteforce", "Authentication Brute-Force"), ("credential_stuffing", "Credential Stuffing"),
    ("idor", "IDOR"), ("misconfiguration", "Misconfiguration"), ("data_exposure", "Data Exposure"),
    ("dos", "DoS"), ("websocket", "WebSocket Abuse"), ("business_logic", "Business Logic Abuse"),
]
LEVELS = ["normal", "low", "medium", "high", "critical"]
LEVEL_COLORS = {"normal": "#4F81BD", "low": "#C0504D", "medium": "#9BBB59",
                "high": "#8064A2", "critical": "#4BACC6"}
BLUE, RED, GREY = "#4F81BD", "#C0504D", "#595959"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "font.size": 11, "axes.edgecolor": "#BFBFBF", "axes.labelcolor": GREY, "xtick.color": GREY,
    "ytick.color": GREY, "text.color": GREY, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def fig6(t4: pd.DataFrame, out: Path) -> None:
    metrics = [("Precision", "Precision (%)"), ("Recall", "Recall (%)"), ("F1", "F1 (%)"), ("FPR", "FPR (%)")]
    rows = {r.operating_point: r for r in t4.itertuples(index=False)}
    a, b = rows["Medium+"], rows["Low+"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = range(len(metrics))
    w = 0.34
    va = [100 * getattr(a, m) for m, _ in metrics]
    vb = [100 * getattr(b, m) for m, _ in metrics]
    ba = ax.bar([i - w / 2 for i in x], va, w, color=BLUE, label="A — Alert (Medium+)")
    bb = ax.bar([i + w / 2 for i in x], vb, w, color=RED, label="B — Suspicious (Low+)")
    for bars, vals in ((ba, va), (bb, vb)):
        for r, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (r.get_x() + r.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(list(x), [lbl for _, lbl in metrics])
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, color="#D9D9D9")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Operating-point trade-off", fontsize=15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    fig.savefig(out)
    plt.close(fig)


def fig7(t5g: pd.DataFrame, t5: pd.DataFrame, out: Path) -> None:
    g = t5g.set_index("attack_group")
    rows = [(name, g.loc[key]) for key, name in CLASSES]
    rows.append(("Benign", t5.set_index("class").loc["benign"]))
    rows = rows[::-1]  # first class on top
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    left = [0.0] * len(rows)
    for lv in LEVELS:
        share = [100 * r[lv] / r["total"] for _, r in rows]
        ax.barh([n for n, _ in rows], share, left=left, height=0.55, color=LEVEL_COLORS[lv], label=lv.capitalize())
        left = [l + s for l, s in zip(left, share)]
    for i, (_, r) in enumerate(rows):
        ax.text(101, i, f"n = {int(r['total']):,}", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of events (%)")
    ax.xaxis.grid(True, color="#D9D9D9")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Risk level distribution by ground-truth class", fontsize=15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=5, frameon=False)
    fig.savefig(out)
    plt.close(fig)


def fig8(t5g: pd.DataFrame, t5: pd.DataFrame, out: Path) -> None:
    g = t5g.set_index("attack_group")

    def rate(r):
        return 100 * (r["medium"] + r["high"] + r["critical"]) / r["total"]

    rows = [(name, rate(g.loc[key]), BLUE) for key, name in CLASSES]
    rows.append(("Benign", rate(t5.set_index("class").loc["benign"]), RED))
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.barh([n for n, _, _ in rows], [v for _, v, _ in rows], height=0.5, color=[c for _, _, c in rows])
    for i, (_, v, _) in enumerate(rows):
        ax.text(v + 1, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Alert rate @Medium+ (%)")
    ax.xaxis.grid(True, color="#D9D9D9")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Alert rate @Medium+ by ground-truth class", fontsize=15)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mixed")
    a = ap.parse_args()
    td = ROOT / "runs" / a.run / "tables"
    t4 = pd.read_csv(td / "table4.csv")
    t5 = pd.read_csv(td / "table5.csv")
    t5g = pd.read_csv(td / "table5_by_group.csv")
    assert int(t5g.total.sum()) == int(t5.set_index("class").loc["attack", "total"])
    fig6(t4, td / "fig6_operating_points.png")
    fig7(t5g, t5, td / "fig7_risk_levels_by_class.png")
    fig8(t5g, t5, td / "fig8_alert_rate_by_class.png")
    print("written to", td)


if __name__ == "__main__":
    main()
