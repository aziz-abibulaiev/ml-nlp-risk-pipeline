"""TASK 08 Part B — adversarial robustness of the deployed text-only content detector.

Static WAF-bypass repertoire (technique lineage: BWAFSQLi ACM TOSEM 2025, AdvSQLi
arXiv:2401.02615, WAFBooster arXiv:2501.14008) applied deterministically (seed 42) to
held-out real+testbed attack payloads. Measures recall on original vs mutated, per family x
per mutation type, at the DEPLOYED operating point (1% FPR on zanbil benign). Also: ML vs the
regex baseline under evasion, and with vs without the fixed-point decode (to show the decode
neutralises encoding-based evasions). NO adversarial training — this is evaluation only.

Run: python -m services.content_detector.adversarial
"""

from __future__ import annotations

import json
import os
import random
import re
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .normalize import decode_fixed_point
from .predict import _EXPLAIN_RULES

MDIR = "models/content_v6"
ZMODEL = "models/content_v5/text_only_zanbil.joblib"
TESTBED = "data/corpus/phase2/labelled/testbed_phase2.parquet"
PER_FAMILY = 120
FAMILIES = ["sqli", "xss", "cmd_injection", "ssti", "path_traversal"]
_WS = ["\t", "\n", "\x0b", "\x0c", "%09", "+", "/**/"]


# ---------------- mutation repertoire ----------------
def m_identity(s, rng):
    return s


def m_case(s, rng):
    return "".join(ch.upper() if (ch.isalpha() and rng.random() < 0.5) else ch.lower() if ch.isalpha() else ch for ch in s)


def m_inline_comment(s, rng):
    s = s.replace(" ", "/**/")
    # also split alpha runs of length>=4 once with /**/
    return re.sub(r"([A-Za-z]{2})([A-Za-z]{2,})", lambda m: m.group(1) + "/**/" + m.group(2), s, count=3)


def m_versioned_comment(s, rng):
    return re.sub(r"\b(union|select|or|and)\b", lambda m: f"/*!50000{m.group(1)}*/", s, flags=re.I)


def m_whitespace(s, rng):
    return re.sub(r" ", lambda _: rng.choice(_WS), s)


def m_url_encode(s, rng):
    return quote(s, safe="")


def m_double_url(s, rng):
    return quote(quote(s, safe=""), safe="")


def m_html_entity(s, rng):
    return "".join(f"&#{ord(c)};" if c in "<>\"'()=;/ " else c for c in s)


def m_equivalent(s, rng):
    r = s
    r = re.sub(r"1\s*=\s*1", rng.choice(["2>1", "'a'='a'", "3-1=2", "1 like 1"]), r, flags=re.I)
    r = re.sub(r"'1'='1'", "'a'='a'", r, flags=re.I)
    r = re.sub(r"=\s*", " LIKE ", r) if rng.random() < 0.3 else r
    return r


def m_keyword_char(s, rng):
    # replace quoted string literals with CHAR() concatenation (SQL)
    def repl(m):
        inner = m.group(1)
        return "CHAR(" + ",".join(str(ord(c)) for c in inner) + ")"
    return re.sub(r"'([^']{1,20})'", repl, s)


def m_xss_alt(s, rng):
    r = s
    r = re.sub(r"<script", lambda _: "<" + m_case("script", rng), r, flags=re.I)
    r = r.replace("javascript:", "java\tscript:")
    r = re.sub(r"onerror", "OnErRoR", r, flags=re.I)
    return r


def m_null(s, rng):
    return s + rng.choice(["%00", "\x00", "-- -", "#", "%23"])


MUTATIONS = {
    "case": (m_case, ["sqli", "xss", "cmd_injection", "ssti", "path_traversal"]),
    "inline_comment": (m_inline_comment, ["sqli", "cmd_injection"]),
    "versioned_comment": (m_versioned_comment, ["sqli"]),
    "whitespace_sub": (m_whitespace, ["sqli", "cmd_injection", "ssti"]),
    "url_encode": (m_url_encode, FAMILIES),
    "double_url_encode": (m_double_url, FAMILIES),
    "html_entity": (m_html_entity, ["xss", "sqli"]),
    "equivalent_syntax": (m_equivalent, ["sqli"]),
    "keyword_char": (m_keyword_char, ["sqli"]),
    "xss_alt": (m_xss_alt, ["xss"]),
    "null_terminator": (m_null, FAMILIES),
}


# ---------------- scoring ----------------
def load_model():
    m = joblib.load(ZMODEL)
    from .multidomain import split_benign
    zan = pd.read_parquet("data/corpus/phase2/external/zanbil.parquet",
                          columns=["timestamp", "http_method", "url_path_raw", "url_query", "headers", "label"])
    zan = zan[zan.label == "benign"]; zan["request_body"] = ""
    _, calib, _, _ = split_benign(zan, "temporal")
    thr = float(np.quantile(m.p(calib), 1 - C.PRIMARY_FPR, method="higher"))
    return m, thr


def _build_probe(tb):
    """Text-only attack-pattern detector: testbed benign vs testbed non-custom attacks.
    Held-out custom-tool attacks are the evasion test set; threshold at 1% testbed benign."""
    from .train_partA_v3 import TextOnly
    ben = tb[tb.label == "benign"]
    atk = tb[(tb.label == "attack") & (tb.attack_tool != "custom")]
    fit = pd.concat([ben.assign(y=0), atk.assign(y=1)], ignore_index=True)
    fit = fit.sample(frac=1.0, random_state=C.SEED)
    cal = ben.sample(frac=0.2, random_state=C.SEED)
    p = TextOnly().fit(pd.DataFrame(_reqcols(fit)), fit.y.values)
    thr = float(np.quantile(p.p(pd.DataFrame(_reqcols(cal))), 1 - C.PRIMARY_FPR, method="higher"))
    return p, thr


def _reqcols(df):
    d = df.copy()
    for c in ["url_path_raw", "url_query", "request_body", "headers"]:
        if c not in d:
            d[c] = ""
    return d[["url_path_raw", "url_query", "request_body", "headers"]].to_dict("records")


def score_with_decode(m, payloads):
    df = pd.DataFrame({"url_path_raw": payloads, "url_query": "", "request_body": "", "headers": ""})
    return m.p(df)


def score_without_decode(m, payloads):
    from scipy.sparse import hstack
    Xa = m.va.transform(list(payloads)); Xb = m.vb.transform([""] * len(payloads))
    return m.clf.predict_proba(hstack([Xa, Xb]).tocsr())[:, 1]


def regex_flags(payloads):
    out = []
    for p in payloads:
        dec, _ = decode_fixed_point(p)
        out.append(any(pat.search(dec) for pat in _EXPLAIN_RULES.values()))
    return np.array(out, dtype=bool)


def main():
    os.makedirs(MDIR, exist_ok=True)
    rng = random.Random(C.SEED)
    m, thr = load_model()

    tb = pd.read_parquet(TESTBED, columns=["url_path_raw", "url_query", "request_body", "headers", "label", "attack_family", "attack_tool"])
    atk = tb[(tb.label == "attack") & (tb.attack_tool == "custom")]

    base = {}  # family -> list of decoded payload strings (held-out custom tool)
    for fam in FAMILIES:
        s = atk[atk.attack_family == fam]
        payloads = []
        seen = set()
        for _, r in s.iterrows():
            dec, _ = decode_fixed_point(f"{r.url_path_raw} {r.url_query} {r.request_body}".strip())
            if dec and dec not in seen:
                seen.add(dec); payloads.append(dec)
            if len(payloads) >= PER_FAMILY:
                break
        base[fam] = payloads

    # ---- same-domain attack-pattern probe (isolates TRUE evasion from OOD) ----
    # A text-only model that learned to separate attacks from SAME-domain (testbed) benign, so
    # a mutation that breaks the attack pattern can actually evade it — unlike the deployed
    # zanbil model, which flags any non-zanbil content (recall confounded by OOD).
    probe, probe_thr = _build_probe(tb)

    results = {"operating_point": {"threshold": thr, "reference": "1% FPR zanbil benign"},
               "probe_operating_point": {"threshold": probe_thr, "reference": "1% FPR testbed benign",
                                         "what": "attack-pattern detector (attacks vs same-domain benign) — the honest evasion probe"},
               "per_family": {}, "probe": {},
               "note": "static WAF-bypass repertoire; no adaptive/white-box attacker; no adversarial training"}

    # per family x mutation
    agg = {mt: {"ml_decode": [], "ml_nodecode": [], "regex": [], "probe": [], "probe_nodecode": [], "n": 0}
           for mt in ["original"] + list(MUTATIONS)}
    for fam, payloads in base.items():
        if not payloads:
            continue
        fam_res = {}
        orig = payloads
        r_ml = float((score_with_decode(m, orig) >= thr).mean())
        r_pr = float((score_with_decode(probe, orig) >= probe_thr).mean())
        r_rx = float(regex_flags(orig).mean())
        fam_res["original"] = {"n": len(orig), "deployed_recall": r_ml, "probe_recall": r_pr, "regex_recall": r_rx}
        agg["original"]["ml_decode"].append(r_ml); agg["original"]["probe"].append(r_pr)
        agg["original"]["regex"].append(r_rx); agg["original"]["n"] += len(orig)
        for mt, (fn, fams) in MUTATIONS.items():
            if fam not in fams:
                continue
            mut = [fn(p, rng) for p in payloads]
            rd = float((score_with_decode(m, mut) >= thr).mean())
            pr = float((score_with_decode(probe, mut) >= probe_thr).mean())
            pr_nd = float((score_without_decode(probe, mut) >= probe_thr).mean())
            rr = float(regex_flags(mut).mean())
            fam_res[mt] = {"deployed_recall": rd, "probe_recall": pr, "probe_recall_nodecode": pr_nd, "regex_recall": rr}
            agg[mt]["ml_decode"].append(rd); agg[mt]["probe"].append(pr); agg[mt]["probe_nodecode"].append(pr_nd)
            agg[mt]["regex"].append(rr); agg[mt]["n"] += len(mut)
        results["per_family"][fam] = fam_res

    # aggregate over families
    def avg(x):
        return float(np.mean(x)) if x else None
    results["aggregate"] = {mt: {"deployed_recall": avg(v["ml_decode"]),
                                 "probe_recall": avg(v["probe"]),
                                 "probe_recall_nodecode": avg(v["probe_nodecode"]),
                                 "regex_recall": avg(v["regex"]), "n": v["n"]} for mt, v in agg.items()}

    with open(os.path.join(MDIR, "adversarial_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    _print(results)
    return results


def _print(r):
    print(f"\n===== ADVERSARIAL ROBUSTNESS (deployed thr={r['operating_point']['threshold']:.4f}, "
          f"probe thr={r['probe_operating_point']['threshold']:.4f}) =====")
    print("  recall (%): deployed = zanbil model (OOD-robust); PROBE = attack-pattern detector")
    print("  vs same-domain benign (the honest evasion number); probe-nodec = probe w/o fixed-point decode")
    print(f"  {'mutation':20s} {'deployed':>9s} {'PROBE':>7s} {'probe-nodec':>12s} {'regex':>7s}   n")
    for mt, a in r["aggregate"].items():
        nd = f"{a['probe_recall_nodecode']*100:11.1f}%" if a["probe_recall_nodecode"] is not None else "         n/a"
        print(f"  {mt:20s} {a['deployed_recall']*100:8.1f}% {a['probe_recall']*100:6.1f}% {nd} {a['regex_recall']*100:6.1f}%  {a['n']}")


if __name__ == "__main__":
    main()
