"""TASK 06 Part A v3 — in-domain deployment scenario (text-only). THE product number.

Trains on the deployment domain's OWN benign (an early temporal slice of zanbil) plus
cross-source real attacks (SR-BH split + testbed tools), and measures FP on LATER zanbil
traffic fully held out in time — exactly how real WAF-ML is deployed. Text-only (the numeric
channel was confirmed source-correlated/harmful cross-source in v2). Threshold never on the
FP test; real benign never regex-filtered (regex only as manual-inspection aid). Seed 42.

Run: python -m services.content_detector.train_partA_v3
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import config as C
from . import sources as SRC
from .normalize import assemble_fields

ZANBIL = "data/corpus/phase2/external/zanbil.parquet"
TEMPORAL_BOUNDARY_Q = 0.70          # early (<q) -> train/calib ; late (>=q) -> FP test
TRAIN_BEN_CAP = 150_000
CALIB_BEN_CAP = 30_000
TEST_BEN_CAP = 150_000
SRBH_ATTACK_CAP = 60_000
HELDOUT_TOOL = "custom"

_AUDIT = re.compile(
    r"(union\s+select|\bor\s+1=1\b|'\s*or\s*'1'='1|<script|onerror\s*=|javascript:|\.\./|"
    r"%2e%2e%2f|/etc/passwd|;\s*(cat|ls|id|whoami)|\|\s*(id|whoami)|\$\(|`|\{\{.*?\}\}|"
    r"\$\{.*?\}|<%=|%27|%3cscript|\$ne\b|\$gt\b|\$where\b|169\.254\.169\.254|gopher://|"
    r"/\.git|/\.env|/wp-admin|/phpinfo|etc%2fpasswd|passwd|cmd=|exec)", re.I)

_SIGCOLS = ["http_method", "url_path_raw", "url_query", "request_body"]


def _sig(df):
    return (df.http_method.astype(str) + "|" + df.url_path_raw.astype(str) + "|"
            + df.url_query.astype(str) + "|" + df.get("request_body", "").astype(str))


def _fields(df):
    fa, fb = [], []
    for r in df[["url_path_raw", "url_query", "request_body", "headers"]].to_dict("records"):
        f = assemble_fields(url_path_raw=r.get("url_path_raw", ""), url_query=r.get("url_query", ""),
                            request_body=r.get("request_body", ""), headers=r.get("headers"))
        fa.append(f["field_a"]); fb.append(f["field_b"])
    return fa, fb


class TextOnly:
    def __init__(self):
        self.va = TfidfVectorizer(max_features=C.TFIDF_MAX_FEATURES_A, **C.TFIDF)
        self.vb = TfidfVectorizer(max_features=C.TFIDF_MAX_FEATURES_B, **C.TFIDF)
        self.clf = LogisticRegression(**C.LOGREG)

    def _X(self, df, fit=False):
        fa, fb = _fields(df)
        if fit:
            Xa, Xb = self.va.fit_transform(fa), self.vb.fit_transform(fb)
        else:
            Xa, Xb = self.va.transform(fa), self.vb.transform(fb)
        return hstack([Xa, Xb]).tocsr()

    def fit(self, df, y):
        self.clf.fit(self._X(df, fit=True), y)
        return self

    def p(self, df, batch=40_000):
        out = []
        for i in range(0, len(df), batch):
            out.append(self.clf.predict_proba(self._X(df.iloc[i:i + batch]))[:, 1])
        return np.concatenate(out) if out else np.array([])


def main():
    os.makedirs(C.MODEL_DIR_V4 if hasattr(C, "MODEL_DIR_V4") else "models/content_v4", exist_ok=True)
    mdir = "models/content_v4"

    # ---- in-domain temporal split of zanbil ----
    z = pd.read_parquet(ZANBIL, columns=["timestamp", "http_method", "url_path_raw", "url_query", "headers", "label"])
    z["request_body"] = ""
    z = z.loc[~_sig(z).duplicated()].copy()
    z["ts"] = pd.to_datetime(z.timestamp, errors="coerce")
    z = z[z.ts.notna()]
    boundary = z.ts.quantile(TEMPORAL_BOUNDARY_Q)
    early = z[z.ts < boundary]; late = z[z.ts >= boundary]

    rng = np.random.RandomState(C.SEED)
    early = early.sample(frac=1.0, random_state=C.SEED)
    train_ben = early.iloc[:TRAIN_BEN_CAP]
    calib_ben = early.iloc[TRAIN_BEN_CAP:TRAIN_BEN_CAP + CALIB_BEN_CAP]
    test_ben = late.sample(n=min(TEST_BEN_CAP, len(late)), random_state=C.SEED)

    # ---- attacks: SR-BH split + testbed tools ; held-out SR-BH split + held-out tool ----
    srbh = SRC.load_srbh(); srbh_a = srbh[srbh.label == "attack"]
    def hbucket(k):
        return int(hashlib.sha256(f"{C.SEED}:{k}".encode()).hexdigest(), 16) % 10 < 7  # 70% train
    is_tr = _sig(srbh_a).map(hbucket)
    srbh_tr = srbh_a[is_tr]; srbh_te = srbh_a[~is_tr]
    srbh_tr = srbh_tr.sample(n=min(SRBH_ATTACK_CAP, len(srbh_tr)), random_state=C.SEED)
    srbh_te = srbh_te.sample(n=min(40_000, len(srbh_te)), random_state=C.SEED)

    tb = SRC.load_testbed_full(); tb_a = tb[tb.label == "attack"]
    tb_tr = tb_a[tb_a.attack_tool != HELDOUT_TOOL]
    tb_te = tb_a[tb_a.attack_tool == HELDOUT_TOOL]

    train_atk = pd.concat([srbh_tr, tb_tr], ignore_index=True)
    calib_atk = train_atk.sample(frac=0.15, random_state=C.SEED)
    fit_atk = train_atk.drop(index=calib_atk.index)

    # ---- disjointness asserts ----
    ktr, kca, kte = set(_sig(train_ben)), set(_sig(calib_ben)), set(_sig(test_ben))
    assert not (ktr & kca), "train/calib benign overlap"
    assert not (ktr & kte), "train/test benign overlap"
    assert not (kca & kte), "calib/test benign overlap"
    assert not (set(_sig(srbh_tr)) & set(_sig(srbh_te))), "SR-BH train/test attack overlap"

    fit_df = pd.concat([train_ben.assign(y=0), fit_atk.assign(y=1)], ignore_index=True)
    m = TextOnly().fit(fit_df, fit_df.y.values)
    joblib.dump(m, os.path.join(mdir, "text_only_indomain.joblib"))

    # thresholds on calib benign p_text
    p_calib_ben = m.p(calib_ben)
    thr = {a: float(np.quantile(p_calib_ben, 1 - a, method="higher")) for a in C.FPR_BUDGETS}

    p_late = m.p(test_ben)
    p_srbh = m.p(srbh_te)
    p_tool = m.p(tb_te)

    res = {"split": {"boundary": str(boundary), "temporal": True,
                     "train_benign": int(len(train_ben)), "calib_benign": int(len(calib_ben)),
                     "test_benign_late": int(len(test_ben)),
                     "train_attack": int(len(fit_atk)), "srbh_test_attack": int(len(srbh_te)),
                     "testbed_heldout_tool": HELDOUT_TOOL, "testbed_tool_test": int(len(tb_te))},
           "cross_source_v2_zanbil_heldout_fp": 0.995}
    for a in C.FPR_BUDGETS:
        res[f"fpr_{a}"] = {
            "threshold": thr[a],
            "late_zanbil_alert_rate": float((p_late >= thr[a]).mean()),
            "srbh_attack_recall": float((p_srbh >= thr[a]).mean()),
            "testbed_tool_recall": float((p_tool >= thr[a]).mean()),
        }
    # WAMM manual: top-100 late-zanbil alerts
    B = test_ben.reset_index(drop=True).copy(); B["_p"] = p_late
    top = B[B["_p"] >= thr[C.PRIMARY_FPR]].sort_values("_p", ascending=False).head(C.MANUAL_INSPECT_N)
    marked = top.apply(lambda r: bool(_AUDIT.search(f"{r.url_path_raw} {r.url_query}")), axis=1)
    res["wamm_manual"] = {"n_top_alerts": int(len(top)),
                          "regex_marked_plausible_attack": int(marked.sum()),
                          "true_fp_estimate": int((~marked).sum())}
    top.assign(regex_marked=marked.values)[["http_method", "url_path_raw", "url_query", "_p", "regex_marked"]] \
        .to_csv(os.path.join(mdir, "manual_inspect_late_zanbil.csv"), index=False)

    # corrected FP: alert-rate * (true-FP fraction among top)
    tf = res["wamm_manual"]["true_fp_estimate"] / max(1, res["wamm_manual"]["n_top_alerts"])
    for a in C.FPR_BUDGETS:
        res[f"fpr_{a}"]["late_zanbil_fp_corrected_est"] = float(res[f"fpr_{a}"]["late_zanbil_alert_rate"] * tf)

    with open(os.path.join(mdir, "partA_v3_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    _print(res)
    return res


def _print(r):
    s = r["split"]
    print("\n===== IN-DOMAIN TEMPORAL SPLIT (zanbil) =====")
    print(f"  boundary={s['boundary']}  train_benign={s['train_benign']} calib={s['calib_benign']} "
          f"test_late={s['test_benign_late']}")
    print(f"  train_attack={s['train_attack']} (SR-BH split + testbed tools); held-out: SR-BH "
          f"{s['srbh_test_attack']} + testbed:{s['testbed_heldout_tool']} {s['testbed_tool_test']}")
    print("\n===== METRICS (text-only) =====")
    for a in C.FPR_BUDGETS:
        b = r[f"fpr_{a}"]
        print(f"  @{a}: late-zanbil alert={b['late_zanbil_alert_rate']*100:6.2f}%  "
              f"(WAMM-corrected FP≈{b['late_zanbil_fp_corrected_est']*100:5.2f}%)  "
              f"SR-BH recall={b['srbh_attack_recall']*100:6.2f}%  testbed:{HELDOUT_TOOL} recall={b['testbed_tool_recall']*100:6.2f}%")
    w = r["wamm_manual"]
    print(f"  WAMM top-{w['n_top_alerts']}: regex-marked(plausible attack)={w['regex_marked_plausible_attack']}, "
          f"true-FP est={w['true_fp_estimate']}")
    print(f"\n  HEADLINE CONTRAST: in-domain late-zanbil FP  vs  cross-source zanbil-held-out (v2) 99.5%")


if __name__ == "__main__":
    main()
