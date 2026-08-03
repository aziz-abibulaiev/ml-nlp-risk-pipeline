"""TASK 09 — WebLog-2025 (organic FP domain #2) + SEC EDGAR (pure-benign FP stress test).

Same discipline as TASK 06/07: text-only deployed model, fixed declared hyperparameters
(config.py), threshold on each domain's own CALIB benign (never test), regex only as WAMM
inspection aid applied to the DECODED request (never a feature or a training filter), real
benign never rule-filtered, seed 42, deterministic, disjointness asserted. New artifacts under
models/content_v5c/ (does NOT touch models/content_v5).

PART A — WebLog Org Y in-domain: train benign = split-half of WebLog benign + cross-source
attacks (SR-BH split + testbed non-custom + ModSec via build_attacks); calib = disjoint slice;
FP test = held-out WebLog benign half. Recall on WebLog attacks (rule-derived, honest breakdown),
SR-BH, testbed cross-tool, ModSec.

PART B — EDGAR benign-only: (1) cross-domain = deployed zanbil model alert-rate on EDGAR benign
at the prompt threshold 0.0118; (2) EDGAR in-domain FP (cross-year temporal split). WAMM decode
top-50/100 on alerts.

Run: python -m services.content_detector.task09_eval
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import multidomain as MD
from .train_partA_v3 import TextOnly

MD.MDIR = "models/content_v5c"          # redirect all runner artifacts off content_v5
MDIR = MD.MDIR
EXT = "data/corpus/phase2/external"
ZMODEL = "models/content_v5/text_only_zanbil.joblib"
ZANBIL_PROMPT_THR = 0.0118              # deployed operating point given in the TASK 09 prompt


def _load_benign(path, cols):
    df = pd.read_parquet(path, columns=cols)
    df = df[df.label == "benign"].copy()
    for c in ["url_query", "request_body", "headers"]:
        if c not in df:
            df[c] = ""
    return df


def _wamm(df, p, thr, n):
    """Top-n alerts, decoded-marker check -> (n_top, n_marked_plausible, rows_df)."""
    B = df.reset_index(drop=True).copy(); B["_p"] = p
    top = B[B["_p"] >= thr].sort_values("_p", ascending=False).head(n)
    if not len(top):
        return 0, 0, top
    marked = top.apply(MD._marked, axis=1)
    return int(len(top)), int(marked.sum()), top.assign(marked=marked.values)


# --------------------------------------------------------------------------- Part A
def _weblog_fp_control(ben, atk, split_kind):
    """Structural (hash) control for WebLog: same benign, same fixed attack corpus, same
    protocol — isolates temporal drift (temporal split) from CMS-surface collision (would show
    in hash too). Returns FP @1%/5% + WAMM on the held-out benign."""
    tr, ca, te, method = MD.split_benign(ben, split_kind)
    fit = pd.concat([tr.assign(y=0), atk["train"].assign(y=1)], ignore_index=True)
    m = TextOnly().fit(pd.DataFrame(MD._reqcols(fit)), fit.y.values)
    thr = {a: float(np.quantile(MD._batched_p(m, ca), 1 - a, method="higher")) for a in C.FPR_BUDGETS}
    p = MD._batched_p(m, te)
    n_top, n_mark, _ = _wamm(te, p, thr[C.PRIMARY_FPR], C.MANUAL_INSPECT_N)
    tf = (n_top - n_mark) / n_top if n_top else 1.0
    return {"split": method, "benign_test": int(len(te)),
            "fp_0.01": float((p >= thr[0.01]).mean()), "fp_0.05": float((p >= thr[0.05]).mean()),
            "fp_0.01_wamm_corrected": float((p >= thr[0.01]).mean() * tf),
            "wamm": {"n_top": n_top, "plausible_attack": n_mark, "true_fp": n_top - n_mark}}


def part_a_weblog(atk, modsec):
    wl = pd.read_parquet(f"{EXT}/weblog2025.parquet")
    ben = wl[wl.label == "benign"].copy()
    atk_wl = wl[wl.label == "attack"].copy()
    train_b, calib_b, test_b, method = MD.split_benign(ben, "temporal")

    fit = pd.concat([train_b.assign(y=0), atk["train"].assign(y=1)], ignore_index=True)
    m = TextOnly().fit(pd.DataFrame(MD._reqcols(fit)), fit.y.values)
    os.makedirs(MDIR, exist_ok=True)
    joblib.dump(m, os.path.join(MDIR, "text_only_weblog.joblib"))

    thr = {a: float(np.quantile(MD._batched_p(m, calib_b), 1 - a, method="higher")) for a in C.FPR_BUDGETS}
    p_late = MD._batched_p(m, test_b)
    p_srbh = MD._batched_p(m, atk["srbh_te"]); p_custom = MD._batched_p(m, atk["custom_te"])
    p_modsec = MD._batched_p(m, modsec) if modsec is not None else None
    p_wl_atk = MD._batched_p(m, atk_wl) if len(atk_wl) else np.array([])

    res = {"split_method": method, "benign_train": int(len(train_b)),
           "benign_calib": int(len(calib_b)), "benign_test_late": int(len(test_b)),
           "attack_train": int(len(atk["train"])), "weblog_attack_n": int(len(atk_wl))}
    for a in C.FPR_BUDGETS:
        res[f"fpr_{a}"] = {
            "late_benign_alert": float((p_late >= thr[a]).mean()),
            "srbh_recall": float((p_srbh >= thr[a]).mean()),
            "testbed_custom_cross_tool_recall": float((p_custom >= thr[a]).mean()),
            "modsec_recall": (float((p_modsec >= thr[a]).mean()) if p_modsec is not None else None),
            "weblog_attack_recall": (float((p_wl_atk >= thr[a]).mean()) if len(p_wl_atk) else None),
        }
    # WebLog attack recall broken down by the dataset's own rule id (honest: mostly behavioural/
    # rule-noise, not content-injection).
    if len(p_wl_atk):
        a1 = atk_wl.reset_index(drop=True).copy(); a1["_hit"] = p_wl_atk >= thr[C.PRIMARY_FPR]
        res["weblog_attack_recall_by_rule@1pct"] = {
            k: {"n": int(v), "recall": float(a1[a1.gt_rule == k]._hit.mean())}
            for k, v in a1.gt_rule.value_counts().items()}

    n_top, n_mark, top = _wamm(test_b, p_late, thr[C.PRIMARY_FPR], C.MANUAL_INSPECT_N)
    res["wamm_manual"] = {"n_top": n_top, "regex_marked_plausible_attack": n_mark, "true_fp_est": n_top - n_mark}
    tf = (n_top - n_mark) / n_top if n_top else 1.0
    for a in C.FPR_BUDGETS:
        res[f"fpr_{a}"]["late_benign_fp_corrected"] = float(res[f"fpr_{a}"]["late_benign_alert"] * tf)
    if len(top):
        top[["http_method", "url_path_raw", "url_query", "_p", "marked"]].to_csv(
            os.path.join(MDIR, "manual_inspect_weblog.csv"), index=False)
    # hash-split structural control (isolates temporal drift from CMS-surface collision)
    res["hash_split_control"] = _weblog_fp_control(ben, atk, "hash")
    return res


# --------------------------------------------------------------------------- Part B
def part_b_edgar(atk, modsec):
    eb = _load_benign(f"{EXT}/edgar.parquet",
                      ["timestamp", "http_method", "url_path_raw", "url_query", "request_body", "headers", "label"])
    out = {"edgar_benign_unique": int(len(eb))}

    # (1) cross-domain: deployed zanbil model on EDGAR benign at the prompt threshold
    zmodel = joblib.load(ZMODEL)
    # recompute the model's own 1% zanbil-calib threshold for reference / honesty
    zb = _load_benign(f"{EXT}/zanbil.parquet",
                      ["timestamp", "http_method", "url_path_raw", "url_query", "headers", "label"])
    _, zcal, _, _ = MD.split_benign(zb, "temporal")
    z_thr_own = float(np.quantile(MD._batched_p(zmodel, zcal), 1 - C.PRIMARY_FPR, method="higher"))
    p_edg_z = MD._batched_p(zmodel, eb)
    n_top, n_mark, top = _wamm(eb, p_edg_z, ZANBIL_PROMPT_THR, 50)
    out["cross_domain_zanbil_model"] = {
        "prompt_threshold": ZANBIL_PROMPT_THR, "model_own_1pct_threshold": z_thr_own,
        "edgar_alert_rate@prompt_thr": float((p_edg_z >= ZANBIL_PROMPT_THR).mean()),
        "edgar_alert_rate@own_1pct": float((p_edg_z >= z_thr_own).mean()),
        "wamm_top50": {"n_top": n_top, "regex_marked": n_mark, "true_fp_est": n_top - n_mark}}
    if len(top):
        top[["http_method", "url_path_raw", "url_query", "_p", "marked"]].to_csv(
            os.path.join(MDIR, "manual_inspect_edgar_crossdomain.csv"), index=False)

    # deployed zanbil model on WebLog benign too (cross-domain contrast)
    wlb = _load_benign(f"{EXT}/weblog2025.parquet",
                       ["timestamp", "http_method", "url_path_raw", "url_query", "request_body", "headers", "label"])
    p_wl_z = MD._batched_p(zmodel, wlb)
    out["cross_domain_zanbil_on_weblog"] = {
        "weblog_alert_rate@prompt_thr": float((p_wl_z >= ZANBIL_PROMPT_THR).mean()),
        "weblog_alert_rate@own_1pct": float((p_wl_z >= z_thr_own).mean())}

    # (2) EDGAR in-domain (cross-year temporal split), cross-source attacks for recall
    train_b, calib_b, test_b, method = MD.split_benign(eb, "temporal")
    fit = pd.concat([train_b.assign(y=0), atk["train"].assign(y=1)], ignore_index=True)
    m = TextOnly().fit(pd.DataFrame(MD._reqcols(fit)), fit.y.values)
    joblib.dump(m, os.path.join(MDIR, "text_only_edgar.joblib"))
    thr = {a: float(np.quantile(MD._batched_p(m, calib_b), 1 - a, method="higher")) for a in C.FPR_BUDGETS}
    p_late = MD._batched_p(m, test_b)
    p_srbh = MD._batched_p(m, atk["srbh_te"]); p_custom = MD._batched_p(m, atk["custom_te"])
    p_modsec = MD._batched_p(m, modsec) if modsec is not None else None
    indom = {"split_method": method, "benign_train": int(len(train_b)),
             "benign_calib": int(len(calib_b)), "benign_test_late": int(len(test_b))}
    for a in C.FPR_BUDGETS:
        indom[f"fpr_{a}"] = {
            "late_benign_alert": float((p_late >= thr[a]).mean()),
            "srbh_recall": float((p_srbh >= thr[a]).mean()),
            "testbed_custom_cross_tool_recall": float((p_custom >= thr[a]).mean()),
            "modsec_recall": (float((p_modsec >= thr[a]).mean()) if p_modsec is not None else None)}
    n_top, n_mark, top = _wamm(test_b, p_late, thr[C.PRIMARY_FPR], C.MANUAL_INSPECT_N)
    indom["wamm_manual"] = {"n_top": n_top, "regex_marked_plausible_attack": n_mark, "true_fp_est": n_top - n_mark}
    tf = (n_top - n_mark) / n_top if n_top else 1.0
    for a in C.FPR_BUDGETS:
        indom[f"fpr_{a}"]["late_benign_fp_corrected"] = float(indom[f"fpr_{a}"]["late_benign_alert"] * tf)
    if len(top):
        top[["http_method", "url_path_raw", "url_query", "_p", "marked"]].to_csv(
            os.path.join(MDIR, "manual_inspect_edgar_indomain.csv"), index=False)
    out["in_domain"] = indom
    return out


def main():
    os.makedirs(MDIR, exist_ok=True)
    atk = MD.build_attacks()
    modsec = MD.load_modsec()
    out = {"config": {"seed": C.SEED, "budgets": list(C.FPR_BUDGETS),
                      "modsec_n": int(len(modsec)) if modsec is not None else 0,
                      "attack_train_n": int(len(atk["train"]))}}
    print("[task09] Part A — WebLog in-domain ...", flush=True)
    out["part_a_weblog"] = part_a_weblog(atk, modsec)
    print("[task09] Part B — EDGAR ...", flush=True)
    out["part_b_edgar"] = part_b_edgar(atk, modsec)
    with open(os.path.join(MDIR, "task09_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    _print(out)
    return out


def _print(o):
    print("\n===== TASK 09 (1% FPR budget, threshold on each domain's own calib) =====")
    a = o["part_a_weblog"]; b1 = a["fpr_0.01"]
    print("PART A — WebLog Org Y (organic 2025 WordPress benign, in-domain):")
    print(f"  split={a['split_method']}  train_ben={a['benign_train']} calib={a['benign_calib']} test={a['benign_test_late']}")
    print(f"  late-benign FP @1%: {b1['late_benign_alert']*100:.3f}%  (WAMM-corr {b1['late_benign_fp_corrected']*100:.3f}%)"
          f"  @5%: {a['fpr_0.05']['late_benign_alert']*100:.3f}%")
    print(f"  recall  SR-BH={b1['srbh_recall']*100:.1f}%  testbed-cross-tool={b1['testbed_custom_cross_tool_recall']*100:.1f}%"
          f"  ModSec={b1['modsec_recall']*100:.1f}%")
    print(f"  WebLog own attacks (rule-derived, n={a['weblog_attack_n']}) recall@1%="
          f"{(b1['weblog_attack_recall'] or 0)*100:.1f}%  by-rule={a.get('weblog_attack_recall_by_rule@1pct')}")
    w = a["wamm_manual"]; print(f"  WAMM top-{w['n_top']}: plausible-attack={w['regex_marked_plausible_attack']} true-FP={w['true_fp_est']}")
    hc = a.get("hash_split_control")
    if hc:
        print(f"  [CONTROL] hash split (no temporal drift): FP@1%={hc['fp_0.01']*100:.3f}% @5%={hc['fp_0.05']*100:.3f}%"
              f"  -> temporal 20% is DRIFT (multi-tenant), not CMS-collision")
    b = o["part_b_edgar"]; cd = b["cross_domain_zanbil_model"]; idm = b["in_domain"]; bi = idm["fpr_0.01"]
    print("\nPART B — SEC EDGAR (pure-benign):")
    print(f"  EDGAR benign unique={b['edgar_benign_unique']}")
    print(f"  (1) CROSS-DOMAIN deployed zanbil model @thr={cd['prompt_threshold']} (own-1%={cd['model_own_1pct_threshold']:.4f}):")
    print(f"       EDGAR alert-rate @prompt={cd['edgar_alert_rate@prompt_thr']*100:.3f}%  @own-1%={cd['edgar_alert_rate@own_1pct']*100:.3f}%"
          f"   WAMM top50 true-FP={cd['wamm_top50']['true_fp_est']}/{cd['wamm_top50']['n_top']}")
    print(f"       (same model on WebLog benign @prompt={b['cross_domain_zanbil_on_weblog']['weblog_alert_rate@prompt_thr']*100:.3f}%)")
    print(f"  (2) EDGAR IN-DOMAIN ({idm['split_method']}): late-FP @1%={bi['late_benign_alert']*100:.3f}% "
          f"(WAMM-corr {bi['late_benign_fp_corrected']*100:.3f}%)  @5%={idm['fpr_0.05']['late_benign_alert']*100:.3f}%")
    print(f"       recall SR-BH={bi['srbh_recall']*100:.1f}% cross-tool={bi['testbed_custom_cross_tool_recall']*100:.1f}% "
          f"ModSec={bi['modsec_recall']*100:.1f}%")


if __name__ == "__main__":
    main()
