"""Cross-corpus transfer evaluation: CSIC-trained models on SR-BH 2020.

Configurations (model weights are frozen throughout — nothing is retrained):

  T1  in-domain baseline   CSIC calib (data/ml_validation.csv) -> CSIC test (data/ml_test.csv)
  T2  naive transfer       CSIC calib -> SR-BH random test
  T3  naive transfer       CSIC calib -> SR-BH temporal test
  T4  recalibrated         SR-BH random calib -> SR-BH random test
  T4t recalibrated         SR-BH temporal calib -> SR-BH temporal test

T1-T3 use each model's stored calibration from models/ml/*_model.meta.json, applied by
the same production code path (score_from_stat_with_calib). T4/T4t re-fit ONLY the
calibration layer on the SR-BH calibration slice: all three branches the pipeline
supports (Platt/attack-logistic with temperature, ECDF quantile mapping, mu/sigma with
sigmoid), selected by NLL exactly as production does
(_select_best_runtime_calibration_candidate). Refit calibrations are written to
models/ml/srbh_recalib/ as new files; nothing existing under models/ is modified.

Prior correction: T1-T3 keep the stored CSIC prior correction (0.2 -> 0.05) because it
is part of the deployed artifact being transferred. T4/T4t attach none, matching the
production default when no deployment prior is configured (ML_DEPLOY_ATTACK_PRIOR unset).
This is stated in the report because it affects Brier/NLL/ECE comparisons.

Thresholds: FPR budgets alpha in {5%, 1%, 0.5%, 0.1%}, always derived from the
configuration's calibration slice (threshold_for_target_fpr), never from test. Each test
slice is scored once per configuration; bootstrap CIs use the existing
_bootstrap_ci_for_threshold_metrics machinery (2,000 resamples, seed 42, 50,000-row cap).

Recovery fraction (Part C), per metric m and budget alpha:

    R(m) = (m_T4 - m_T2) / (m_T1 - m_T2)      computed when |m_T1 - m_T2| > 1e-9

i.e. the share of the in-domain -> naive-transfer degradation that recalibration alone
recovers. R=1 means fully recovered to the in-domain level, R=0 means recalibration
bought nothing, R<0 means it hurt. The same formula with T4t/T3 gives the temporal
variant. Note T1 is measured on CSIC test and T2/T4 on SR-BH test; R treats T1 as the
in-domain reference level, which is the deployment question being asked.

Deterministic: seed 42 everywhere; two runs produce identical JSON.

Usage:
    python jobs/retrain/eval_srbh_transfer.py            # full run
    python jobs/retrain/eval_srbh_transfer.py --no-ci    # skip bootstrap (fast pass)
"""

import argparse
import json
import logging
import os
import pathlib
import sys

import numpy as np
import pandas as pd

# --- import the production training/eval module outside its Docker environment ---
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "jobs", "retrain"))

_orig_mkdir = pathlib.Path.mkdir


def _safe_mkdir(self, *a, **k):
    try:
        return _orig_mkdir(self, *a, **k)
    except OSError:
        return None


pathlib.Path.mkdir = _safe_mkdir  # main.py creates /models at import time (Docker path)
import main as m  # noqa: E402

pathlib.Path.mkdir = _orig_mkdir

from ml_http_features import FEATURE_COLS  # noqa: E402

log = logging.getLogger("eval_srbh_transfer")

SEED = 42
ALPHAS = [0.05, 0.01, 0.005, 0.001]
N_BOOT = 2000
BOOT_MAX_ROWS = 50000

MODELS_DIR = os.path.join(_REPO, "models", "ml")
RECALIB_DIR = os.path.join(MODELS_DIR, "srbh_recalib")
SPLITS_DIR = os.path.join(_REPO, "data", "srbh2020", "splits")
OUT_DIR = os.path.join(_REPO, "data", "srbh2020", "transfer")
FIG_DIR = os.path.join(_REPO, "docs", "figures", "srbh_recalib")

NOVELTY_KINDS = ["if", "ocsvm", "ae", "lstm_ae"]
SUPERVISED_KINDS = ["sup_lr", "sup_hgb", "sup_rf", "sup_et"]
ALL_KINDS = NOVELTY_KINDS + SUPERVISED_KINDS

# Ensemble membership and post-aggregation calibration, from ml_ensemble.meta.json.
PRIMARY_MEMBERS = ["sup_hgb", "sup_rf", "sup_et"]
AUX_MEMBERS = ["if", "ocsvm", "ae", "lstm_ae"]


def load_artifacts() -> dict:
    import joblib

    art = {}
    ens_meta = json.load(open(os.path.join(MODELS_DIR, "ml_ensemble.meta.json")))
    for kind in ALL_KINDS:
        meta = json.load(open(os.path.join(MODELS_DIR, f"{kind}_model.meta.json")))
        if list(meta["feature_order"]) != list(FEATURE_COLS):
            raise RuntimeError(f"{kind}: feature_order in meta disagrees with FEATURE_COLS")
        bundle = joblib.load(os.path.join(MODELS_DIR, f"{kind}_model.joblib"))
        art[kind] = {"model": bundle, "meta": meta, "calib": dict(meta.get("calibration") or {})}
    art["_ensemble_meta"] = ens_meta
    if ens_meta["metrics"]["primary"]["selected_models"] != PRIMARY_MEMBERS:
        raise RuntimeError("primary ensemble membership drifted from expectation")
    if ens_meta["metrics"]["aux_novelty"]["selected_models"] != AUX_MEMBERS:
        raise RuntimeError("aux ensemble membership drifted from expectation")
    return art


def _keras_forward(model, arr: np.ndarray, batch: int = 65536) -> np.ndarray:
    """Batched inference-mode forward pass.

    Deliberately NOT model.predict(): keras 3.15 over tensorflow 2.20 (the version pair
    required to deserialize these artifacts) deadlocks inside predict's tf.data loop.
    Calling the model directly is the same forward pass in inference mode and returns
    identical values, without that pipeline.
    """
    outs = []
    for i in range(0, len(arr), batch):
        outs.append(np.asarray(model(arr[i : i + batch], training=False), dtype=float))
    return np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]


def raw_stats(kind: str, bundle, X: np.ndarray) -> np.ndarray:
    """Raw pre-calibration statistic, matching _collect_ensemble_model_scores."""
    if kind in ("if", "ocsvm"):
        return m._raw_anomaly_stat(bundle, X, kind=kind)
    if kind == "ae":
        X_sc = bundle["scaler"].transform(X)
        X_rec = _keras_forward(bundle["model"], X_sc)
        return np.mean((X_sc - X_rec) ** 2, axis=1)
    if kind == "lstm_ae":
        X_sc = bundle["scaler"].transform(X)
        X_seq = X_sc.reshape((-1, int(bundle["timesteps"]), 1))
        X_rec = _keras_forward(bundle["model"], X_seq)
        return np.mean((X_seq - X_rec) ** 2, axis=(1, 2))
    if kind in SUPERVISED_KINDS:
        return m._supervised_raw_scores(bundle, X)
    raise ValueError(kind)


def ensemble_score(per_model: dict[str, np.ndarray], members: list[str], ens_calib: dict) -> np.ndarray:
    s = m._aggregate_prob_arrays([per_model[k] for k in members], method="mean_logit")
    t = ens_calib.get("temperature")
    if t is not None:
        s = m._apply_temperature_scaling(s, float(t))
    pc = ens_calib.get("prior_correction")
    if isinstance(pc, dict) and pc.get("enabled"):
        s = m._apply_prior_correction_probs(s, pc)
    return s


# ---------------------------------------------------------------- Part C: refit


def refit_calibration(kind: str, stats_cal: np.ndarray, y_cal: np.ndarray) -> dict:
    """Re-fit the calibration layer only, on the SR-BH calibration slice.

    Candidates mirror production: Platt/attack-logistic (+temperature), ECDF quantile
    mapping fit on benign-only stats (+temperature), mu/sigma (median/IQR) with sigmoid
    (+temperature); supervised models additionally get the raw_prob(+temperature)
    candidate production uses. Selection by NLL-first ranking, exactly
    _select_best_runtime_calibration_candidate.
    """
    orientation = m._default_score_orientation(m._normalize_anomaly_kind(kind))
    oriented = m._oriented_stats(np.asarray(stats_cal, float).reshape(-1), orientation)
    benign_oriented = oriented[y_cal == 0]
    candidates = []

    def add_candidate(name: str, calib: dict) -> None:
        scores = m.score_from_stat_with_calib(stats_cal, calib, kind=kind)
        t, t_metrics = m.fit_temperature_scaler(scores, y_cal, kind=f"{kind}_{name}")
        if t_metrics.get("used"):
            calib = dict(calib, temperature=float(t))
            scores = m._apply_temperature_scaling(scores, t)
        candidates.append(
            {
                "name": name,
                "calibration": calib,
                "scores": scores,
                "eval": m._eval_scores_only_with_labels(scores, y_cal),
                "temperature_scaling": t_metrics,
            }
        )

    if kind in SUPERVISED_KINDS:
        add_candidate("raw_prob", {"orientation": orientation})

    logit_params, _, _ = m.fit_attack_logistic_calibrator(
        stats_cal, y_cal, kind=kind, orientation=orientation, standardize_input=True
    )
    if logit_params is not None:
        add_candidate("platt", {"orientation": orientation, "attack_logistic": dict(logit_params)})

    ecdf_map = m.fit_ecdf_risk_mapping(benign_oriented)
    if ecdf_map is not None:
        add_candidate("ecdf", {"orientation": orientation, "ecdf_risk": ecdf_map})

    mu = float(np.median(benign_oriented))
    q25, q75 = np.quantile(benign_oriented, [0.25, 0.75])
    sigma = float((q75 - q25) / 1.349)
    if (not np.isfinite(sigma)) or sigma <= 1e-9:
        sigma = float(np.std(benign_oriented) or 1e-6)
    add_candidate("mu_sigma", {"orientation": orientation, "mu": mu, "sigma": sigma, "scale_method": "median_iqr"})

    best, comparison, reason = m._select_best_runtime_calibration_candidate(candidates)
    return {
        "kind": kind,
        "selected": best.get("name"),
        "selection_reason": reason,
        "calibration": best.get("calibration"),
        "candidates": {
            k: {kk: vv for kk, vv in v.items() if kk in ("log_loss", "brier", "ece_15", "roc_auc", "pr_auc")}
            for k, v in comparison.items()
        },
    }


# ---------------------------------------------------------------- evaluation


def eval_config(
    name: str,
    scores_cal: dict[str, np.ndarray],
    y_cal: np.ndarray,
    scores_test: dict[str, np.ndarray],
    y_test: np.ndarray,
    with_ci: bool,
    csic_thresholds: dict[str, float] | None = None,
) -> dict:
    out = {"name": name, "scorers": {}}
    for scorer, s_test in scores_test.items():
        s_cal = scores_cal[scorer]
        entry = {
            "calibration_quality_test": {
                k: v
                for k, v in m._eval_scores_only_with_labels(s_test, y_test).items()
                if k in ("roc_auc", "pr_auc", "log_loss", "brier", "ece_15", "score_p01", "score_p99")
            },
            "alphas": {},
        }
        for alpha in ALPHAS:
            thr, thr_info = m.threshold_for_target_fpr(y_cal, s_cal, alpha)
            if thr is None:
                entry["alphas"][str(alpha)] = {"error": "threshold_unavailable"}
                continue
            mm = m._binary_metrics_at_threshold(y_test, s_test, thr)
            rec = {
                "threshold": float(thr),
                "calib_actual_fpr": thr_info.get("actual_fpr"),
                "roc_auc": mm.get("roc_auc"),
                "pr_auc": mm.get("pr_auc"),
                "f1": mm.get("f1"),
                "precision": mm.get("precision"),
                "recall": mm.get("recall"),
                "fpr": mm.get("fpr"),
                "tp": mm["tp"], "fp": mm["fp"], "fn": mm["fn"], "tn": mm["tn"],
            }
            if with_ci:
                ci = m._bootstrap_ci_for_threshold_metrics(
                    y_test, s_test, thr=thr, n_boot=N_BOOT, alpha=0.05,
                    random_state=SEED, max_rows=BOOT_MAX_ROWS,
                )
                if ci:
                    rec["ci95"] = {
                        k: [round(v["low"], 6), round(v["high"], 6)]
                        for k, v in ci["metrics"].items()
                        if k in ("roc_auc", "pr_auc", "f1", "precision", "recall", "fpr")
                    }
            entry["alphas"][str(alpha)] = rec
        if csic_thresholds and scorer in csic_thresholds:
            thr0 = float(csic_thresholds[scorer])
            mm0 = m._binary_metrics_at_threshold(y_test, s_test, thr0)
            entry["csic_threshold_unchanged"] = {
                "threshold": thr0,
                "realised_fpr": mm0.get("fpr"),
                "recall": mm0.get("recall"),
                "precision": mm0.get("precision"),
                "f1": mm0.get("f1"),
            }
        out["scorers"][scorer] = entry
    return out


def main_cli(argv=None) -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ci", action="store_true", help="skip bootstrap CIs (fast pass)")
    args = ap.parse_args(argv)
    with_ci = not args.no_ci

    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RECALIB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    art = load_artifacts()
    ens_meta = art["_ensemble_meta"]
    primary_ens_calib = dict(ens_meta.get("calibration") or {})
    aux_ens_calib = dict(ens_meta.get("aux_novelty_calibration") or {})

    datasets = {
        "csic_calib": pd.read_csv(os.path.join(_REPO, "data", "ml_validation.csv"), low_memory=False),
        "csic_test": pd.read_csv(os.path.join(_REPO, "data", "ml_test.csv"), low_memory=False),
        "srbh_calib": pd.read_parquet(os.path.join(SPLITS_DIR, "srbh_calib.parquet")),
        "srbh_test": pd.read_parquet(os.path.join(SPLITS_DIR, "srbh_test.parquet")),
        "srbh_t_calib": pd.read_parquet(os.path.join(SPLITS_DIR, "srbh_temporal_calib.parquet")),
        "srbh_t_test": pd.read_parquet(os.path.join(SPLITS_DIR, "srbh_temporal_test.parquet")),
    }
    X = {k: m.ensure_columns(df, FEATURE_COLS).fillna(0.0).astype(float).to_numpy() for k, df in datasets.items()}
    Y = {k: np.asarray(df["label"], dtype=int) for k, df in datasets.items()}

    # ---- raw statistics, computed once per (model, dataset) and cached on disk ----
    stats_path = os.path.join(OUT_DIR, "raw_stats_cache.npz")
    if os.path.exists(stats_path):
        cached = dict(np.load(stats_path))
        log.info("loaded raw-stats cache: %s arrays", len(cached))
    else:
        cached = {}
    for kind in ALL_KINDS:
        for ds in datasets:
            key = f"{kind}__{ds}"
            if key not in cached:
                log.info("raw stats: %s on %s (%s rows)", kind, ds, len(X[ds]))
                cached[key] = raw_stats(kind, art[kind]["model"], X[ds])
        np.savez_compressed(stats_path, **cached)

    def stats_of(kind, ds):
        return np.asarray(cached[f"{kind}__{ds}"], dtype=float).reshape(-1)

    # ---- calibrated scores under the stored CSIC calibration (T1-T3) ----
    def csic_scores(ds: str) -> dict[str, np.ndarray]:
        per = {k: m.score_from_stat_with_calib(stats_of(k, ds), art[k]["calib"], kind=k) for k in ALL_KINDS}
        per["ml_ensemble"] = ensemble_score(per, PRIMARY_MEMBERS, primary_ens_calib)
        per["ml_aux_novelty"] = ensemble_score(per, AUX_MEMBERS, aux_ens_calib)
        return per

    # ---- Part C: refit calibration on the SR-BH calib slices ----
    def refit_family(calib_ds: str, tag: str) -> tuple[dict, dict]:
        refits = {}
        for kind in ALL_KINDS:
            r = refit_calibration(kind, stats_of(kind, calib_ds), Y[calib_ds])
            refits[kind] = r
            with open(os.path.join(RECALIB_DIR, f"{kind}_{tag}_calibration.json"), "w") as fh:
                json.dump(r, fh, indent=2, default=float)
            log.info("[recalib:%s] %s -> %s (%s)", tag, kind, r["selected"], r["selection_reason"])

        def rescored(ds: str) -> dict[str, np.ndarray]:
            per = {
                k: m.score_from_stat_with_calib(stats_of(k, ds), refits[k]["calibration"], kind=k)
                for k in ALL_KINDS
            }
            for ens_name, members in (("ml_ensemble", PRIMARY_MEMBERS), ("ml_aux_novelty", AUX_MEMBERS)):
                s_cal = m._aggregate_prob_arrays([
                    m.score_from_stat_with_calib(stats_of(k, calib_ds), refits[k]["calibration"], kind=k)
                    for k in members
                ], method="mean_logit")
                t, t_metrics = m.fit_temperature_scaler(s_cal, Y[calib_ds], kind=f"{ens_name}_{tag}")
                ens_calib = {"temperature": float(t)} if t_metrics.get("used") else {}
                refits[f"_{ens_name}"] = {"calibration": ens_calib, "temperature_scaling": t_metrics}
                per[ens_name] = ensemble_score(per, members, ens_calib)
            return per

        return refits, {"calib": rescored(calib_ds), }

    refits_rand, t4_partial = refit_family("srbh_calib", "random")
    refits_temp, t4t_partial = refit_family("srbh_t_calib", "temporal")

    def refit_scores(refits: dict, calib_ds: str, ds: str) -> dict[str, np.ndarray]:
        per = {
            k: m.score_from_stat_with_calib(stats_of(k, ds), refits[k]["calibration"], kind=k)
            for k in ALL_KINDS
        }
        for ens_name, members in (("ml_ensemble", PRIMARY_MEMBERS), ("ml_aux_novelty", AUX_MEMBERS)):
            ens_calib = refits[f"_{ens_name}"]["calibration"]
            per[ens_name] = ensemble_score(per, members, ens_calib)
        return per

    # CSIC alpha=5% recommended thresholds, applied unchanged (deployment-risk number).
    csic_thr = {k: float(art[k]["meta"]["recommended_threshold"]) for k in ALL_KINDS}
    csic_thr["ml_ensemble"] = float(ens_meta["recommended_threshold"])
    csic_thr["ml_aux_novelty"] = float(ens_meta["recommended_aux_novelty_threshold"])

    results = {}
    sc_csic_cal = csic_scores("csic_calib")
    results["T1"] = eval_config("T1", sc_csic_cal, Y["csic_calib"], csic_scores("csic_test"), Y["csic_test"], with_ci)
    results["T2"] = eval_config(
        "T2", sc_csic_cal, Y["csic_calib"], csic_scores("srbh_test"), Y["srbh_test"], with_ci, csic_thresholds=csic_thr
    )
    results["T3"] = eval_config(
        "T3", sc_csic_cal, Y["csic_calib"], csic_scores("srbh_t_test"), Y["srbh_t_test"], with_ci, csic_thresholds=csic_thr
    )
    results["T4"] = eval_config(
        "T4",
        refit_scores(refits_rand, "srbh_calib", "srbh_calib"),
        Y["srbh_calib"],
        refit_scores(refits_rand, "srbh_calib", "srbh_test"),
        Y["srbh_test"],
        with_ci,
    )
    results["T4t"] = eval_config(
        "T4t",
        refit_scores(refits_temp, "srbh_t_calib", "srbh_t_calib"),
        Y["srbh_t_calib"],
        refit_scores(refits_temp, "srbh_t_calib", "srbh_t_test"),
        Y["srbh_t_test"],
        with_ci,
    )

    # ---- recovery fractions ----
    def recovery(base: str, naive: str, recal: str) -> dict:
        out = {}
        for scorer in results[naive]["scorers"]:
            per_alpha = {}
            for alpha in ALPHAS:
                a = str(alpha)
                t1 = results[base]["scorers"][scorer]["alphas"].get(a, {})
                t2 = results[naive]["scorers"][scorer]["alphas"].get(a, {})
                t4 = results[recal]["scorers"][scorer]["alphas"].get(a, {})
                row = {}
                for metric in ("roc_auc", "pr_auc", "f1", "precision", "recall"):
                    v1, v2, v4 = t1.get(metric), t2.get(metric), t4.get(metric)
                    if v1 is None or v2 is None or v4 is None or abs(v1 - v2) <= 1e-9:
                        row[metric] = None
                    else:
                        row[metric] = round((v4 - v2) / (v1 - v2), 4)
                per_alpha[a] = row
            cq = {}
            for metric in ("log_loss", "brier", "ece_15"):
                v1 = results[base]["scorers"][scorer]["calibration_quality_test"].get(metric)
                v2 = results[naive]["scorers"][scorer]["calibration_quality_test"].get(metric)
                v4 = results[recal]["scorers"][scorer]["calibration_quality_test"].get(metric)
                if v1 is None or v2 is None or v4 is None or abs(v1 - v2) <= 1e-9:
                    cq[metric] = None
                else:
                    cq[metric] = round((v4 - v2) / (v1 - v2), 4)
            out[scorer] = {"per_alpha": per_alpha, "calibration_quality": cq}
        return out

    results["recovery_T1_T2_T4"] = recovery("T1", "T2", "T4")
    results["recovery_T1_T3_T4t"] = recovery("T1", "T3", "T4t")
    results["recovery_formula"] = "R(m) = (m_T4 - m_T2) / (m_T1 - m_T2); guarded when |m_T1 - m_T2| <= 1e-9"
    results["config"] = {
        "seed": SEED, "alphas": ALPHAS, "n_boot": N_BOOT, "boot_max_rows": BOOT_MAX_ROWS,
        "primary_members": PRIMARY_MEMBERS, "aux_members": AUX_MEMBERS,
        "prior_correction": "T1-T3: stored CSIC prior correction (0.2->0.05); T4/T4t: none (production default)",
        "with_ci": with_ci,
    }
    results["recalib_selection"] = {
        "random": {k: {"selected": v["selected"], "candidates": v["candidates"]} for k, v in refits_rand.items() if not k.startswith("_")},
        "temporal": {k: {"selected": v["selected"], "candidates": v["candidates"]} for k, v in refits_temp.items() if not k.startswith("_")},
    }

    out_json = os.path.join(OUT_DIR, "srbh_transfer_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    log.info("results -> %s", out_json)

    # ---- reliability diagrams: T2 vs T4, per ensemble and two representative models ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def reliability(ax, scores, y, label):
        bins = np.linspace(0, 1, 16)
        mids, accs, confs = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (scores >= lo) & (scores < hi) if hi < 1 else (scores >= lo) & (scores <= hi)
            if sel.sum() < 10:
                continue
            mids.append((lo + hi) / 2)
            accs.append(float(np.mean(y[sel])))
            confs.append(float(np.mean(scores[sel])))
        ax.plot(confs, accs, marker="o", label=label)

    sc_t2 = csic_scores("srbh_test")
    sc_t4 = refit_scores(refits_rand, "srbh_calib", "srbh_test")
    y_te = Y["srbh_test"]
    for scorer in ("ml_ensemble", "ml_aux_novelty", "sup_hgb", "if"):
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        reliability(ax, sc_t2[scorer], y_te, "T2 naive (CSIC calibration)")
        reliability(ax, sc_t4[scorer], y_te, "T4 recalibrated (SR-BH calib)")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed attack frequency")
        ax.set_title(f"Reliability on SR-BH test: {scorer}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"reliability_{scorer}_t2_vs_t4.png"), dpi=150)
        plt.close(fig)
    log.info("reliability diagrams -> %s", FIG_DIR)

    print(json.dumps({"done": True, "out": out_json}, indent=2))


if __name__ == "__main__":
    main_cli()
