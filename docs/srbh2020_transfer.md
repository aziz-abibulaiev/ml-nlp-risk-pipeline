# CSIC → SR-BH 2020: transfer, and what recalibration can and cannot recover

Companion to `jobs/retrain/build_srbh_splits.py`, `jobs/retrain/eval_srbh_transfer.py`
and `jobs/retrain/eval_encoding_depth.py`. Continues TASK 01
(`docs/srbh2020_mapping.md`, `docs/srbh2020_feature_report.md`).

Nothing was retrained. Model weights are frozen throughout; only the calibration layer is
re-fit, and only in T4/T4t, written to `models/ml/srbh_recalib/` as new files.

---

## Headline

**None of the ranking loss in cross-corpus transfer is a calibration problem.**

Recalibration on an SR-BH slice recovers essentially **100 % of the probability-quality
loss** (ECE 0.636 → 0.018 on the primary ensemble) and **0 % of the discriminative loss**
(ROC-AUC 0.673 → 0.663; recovery fraction −0.031). This is not a surprise once stated
plainly — Platt, ECDF and μ/σ are all *monotone* maps, and a monotone map cannot change
the ordering of scores, so it cannot change ROC-AUC. What the experiment establishes is
the *magnitude*: the entire 0.984 → 0.673 collapse is representation, not scale.

The single exception is instructive and is discussed under "Saturation" below.

---

## Part A — splits

### Deduplication: SR-BH is far more redundant than CSIC

| | rows | duplicate rate |
|---|---|---|
| whole corpus | 907,815 → 508,821 | **43.95 %** |
| benign only | 525,195 | **70.48 %** |
| attacks only | 382,620 | **7.53 %** |

For reference, `data/ml_train.csv` carries 19.5 %, which was never removed.

Benign SR-BH traffic is 70 % duplicated at the feature-vector level. This is exactly what
an internet-exposed honeypot produces: the same crawler fetching the same handful of
WordPress paths tens of thousands of times. **Any evaluation on this corpus that does not
deduplicate is reporting the crawler's repetition rate, not detection quality.**

2,910 feature vectors (11,022 rows) carry *both* labels — identical 24-feature
representations that are benign in one row and attack in another. These are resolved
first-occurrence-wins, and they place a hard ceiling on achievable accuracy in this
feature space: those rows cannot all be classified correctly by any model reading only
these 24 features.

Deduplication is applied **before** splitting, so a repeated request cannot straddle a
boundary. Pairwise feature-vector disjointness is asserted across every split pair in both
families and the build fails loudly otherwise.

### ⚠️ The temporal split does not do what the task assumed

The corpus is **not** 12 evenly-covered days of July 2020. Actual distribution
(deduplicated):

| Day | rows | attacks |
|---|---|---|
| 2020-07-17 | 169,223 | 131,790 |
| 2020-07-18 | 284,817 | 197,374 |
| 2020-07-19 | 37,166 | 24,631 |
| 2020-07-20 … 07-27 | 16,439 total | **0** |
| **2020-11-01** | 1,175 | 1 |

Two facts break the intended design:

1. **97 % of rows fall in the first three days**, and 89 % in the first two. A "last 20 %
   of rows" cut lands at **2020-07-18 17:00:48**, i.e. *inside* day two, mid-campaign —
   not at a day boundary.
2. **Every row after 2020-07-19 is benign** (one attack on 1 November excepted). Any cut
   placed at a natural day boundary produces an attack-free test set and no measurable
   recall.

There is also a stray **1 November 2020** batch, three months after the stated capture
window, which the dataset publication does not mention.

The chosen cut is therefore the least-bad option, not a clean one:

| Window | rows | boundary |
|---|---|---|
| train | 356,175 | … → 2020-07-18 13:21:19+02:00 |
| calib | 50,882 | 2020-07-18 13:21:19 → 2020-07-18 17:00:48 |
| **test** | 101,764 | **2020-07-18 17:00:48+02:00** → 2020-11-01 22:59:09+01:00 |

**Consequence for interpretation.** T3 differs from T2 substantially, but this cannot be
attributed to campaign leakage alone, because the two test sets have different class
composition (85.1 % attacks in the random test, 70.1 % in the temporal test) and different
traffic character (the temporal test contains all the attack-free later days). T3 vs T2 is
a *composition* comparison at least as much as a *leakage* comparison. Reported for both,
with that caveat attached; I would not claim the temporal split is "the honest one" on
this corpus.

---

## Part B/C — transfer and recalibration

Configurations. Weights frozen throughout.

| | Trained | Calibrated | Tested |
|---|---|---|---|
| T1 | CSIC | CSIC (`ml_validation.csv`) | CSIC (`ml_test.csv`) |
| T2 | CSIC | CSIC | SR-BH random test |
| T3 | CSIC | CSIC | SR-BH temporal test |
| T4 | CSIC | **SR-BH random calib** | SR-BH random test |
| T4t | CSIC | **SR-BH temporal calib** | SR-BH temporal test |

Thresholds come from each configuration's calibration slice at FPR budgets
α ∈ {5 %, 1 %, 0.5 %, 0.1 %}; test slices are scored once per configuration. Bootstrap
95 % CIs use the repo's `_bootstrap_ci_for_threshold_metrics` (2,000 resamples, seed 42,
50,000-row cap). T1 reproduces the stored `ml_ensemble.meta.json` test numbers exactly
(ROC-AUC 0.9843, PR-AUC 0.9511, F1 0.8400, FPR 0.0512), which validates the harness.

### 🚨 The deployment-risk number: CSIC thresholds on real traffic

Each model's own `recommended_threshold` from its meta file, applied unchanged to SR-BH:

| Scorer | CSIC threshold | realised FPR (random) | realised FPR (temporal) |
|---|---|---|---|
| `ml_aux_novelty` | 0.04657 | **100.00 %** | **100.00 %** |
| `sup_rf` | 0.12305 | **99.99 %** | **100.00 %** |
| `sup_hgb` | 0.10438 | **99.99 %** | **99.99 %** |
| `ml_ensemble` | 0.03845 | **99.81 %** | **99.96 %** |
| `if` | 0.18587 | **97.88 %** | **91.28 %** |
| `ocsvm` | 0.07114 | **95.21 %** | **80.62 %** |
| `lstm_ae` | 0.09862 | **96.54 %** | **88.81 %** |
| `ae` | 0.05218 | **100.00 %** | **100.00 %** |
| `sup_lr` | 0.20140 | **37.06 %** | **62.26 %** |
| `sup_et` | 0.28866 | **7.93 %** | **7.72 %** |

A threshold tuned for 5 % FPR on CSIC flags **essentially every benign request** in real
honeypot traffic. The system would alert on 100 % of normal traffic on day one. Only
`sup_et` degrades gracefully (7.9 % vs the 5 % budget), and that is an accident of its
higher operating threshold rather than better transfer — its ROC-AUC drops to 0.681 like
everything else.

This is the practical finding for the paper's deployment-guidance section: **a calibration
fitted on one corpus provides no FPR guarantee whatsoever on another**, and the failure is
not graceful.

### Discriminative collapse, with 95 % CIs

ROC-AUC at α = 5 %, bootstrap percentile CIs (2,000 resamples, seed 42). Full tables for
all four budgets and all ten scorers are in
`data/srbh2020/transfer/srbh_transfer_results.json`.

| Scorer | T1 (CSIC) | T2 (SR-BH random) | T3 (SR-BH temporal) | T4 (recalibrated) |
|---|---|---|---|---|
| `ml_ensemble` | 0.9843 [0.982, 0.986] | 0.6727 [0.665, 0.680] | 0.6939 [0.690, 0.702] | 0.6632 [0.656, 0.671] |
| `ml_aux_novelty` | 0.9170 [0.911, 0.923] | 0.6949 [0.690, 0.703] | 0.9402 [0.938, 0.943] | 0.7290 [0.724, 0.736] |
| `sup_hgb` | 0.9838 [0.982, 0.986] | 0.6001 [0.596, 0.609] | 0.7527 [0.750, 0.760] | 0.6001 [0.596, 0.609] |
| `sup_rf` | 0.9835 [0.981, 0.985] | 0.5483 [0.542, 0.557] | **0.3853 [0.378, 0.390]** | 0.5483 [0.542, 0.557] |
| `sup_et` | 0.9839 [0.982, 0.986] | 0.6812 [0.674, 0.687] | 0.7150 [0.711, 0.722] | 0.6812 [0.674, 0.687] |
| `sup_lr` | 0.8599 [0.851, 0.869] | 0.6650 [0.661, 0.672] | 0.8673 [0.864, 0.870] | 0.6650 [0.661, 0.672] |
| `if` | 0.9140 [0.908, 0.920] | 0.6709 [0.665, 0.678] | 0.9345 [0.933, 0.937] | 0.6709 [0.665, 0.678] |
| `ocsvm` | 0.7500 [0.735, 0.765] | 0.6714 [0.665, 0.678] | 0.9040 [0.901, 0.907] | 0.6714 [0.665, 0.678] |
| `ae` | 0.8995 [0.892, 0.907] | **0.5000 [0.500, 0.500]** | **0.5000 [0.500, 0.500]** | 0.7571 [0.752, 0.764] |
| `lstm_ae` | 0.8001 [0.790, 0.811] | 0.6786 [0.674, 0.688] | 0.8978 [0.894, 0.901] | 0.7122 [0.708, 0.721] |

The intervals are narrow (SR-BH test slices are ~100 k rows), so every T1 → T2 drop is
far outside sampling error. Note the T2 and T4 columns are **identical** wherever the
stored calibration was well-behaved — the visible proof that a monotone recalibration
cannot move ROC-AUC. `ae`'s CI of exactly [0.500, 0.500] is complete saturation: not one
resample produced any ranking signal.

`sup_rf` under the temporal split reaches 0.3853 [0.378, 0.390] — **significantly worse
than random**, i.e. its score ordering is actively inverted on that window.

### What recalibration recovers

Recovery fraction, per metric *m*, at each budget:

> **R(m) = (m_T4 − m_T2) / (m_T1 − m_T2)**, computed only when |m_T1 − m_T2| > 1e-9.

R = 1 means recalibration restores the in-domain level; R = 0 means it bought nothing.

**Threshold-free metrics (the meaningful ones):**

| Scorer | ROC-AUC R | log-loss R | Brier R | ECE R |
|---|---|---|---|---|
| `ml_ensemble` | **−0.031** | 0.949 | 0.994 | 1.265 |
| `sup_lr` | **0.000** | 1.017 | 1.019 | 1.283 |
| `sup_hgb` | **0.000** | 0.030 | 0.028 | −5.593 |
| `sup_rf` | **0.000** | 0.184 | 0.196 | 1.627 |
| `sup_et` | **0.000** | 0.933 | 0.946 | 1.311 |
| `if` | **0.000** | 0.956 | 0.856 | 5.231 |
| `ocsvm` | **−0.000** | 1.304 | 2.490 | 5.946 |
| `ae` | **0.643** | 1.010 | 0.950 | 1.856 |
| `lstm_ae` | **0.276** | 1.077 | −1.634 | −38.051 |
| `ml_aux_novelty` | **0.153** | 2.513 | −3.156 | −4.345 |

ROC-AUC recovery is **exactly zero** for every model whose CSIC calibration was
well-behaved. That is the arithmetic consequence of monotonicity, and it is the point:
recalibration is not a transfer technique.

Absolute calibration quality on the SR-BH random test, primary ensemble:

| | log-loss | Brier | ECE-15 |
|---|---|---|---|
| T1 (in-domain) | 0.354 | 0.120 | 0.147 |
| T2 (naive transfer) | 1.339 | 0.527 | 0.636 |
| **T4 (recalibrated)** | **0.405** | **0.122** | **0.018** |

Recalibration restores probability quality to *better than in-domain* on ECE, while
leaving ranking untouched. Reliability diagrams: `docs/figures/srbh_recalib/`.

**All 16 refits (8 models × 2 families) selected the Platt/attack-logistic branch** by NLL.
Neither ECDF nor μ/σ was ever chosen, on either split family. Worth noting for the paper's
calibration-branch discussion: the three-branch machinery collapses to one branch under
distribution shift, at least here.

### Why the recovery fraction is meaningless for F1/precision/recall here

The table above deliberately omits threshold metrics. Their recovery fractions come out as
9.4, 15.2, even 99.9 — not because recalibration works miracles, but because the
denominator is *negative*: **F1 is higher under naive transfer than in-domain**
(0.9194 vs 0.8400 for the primary ensemble). With 85.1 % attack prevalence in the SR-BH
test slice and a threshold that flags ~100 % of traffic, predicting "attack" for
everything scores F1 = 0.919. The naive-transfer configuration is not doing well; it has
degenerated to a constant classifier that the metric happens to reward.

R is only interpretable where the in-domain configuration is genuinely better. For this
corpus that means ROC-AUC and the proper scoring rules, and the report treats F1/recall at
matched α as descriptive only.

### Saturation: where the CSIC calibration destroyed the ranking

`ae` reaches **ROC-AUC exactly 0.5000** under T2 and T3 — literally uninformative. Its
stored μ/σ calibration maps every SR-BH reconstruction error to the same saturated
probability, so all ordering is lost in the calibration layer, not in the model.
Recalibration recovers ROC-AUC to 0.757 (random) and 0.944 (temporal) — the only cases
where R(ROC-AUC) is materially positive (0.643, and 0.276 for `lstm_ae`).

So the one exception to "recalibration cannot recover ranking" is when the *original*
calibration was itself lossy. That is a bug-class finding, not a transfer finding: a
calibration layer that saturates silently discards information the model did compute. It
is worth a sentence in Limitations and a guard in production.

`sup_rf` under T3 scores **ROC-AUC 0.385 — worse than random**, meaning its ordering is
actively inverted on the temporal test window. Recalibration cannot fix this either, for
the same monotonicity reason.

---

## Part D — `encoding_depth`

Definition: number of `unquote_plus` passes until fixpoint, capped at 4, taken as
max(depth(URI), depth(body)). Prototype only — **not** added to `FEATURE_COLS`, which
would break feature-order parity with every trained model.

**Standalone single-feature ROC-AUC for attack detection:**

| Corpus | rows | ROC-AUC |
|---|---|---|
| SR-BH 2020 | 907,815 | **0.8737** |
| CSIC 2010 | 61,065 | **0.7124** |

Depth distribution on SR-BH: benign is 80.6 % depth-0 and 0.26 % depth-≥2; attacks are
8.8 % depth-≥2 including 1,065 at depth 3. The signal is real and it is not what
`pct_encoded_ratio` measures.

**Incremental value over the existing 24 features** (HistGradientBoosting, seed 42,
trained on `srbh_supervised_train`, evaluated once on `srbh_test`):

| | ROC-AUC |
|---|---|
| 24 features | 0.980304 |
| 24 + `encoding_depth` | 0.981260 |
| **Δ** | **+0.000956, 95 % CI [+0.000669, +0.001239]** |

The CI excludes zero, so the gain is real — but it is **+0.001 ROC-AUC**, three orders of
magnitude smaller than the standalone AUC of 0.87 would suggest. The existing 24 features
already capture most of what nesting depth encodes, presumably through
`pct_encoded_ratio`, `suspicious_char_ratio` and the path entropy features.

**Recommendation: do not schedule a feature-set version bump and full retrain for
+0.001 ROC-AUC.** The cost (retraining ExtraTrees at 765 MB, re-calibrating every model,
invalidating `feature_order.sig` and every stored artifact) is not justified by the
measured gain. The feature is more valuable as a *reported diagnostic* — "8 % of real
attacks use nested encoding that single-pass normalisation does not resolve" is a finding
about the NLP channel's blind spot, which is where it should be cited.

---

## Environment note (affects reproducibility)

The stored artifacts were pickled with **scikit-learn 1.8.0** and a Keras newer than
3.12.0. The workstation had scikit-learn 1.7.2 and Keras 3.12.0, under which:

* every sklearn model raised `InconsistentVersionWarning` ("may lead to breaking code or
  invalid results");
* `ae` and `lstm_ae` **could not be deserialized at all** (`Unrecognized keyword argument
  'quantization_config'`).

Rather than score with mismatched libraries, all results here were produced in an isolated
venv with `scikit-learn==1.8.0` and `keras==3.15.1`, under which all eight artifacts load
without warnings.

One workaround was required and is flagged because it touches the numbers: under
keras 3.15 / tensorflow 2.20, `keras.Model.predict()` **deadlocks** (zero CPU progress,
indefinitely). `eval_srbh_transfer._keras_forward` calls the model directly in inference
mode with manual batching instead. This is the same forward pass and returns identical
values; it only avoids the `tf.data` loop that hangs.

For the paper's reproducibility section, the training environment must be pinned —
currently it is not recorded anywhere in the repo.

### ⚠️ `ml_http_features.py` is not in version control

`jobs/retrain/ml_http_features.py` defines `FEATURE_COLS` — the 24-feature contract the
whole numeric pipeline is built on — and has **never been committed**. It is untracked,
not gitignored. Five tracked files import it, including `build_ml_numeric_from_csic.py`,
which predates this work:

```
jobs/retrain/build_ml_numeric_from_csic.py
jobs/retrain/build_ml_numeric_attacklab.py
jobs/retrain/build_ml_numeric_from_srbh.py
tests/srbh_ingestion_smoke.py
(+ the TASK 02 scripts, also untracked by request)
```

A fresh clone therefore cannot run the numeric pipeline or reproduce any dataset build.
Left untracked at the author's direction; recorded here because it blocks the Data/Code
Availability statement the target journals require.

## Determinism

Verified twice. Two consecutive no-CI runs produce byte-identical
`srbh_transfer_results.json`. The full run with bootstrap CIs reproduces all **1,395**
point estimates from the no-CI run with **zero** differences, and emits all 200 CI sets
(5 configurations × 10 scorers × 4 budgets). Seed 42 throughout.

Runtime note: the bootstrap pass takes ~50 min of CPU. `_bootstrap_ci_for_threshold_metrics`
recomputes ROC-AUC and PR-AUC inside every resample even though neither depends on the
threshold, and it re-seeds to 42 with the same row count on every call — so the resample
indices are identical across the four budgets and those two metrics are computed four
times redundantly. Caching them per (configuration, scorer) would cut the run to roughly
half. Not changed here, because it would mean editing the shared CI path that the rest of
the pipeline also uses.
