# SR-BH 2020 vs CSIC 2010: feature distribution report

Benign rows only. Generated from `data/srbh2020/srbh_features.parquet` (525,195 benign
rows, `label == 0`) and `data/ml_test.csv` (8,000 benign rows, `source == csic`).
Seed 42 throughout. Feature definitions from `jobs/retrain/ml_http_features.py`; no
feature was recomputed or reordered.

## Headline result

**A HistGradientBoostingClassifier separates SR-BH benign from CSIC benign with
cross-validated ROC-AUC = 1.0000 ± 0.0000** (5-fold stratified, 8,000 rows per class,
SR-BH subsampled to match CSIC, seed 42).

This is expected and is reported rather than hidden. It is the same magnitude previously
measured between the `csic` and `mahendra` benign sources inside `ml_train.csv`. Two
benign corpora captured a decade apart, on different applications, through different
capture schemas, are trivially distinguishable in this feature space.

**The separation is not an artifact of the header mapping.** Removing all four
header-derived features (`header_count`, `header_value_total_len`, `header_value_max_len`,
`headers_entropy`) still gives **CV ROC-AUC = 1.0000 ± 0.0000** on the remaining 20
features. The domain gap is a property of the traffic, not of the ingestion choices
documented in `srbh2020_mapping.md`.

Ten of the 24 features exceed 0.97 ROC-AUC **on their own**:

| Feature | single-feature CV ROC-AUC |
|---|---|
| `header_count` | 1.0000 |
| `header_value_max_len` | 0.9999 |
| `header_value_total_len` | 0.9991 |
| `path_digit_ratio` | 0.9968 |
| `suspicious_char_ratio` | 0.9911 |
| `path_entropy` | 0.9885 |
| `endpoint_len` | 0.9854 |
| `payload_size` | 0.9778 |
| `path_depth` | 0.9714 |
| `headers_entropy` | 0.9673 |

Permutation importance on a 30 % holdout attributes essentially all of it to
`header_count` (0.380; every other feature 0.000) — not because the others carry no
signal, but because `header_count` alone already achieves perfect separation, so permuting
anything else cannot degrade an already-saturated score. The single-feature column above is
the honest view of redundancy.

The mechanism for `header_count` is direct and worth stating plainly: CSIC benign takes
only the values 10 and 11, SR-BH benign spans 0–10 with mode 5 and only **9 rows** at 10.
The two ranges barely touch. This is the capture schema showing through — CSIC replays a
fixed browser header template, SR-BH records 11 named header slots of which real clients
populate a handful.

**Implication for the paper.** A model trained on CSIC and evaluated on SR-BH is a genuine
out-of-domain transfer, not a mild covariate shift, and its metrics must be reported as
such. Prior correction (Eq. 24) addresses the base-rate difference (42.1 % attacks in SR-BH
versus 20 % in `ml_test.csv`); it does not address this covariate shift. Do not present
cross-dataset numbers without this figure alongside them.

## Per-feature comparison

`mode share` = frequency of the most common value. `SMD` = standardised mean difference,
(mean_SRBH − mean_CSIC) / sqrt((sd_SRBH² + sd_CSIC²)/2); positive means larger in SR-BH.
`near-const` flags mode share ≥ 0.95.

| Feature | SR-BH n_uniq | SR-BH mode share | SR-BH mean | SR-BH std | CSIC n_uniq | CSIC mode share | CSIC mean | CSIC std | SMD | near-const in |
|---|---|---|---|---|---|---|---|---|---|---|
| `endpoint_len` | 149 | 0.569 | 26.7519 | 34.8257 | 12 | 0.271 | 28.4361 | 4.1737 | -0.07 | — |
| `header_count` | 11 | 0.704 | 4.5304 | 0.8587 | 2 | 0.772 | 10.2284 | 0.4198 | **-8.43** | — |
| `payload_size` | 868 | 0.039 | 257.0007 | 173.2547 | 174 | 0.103 | 406.4751 | 90.0580 | -1.08 | — |
| `headers_entropy` | 43580 | 0.021 | 5.1827 | 0.1585 | 4233 | 0.002 | 5.3940 | 0.0181 | -1.87 | — |
| `query_param_count` | 7 | 0.917 | 0.0903 | 0.3203 | 5 | 0.771 | 1.2256 | 3.1961 | -0.50 | — |
| `path_depth` | 13 | 0.569 | 2.1454 | 2.6693 | 3 | 0.890 | 3.0546 | 0.3277 | -0.48 | — |
| `suspicious_char_ratio` | 3977 | 0.037 | 0.0281 | 0.0124 | 174 | 0.103 | 0.0316 | 0.0027 | -0.39 | — |
| `is_json_payload` | 1 | 1.000 | 0.0000 | 0.0000 | 1 | 1.000 | 0.0000 | 0.0000 | +0.00 | **SR-BH, CSIC** |
| `http_method_code` | 5 | 0.987 | 1.0421 | 0.4543 | 2 | 0.772 | 1.2284 | 0.4198 | -0.43 | SR-BH |
| `query_len` | 152 | 0.917 | 1.5782 | 9.1314 | 96 | 0.771 | 21.3663 | 60.3435 | -0.46 | — |
| `query_entropy` | 775 | 0.919 | 0.2559 | 0.8770 | 773 | 0.771 | 0.9221 | 1.7476 | -0.48 | — |
| `query_value_len_mean` | 177 | 0.920 | 0.8265 | 4.9256 | 88 | 0.771 | 2.3576 | 5.2806 | -0.30 | — |
| `query_value_len_max` | 78 | 0.920 | 0.9712 | 6.1809 | 48 | 0.771 | 3.8475 | 8.4391 | -0.39 | — |
| `empty_query_value_ratio` | 5 | 0.998 | 0.0022 | 0.0471 | 1 | 1.000 | 0.0000 | 0.0000 | +0.07 | **SR-BH, CSIC** |
| `path_entropy` | 39670 | 0.569 | 1.8349 | 2.1278 | 24 | 0.087 | 3.9415 | 0.1521 | -1.40 | — |
| `path_digit_ratio` | 2615 | 0.705 | 0.0559 | 0.1092 | 12 | 0.271 | 0.0394 | 0.0149 | +0.21 | — |
| `path_upper_ratio` | 753 | 0.820 | 0.0079 | 0.0212 | 1 | 1.000 | 0.0000 | 0.0000 | +0.53 | CSIC |
| `body_len` | 192 | 0.994 | 0.7104 | 10.4702 | 96 | 0.772 | 21.3385 | 60.3468 | -0.48 | SR-BH |
| `body_entropy` | 2377 | 0.994 | 0.0272 | 0.3540 | 774 | 0.772 | 0.9187 | 1.7459 | -0.71 | SR-BH |
| `body_digit_ratio` | 497 | 0.994 | 0.0004 | 0.0059 | 384 | 0.799 | 0.0260 | 0.0592 | -0.61 | SR-BH |
| `body_punct_ratio` | 366 | 0.994 | 0.0010 | 0.0126 | 259 | 0.772 | 0.0346 | 0.0680 | -0.69 | SR-BH |
| `pct_encoded_ratio` | 1369 | 0.806 | 0.0113 | 0.0291 | 265 | 0.671 | 0.0097 | 0.0154 | +0.07 | — |
| `header_value_total_len` | 690 | 0.046 | 219.2295 | 146.8332 | 2 | 0.772 | 319.2769 | 18.8903 | -0.96 | — |
| `header_value_max_len` | 186 | 0.061 | 162.7839 | 119.4224 | 1 | 1.000 | 99.0000 | 0.0000 | +0.76 | CSIC |

## Near-constant features

### The four features known constant in CSIC benign

The audit recorded `is_json_payload`, `empty_query_value_ratio`, `path_upper_ratio` and
`header_value_max_len` as constant in CSIC benign. Whether they vary in SR-BH:

| Feature | Constant in CSIC benign | Varies in SR-BH benign? | Evidence |
|---|---|---|---|
| `is_json_payload` | yes (0) | **No — still constant 0** | Only **3 rows in the entire 907,815-row corpus** have a JSON content type, and all 3 are attacks. A 2020 WordPress site serves form-encoded requests. |
| `empty_query_value_ratio` | yes (0) | **Marginally** — 5 distinct values, but 1,178 / 525,195 benign rows (0.22 %) are non-zero; mode share 0.998 | Still near-constant; no useful variance gained |
| `path_upper_ratio` | yes (0) | **Yes** — 753 distinct values, 94,725 / 525,195 (18.0 %) non-zero, mode share 0.820 | Real traffic has mixed-case paths; CSIC does not |
| `header_value_max_len` | yes (99) | **Yes, strongly** — 186 distinct values, mode share 0.061, range 0–624, median 119 | The dominant variance source is user-agent / cookie length |

So **two of the four recover real variance on real traffic and two do not.**

This refines the planned hygiene fix (memory §3.-1, item 3). Dropping all four as
uninformative is correct for a CSIC-only pipeline. If SR-BH becomes a training corpus
rather than only an external test set, `path_upper_ratio` and `header_value_max_len` must
be reinstated — they are constant because of CSIC, not because of the feature definition.
`is_json_payload` and `empty_query_value_ratio` are dead in both corpora and can be dropped
unconditionally.

### Additional near-constant features in SR-BH benign (mode share ≥ 0.95)

`http_method_code` (0.987 — benign traffic is overwhelmingly GET), and the four body
features `body_len`, `body_entropy`, `body_digit_ratio`, `body_punct_ratio` (all 0.994),
which follow directly from `request_body` being non-empty in only 2.9 % of rows.

**Consequence for the NLP channel:** on SR-BH the body carries almost no signal. The NLP
channel will operate on the URI path and the header blob, not on payload bodies. Any claim
about body-based detection cannot be validated on this corpus.

## Reproduction

The generating script is not committed (this task's scope was the ingestion script and the
two documents). It reads the two corpora, computes the table above, and runs
`HistGradientBoostingClassifier(random_state=42)` under `StratifiedKFold(5, shuffle=True,
random_state=42)` with `scoring="roc_auc"`, plus `permutation_importance(n_repeats=10,
random_state=42)` on a 30 % stratified holdout. Say the word and it can be added under
`jobs/retrain/` so the numbers are regenerable from the repository.
