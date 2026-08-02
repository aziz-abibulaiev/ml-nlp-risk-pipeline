# Content detector (per-request) — honest, cross-tool-validated

Built on the honest corpus (`feat/honest-corpus` `17c7b69`). Branch `feat/content-detector`.
Generation of a model + honest internal evaluation only — no external validation, no rule
baseline, no ablations (those are TASK 06). Provenance-only labels; fixed declared
hyperparameters (`services/content_detector/config.py`); nothing tuned; nothing touched the
test split except the final frozen evaluation.

## Architecture (D16)

- **Shared normalization** (`normalize.py`, identical in train and inference):
  `decode_fixed_point(s)` applies `unquote_plus` to a fixed point, capped at 3 passes — this
  fixes the measured double-encoding blind spot. Two text fields:
  A = `decode_fixed_point(url_path_raw + url_query + request_body)`,
  B = `decode_fixed_point(user_agent + referer)` (separate vectorizer, weighted separately).
- **`p_text`**: two char-TFIDF vectorizers (A, B) `analyzer=char, ngram=(3,5), min_df=2,
  lowercase=False, sublinear_tf`, concatenated, → one `LogisticRegression(class_weight=
  balanced)`. Chosen design: **one linear model over the concatenated two-field TF-IDF**, so
  field A and field B are separate feature blocks whose weights differ but contribution stays
  in one interpretable model.
- **`p_numeric`**: shape-only statistics (lengths, entropy, char-class ratios, punctuation
  density, percent-encoded ratio, `encoding_depth`) per field → `HistGradientBoosting`. No
  feature encodes an attack token — every one answers "how is the string shaped".
- **Fusion**: `p_content = Platt( 0.5·logit(p_text) + 0.5·logit(p_numeric) )`, the Platt
  layer fit **only on calib**. Declared rule (`mean_logit`); no meta-model search.
- **Operating point**: FPR-budget threshold from **calib benign** only.
- **Two levels**: binary core (generalizing claim) + family head (testbed-only).

## Fixed hyperparameters (declared, not tuned)

```
TFIDF   analyzer=char ngram=(3,5) min_df=2 lowercase=False sublinear_tf=True
        max_features A=200000 B=50000
LOGREG  class_weight=balanced C=1.0 solver=liblinear max_iter=2000
HGB     learning_rate=0.1 max_iter=300 max_leaf_nodes=31 l2=1.0
FUSION  mean_logit, weights (text 0.5, numeric 0.5), Platt on calib
FPR budgets 0.01 (primary), 0.05     SEED 42
```

## Split (session-level, family-stratified, tool-aware)

Structural fact: **every content-injection attack lives in 3 sessions** (one src_ip per
tool — ffuf, custom, sqlmap). A plain session hash is degenerate for content, so the split
assigns the **ffuf+sqlmap sessions to train and the custom session to test**. This makes the
main split a **cross-tool split for content by construction** — the Part A test metric is
already an unseen-tool number for content. Behavioural (10 sessions/family) is stratified
6/2/2; context families distributed one per split; benign hashed 60/20/20. Zero session
overlap asserted.

| split | rows | benign | attack | note |
|---|---|---|---|---|
| train | 13,025 | 7,026 | 5,999 | content = ffuf/sqlmap |
| calib | 3,075 | 2,261 | 814 | no content-injection (behavioural+context only) |
| test | 5,036 | 1,621 | 3,415 | content = custom (held-out tool) |

Consequence to note: calib carries no content-injection attacks, so the fusion Platt is fit
on benign + behavioural/context attacks. The FPR threshold is a benign-quantile so it stays
valid; ROC/PR are threshold-free. This is honest (calib never sees test) but it is why the
numeric channel's realized test FPR overshoots its budget (below).

## Part A — binary core + late fusion (test)

| budget | channel | ROC-AUC | PR-AUC | F1 | Prec | Recall | FPR (test) |
|---|---|---|---|---|---|---|---|
| 1% | `p_text` | 0.9876 | 0.9943 | 0.897 | 0.997 | 0.815 | 0.006 |
| 1% | `p_numeric` | 0.9646 | 0.9815 | 0.812 | 0.972 | 0.698 | 0.042 |
| 1% | **`p_content`** | 0.9822 | 0.9915 | **0.912** | 0.977 | **0.854** | 0.042 |
| 5% | `p_text` | 0.9876 | 0.9943 | 0.952 | 0.982 | 0.923 | 0.036 |
| 5% | `p_content` | 0.9822 | 0.9915 | 0.908 | 0.961 | 0.860 | 0.073 |

Reads: the **text channel alone is excellent** (ROC 0.988) and its threshold transfers
cleanly (realized FPR 0.006 ≈ budget). The **numeric channel is weaker (0.965)** and its
calib threshold transfers poorly (realized FPR 0.042 at a 1% budget) — a small-sample
threshold-estimation effect, inherited by `p_content`. Fusion **raises operating-point
recall** (0.854 vs 0.815 for text at the 1% budget) at the cost of a higher realized FPR;
ROC-AUC of the fusion sits between the two channels. The value of fusion is clearest in
Part C, not in a single headline number.

## Part B — cross-tool holdout ⭐ (the headline)

Train the binary core on one tool's payloads for a family, test on the **held-out tool's**
payloads. High held-out recall ⇒ the detector learned the attack, not the tool. Recall at
the 1% FPR budget (realized benign FPR in parentheses):

| family | ffuf → custom | custom → ffuf |
|---|---|---|
| `sqli` | **0.893** (0.016) | **0.863** (0.032) |
| `xss` | **0.920** (0.012) | **1.000** (0.039) |
| `cmd_injection` | **0.167** (0.013) | 1.000 (0.008) |
| `ssti` | 0.562 (0.017) | 0.682 (0.021) |
| `path_traversal`† | 0.713 (0.017) | 0.995 (0.044) |
| `scan`† | 0.491 (0.017) | 1.000 (0.009) |

† 4 shared payloads dropped for `path_traversal`, 9 for `scan`, before the holdout.
`nosql_injection` and `ssrf` are **excluded** from the per-family claim (their ffuf slice is
off-family content — SQLi/LFI relabeled by intent); they count only toward the binary core.

**Honest reading — this is the most important result and it is mixed.** `sqli` and `xss`
generalize across tools well in both directions: the detector learned the attack. But
`cmd_injection` collapses one direction (train ffuf → test custom = **0.167**) while the
reverse is perfect — the model trained on SecLists commix payloads does not recognize the
curated custom command-injection strings, whereas a model trained on the curated set
recognizes the SecLists ones. `ssti` and `scan` are similarly asymmetric. This is exactly
the tool-fingerprint risk D15 warned about, made visible: for some families a char-TFIDF
detector partly learns the payload *style* of its training tool. Reporting it, not hiding
it — it directly motivates multi-tool training and is a genuine Threats-to-Validity result.

## Part C — channel complementarity (test attacks, 1% budget)

Attacks partitioned by the audit regex (measurement instrument only, never a feature):

| partition | `p_text` | `p_numeric` | `p_content` |
|---|---|---|---|
| marked (n=823) | 0.903 | 0.836 | **0.956** |
| marker-free (n=2592) | 0.787 | 0.654 | **0.822** |

Fusion beats either channel on **both** partitions, so late fusion adds value. But the
hypothesized division of labour is only partial: text is stronger than numeric even on the
marker-free partition (0.787 vs 0.654). The reason is honest — the audit regex is narrow, so
many "marker-free" attacks still carry sub-token lexical structure the char-TFIDF catches;
the numeric channel's contribution is real but secondary, mainly lifting recall where text
is uncertain. Reported as measured.

## Part D — family head (TESTBED-ONLY, not real-traffic-validated)

Text char-TFIDF → multinomial LogReg, trained on train-split attacks, tested on test-split
attacks. **Because the split is cross-tool (train ffuf, test custom), this is a strict
cross-tool family-ID test**, and family identification is more tool-sensitive than binary
detection — so families that already showed a cross-tool gap in Part B (cmd_injection, ssti,
scan) degrade here too. Per-family P/R on test:

| family | support | P | R | | family | support | P | R |
|---|---|---|---|---|---|---|---|---|
| xss | 150 | 1.00 | 0.89 | | scan | 500 | 0.60 | 0.95 |
| sqli | 150 | 0.43 | 0.43 | | ssti | 500 | 0.65 | 0.35 |
| cmd_injection | 150 | 0.00 | 0.00 | | nosql_injection | 500 | 0.00 | 0.00 |
| path_traversal | 150 | 0.00 | 0.00 | | ssrf | 500 | 0.12 | 0.05 |
| brute_force | 207 | 0.43 | 1.00 | | cred_stuffing | 208 | 0.38 | 1.00 |
| resource_exhaustion | 198 | 0.77 | 1.00 | | | | | |

The behavioural families (brute/cred/resource) get high recall but low precision — they are
marker-free and mutually similar, so the content family head confuses them (as expected;
they belong to the behavioural detector, not this one). Family ID is a best-effort testbed
signal, not a validated capability.

## Deliverable

`services/content_detector/predict.py` — `predict(method, path, query, body, headers)`
returns `{is_attack, p_attack, family, family_confidence, explanation, channel:{p_text,
p_numeric}}`. `explanation` is a post-hoc named-regex match on the decoded field, decoupled
from the score (display only — never a feature or label). Verified on SQLi/XSS/benign and a
doubly-encoded traversal (the fixed-point decode fires the explanation on the decoded text).

## Part E — external FPR/recall read (a smoke, not the full validation)

Frozen binary core, threshold from calib (0.4757 at the 1% budget), run on external
held-out sets. **Nothing was tuned to these numbers.**

| set | n | metric | value |
|---|---|---|---|
| zanbil sample (organic benign) | 200,000 | alert rate | **82.97%** |
| SR-BH benign | 525,195 | alert rate | **60.14%** |
| SR-BH attack | 382,620 | recall | 96.10% |

### 🚨 This is catastrophic, and it is the point

An **82.97%** alert rate on real organic traffic (and 60% on SR-BH benign) is exactly the
"0.99 in the lab / unusable on real traffic" failure this whole project exists to expose —
now measured, not assumed. The binary core catches 96% of real SR-BH attacks, but at a
false-alarm rate that makes it undeployable as-is.

The cause is the same covariate shift seen in TASK 02 (CSIC→SR-BH): the detector's benign
class is **scripted testbed benign** (VAmPI/crAPI), which is narrow and clean. The
char-TFIDF text channel effectively learned "benign looks like our testbed", so real benign
traffic — Persian e-commerce URLs (zanbil), honeypot benign (SR-BH) — falls outside that
manifold and scores as attack. The calib threshold (already loose at 4.2% realized FPR on
testbed) is nowhere near a real-traffic operating point.

Per the task this is reported prominently and **left unadjusted**. It is the central
Threats-to-Validity finding and it defines the TASK 06 agenda: a rule baseline for
comparison, real-benign augmentation of the training/calibration benign (D13 — measure FPR
on real benign, not testbed benign), and ablations. The honest-corpus attack side is sound
(96% recall on real attacks); the benign side must include real traffic before any FPR claim
is credible.

