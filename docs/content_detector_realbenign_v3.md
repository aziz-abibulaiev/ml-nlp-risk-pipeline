# TASK 06 Part A v3 — in-domain deployment scenario (the product number)

Branch `feat/real-benign-validation`, off `fef176a`. The one experiment that answers the
product question: **does the detector work when trained on the deployment domain's own
benign?** — which is exactly how real WAF-ML is deployed (trained on the customer's own
traffic). Text-only (the numeric channel was confirmed source-correlated/harmful cross-source
in v2); fixed hyperparameters; threshold never on the test slice; real benign never
regex-filtered (regex only as the manual-inspection aid). Seed 42.

## Verdict: PASSES. In-domain training delivers a usable detector on modern real traffic.

At the 1 % FPR budget: **0.42 % alert-rate on held-out later zanbil traffic (≈0.29 % true FP
after the WAMM correction), 100 % recall on held-out SR-BH real attacks, 99.9 % recall on an
entirely unseen testbed tool.** Headline contrast: **0.42 % in-domain vs 99.5 %** when zanbil
was held out cross-source in v2 — a ~237× false-positive reduction from training on the
deployment domain.

## In-domain temporal split of zanbil

Deduplicated zanbil (893,936 unique of 10.36 M), then split **by time** (first-seen
timestamp): earlier requests → train/calib, later requests → the FP test — "train on your
site's history, detect on new traffic".

| slice | rows | window |
|---|---|---|
| train benign (early) | 150,000 | < 2019-01-25 04:20:40 |
| calib benign (early, disjoint) | 30,000 | < boundary |
| **test benign (late, held out)** | 150,000 | ≥ boundary |

Disjointness asserted: train/calib, train/test, calib/test benign share no rows (dedup is
global on the request signature, so each unique request lands in exactly one slice; late-test
requests are ones that *first appear* after the boundary). Deterministic, seed 42.

**Attacks (cross-source real):** train = a 70 % hash split of SR-BH real attacks (60 k) +
testbed tools (all except `custom`); held-out test = the 30 % SR-BH split (40 k, disjoint
rows) + the entirely unseen **testbed `custom` tool** (3,094). SR-BH train/test attack rows
asserted disjoint.

## Metrics (text-only)

| budget | late-zanbil alert | WAMM-corrected FP | SR-BH recall | testbed:custom recall |
|---|---|---|---|---|
| **1 %** | **0.42 %** | **≈0.29 %** | **100.0 %** | **99.9 %** |
| 5 % | 2.64 % | ≈1.82 % | 100.0 % | 100.0 % |

Recall is 100 % on held-out SR-BH real attacks and 99.9 % on the unseen `custom` tool — so
both cross-source attack transfer (SR-BH, same source different rows: the realistic "train on
captured attacks, catch more of the same") and cross-tool transfer (custom never in training)
hold at this operating point.

## WAMM manual check (top-100 late-zanbil alerts; regex as aid only)

**31 regex-marked (plausible real attacks), 69 true false positives** → corrected FP ≈
0.42 % × 0.69 ≈ **0.29 %**. The marked ones are genuine attacks hiding in zanbil's nominal
"benign" — the detector surfaced them, the dataset mislabeled them:
- `/wp-content/plugins/google-mp3-audio-player/direct_download.php?file=../../../wp-config.php`
  (real LFI against a known WordPress plugin vuln)
- `/wp-content/plugins/recent-backups/download-file.php?file_link=../../../wp-config.php`
- `/image/29000?name=...' AND 1152=CONVERT(INT,(SELECT CHAR(113)+...` (textbook sqlmap SQLi)

The true FPs are ordinary traffic: `/index.action` (a Struts-looking but benign path, the
single highest-scored alert at 0.97) and Persian e-commerce browse/product pages
(`/browse/cooktop/…`, `/product/28108/…`) scored 0.50–0.65. So the alert-rate is an upper
bound on FP, and even the true-FP tail is a handful of suspicious-looking-but-benign paths —
tractable for analyst review at 0.42 % of traffic.

## Honest read

**In-domain training works, and this is the deployable product number.** Trained on the
deployment domain's own benign, the text-only detector flags 0.42 % of later real traffic
(≈0.29 % genuine FP) while catching 100 % of held-out real SR-BH attacks and 99.9 % of an
unseen attack tool. That is a usable WAF-ML operating point on modern real traffic.

This completes the honest arc, and both halves are results:
- TASK 05 (testbed-only benign): 83 % real-benign FP — fails.
- v1 A1/A2 (one real source): memorisation — fails.
- v2 LOSO (many sources): conditional transferability; numeric hurts → text-only; works only
  if a *similar* source is in training.
- **v3 (in-domain): 0.42 % FP at 100 % recall — works.**

The paper's claim is therefore precise and defensible: *the detector is deployable when
trained on the target domain's own benign (0.42 % FP, 100 % real-attack recall); cross-domain
it degrades, and that domain-coverage requirement is the central Threats-to-Validity finding.*
No universal-low-FP claim; a demonstrated in-domain result plus an honest boundary.

## Gate decision

**STOP for review.** The in-domain deployment condition passes. Recommendation for Parts B–E,
if approved: deploy this **text-only in-domain** model; add the rule baseline (Part B) against
it, cross-tool table (Part C, already have TASK 05 + v2), ablations text-vs-fused (Part D — v2
already shows numeric hurts cross-source; complement with the in-domain figure), and the
LLM/human family relabelling (Part E). Frame around the in-domain product number + the
domain-coverage Threats-to-Validity.

Config / determinism: fixed declared hyperparameters (`config.py`), seed 42, deterministic
sampling. Artifacts under `models/content_v4/` (joblib gitignored, regenerable via
`python -m services.content_detector.train_partA_v3`).
