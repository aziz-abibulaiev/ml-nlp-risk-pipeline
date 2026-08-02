# TASK 06 Part A v2 — multi-source + leave-one-source-out (revised gate)

Branch `feat/real-benign-validation`, off `d1b1a92`. Supersedes the first Part A (A1/A2),
which failed because the positive class was testbed-attacks-only and a single benign source
plus the SR-BH benign/attack collision made "attack" ≈ "unlike my benign". Fix: train on many
diverse real sources — multiple benign AND multiple real-attack — each both-class source
contributing both classes, then hold out ENTIRE sources for FP and recall. No autotune; fixed
declared hyperparameters; real benign enters as-is (never regex-filtered); nothing calibrates
or thresholds on a held-out source. Seed 42.

## Source registry (after within-source dedup)

| source | nature | body | benign | attack | label origin |
|---|---|---|---|---|---|
| zanbil | real | no | 893,936 | — | real e-commerce nginx |
| srbh | real | yes | 90,917 | 287,710 | honeypot binary (WAMM: ~10% benign are hidden attacks) |
| testbed | testbed | yes | 7,460 | 5,948 | tool provenance (ffuf/sqlmap/custom/behavioural) |
| csic | semi-synthetic | yes | 9,644 | 15,954 | CSIC 2010 classification |
| nasa | real | no | 8,436 | — | NASA-HTTP 1995 access log |
| clarknet | real | no | 18,411 | — | ClarkNet 1995 access log |
| calgary | real | no | 8,152 | — | Calgary 1994 access log |

**7 benign sources (5 real production logs + honeypot benign + testbed + semi-synthetic),
3 attack sources.** Fetched offline: NASA/ClarkNet/Calgary from the Internet Traffic Archive.
**Not fetched:** ModSec-2025 (Zenodo, ~12 GB — impractical to sample offline in time) and
AIT-LDS-v2 (multi-GB) — skipped and recorded; the bar (≥3 benign, ≥2 attack) is met without
them.

## Leave-one-source-out folds (threshold on a real benign slice, never on the held-out source)

Held-out-benign alert-rate (FP) and held-out-attack recall, at the 1% budget:

| fold | B* (FP src) | A* (recall src) | text-only FP | text-only recall | fused FP | fused recall |
|---|---|---|---|---|---|---|
| F1 | zanbil | srbh | 99.5% | 99.8% | 100% | 99.9% |
| F2 | nasa | testbed | **0.01%** | 2.4% | 0.00% | 10.7% |
| F3 | srbh | csic | 98.1% | 3.4% | 95.1% | 6.5% |
| **F4** | **clarknet** | **srbh** | **0.07%** | **94.8%** | 4.19% | 98.4% |
| F5 | calgary | testbed | **0.00%** | 2.7% | 0.00% | 13.8% |
| F6 | zanbil | testbed:custom | 19.6% | 19.9% | 69.3% | 24.1% |

## Two decisive findings

### 1. Text-only beats fused cross-source — the numeric channel encodes the source

The numeric shape channel **never lowers** held-out FP and clearly raises it where the model
has discrimination room: F4 clarknet 0.07% → **4.19%**, F6 zanbil 19.6% → **69.3%** (@1%).
It buys recall (F2 2.4→10.7%, F4 94.8→98.4%) but at a far larger FP cost on unseen sources.
This confirms the D16 suspicion: **shape statistics (lengths, entropy, char-class ratios)
are source-correlated — they partly identify the capture, not the attack.** The deployed
cross-source detector should be **text-only** (`p_text`).

### 2. Transferability is conditional on manifold coverage — not universal

A held-out benign source transfers **iff a similar source was in training**:

- **F4 (hold out ClarkNet, zanbil still in train): text-only 0.07% FP + 94.8% SR-BH recall @1%.**
  This is a genuinely working, transferable content detector — held-out benign source AND
  held-out real-attack source, both unseen, both excellent. It exists.
- **F2/F5 (hold out NASA/Calgary): ~0% FP** — simple 1990s access logs sit inside the benign
  manifold covered by the other logs. (Recall is low there only because A*=testbed tool
  attacks have a tool-specific style the csic+srbh-trained model doesn't cover — the same
  cross-tool gap as TASK 05.)
- **F1/F3/F6 (hold out zanbil or SR-BH): 95–100% FP.** With no *similar* source left in
  training, a complex real source (Persian e-commerce; honeypot) is out-of-distribution even
  with five other benign sources present. Adding more *dissimilar* sources does not help.

So the honest boundary is not "more sources always fix FP" but "**you need a training benign
source from the deployment domain**". The benign manifold of the real web is not covered by
any finite set of dissimilar sources.

### WAMM manual check (top-100 held-out-benign alerts; regex as inspection aid only)

| fold | B* | regex-marked (plausible real attack) | true FP |
|---|---|---|---|
| F4 | clarknet | 1 | 99 |
| F1 | zanbil | 49 | 51 |
| F3 | srbh | 92 | 8 |
| F6 | zanbil | 88 | 12 |

For the noisy held-out sources the "FP" is heavily inflated by real hidden attacks: SR-BH
benign's top alerts are 92% real attacks (WAMM confirmed — `<script>` payloads in "benign"),
and wild zanbil's are ~50–88% regex-marked (real bot/scan probes and/or regex over-match).
The clean access log (ClarkNet) is 99% true FP of a tiny alert set. So held-out-benign
alert-rate is an **upper bound** on FP, materially so for honeypot/wild sources.

## Honest read

Multi-source training is a large, real improvement over A1/A2 and **it works in principle**:
F4 demonstrates a text-only detector that transfers to an unseen benign source (0.07% FP) and
an unseen real-attack source (94.8% recall) simultaneously. Two conditions make it work:
(i) drop the numeric channel — it transfers source identity; (ii) have a training benign
source from the deployment domain. Where (ii) fails (hold out zanbil/SR-BH with nothing
similar left), FP stays at the ceiling — that is the honest transferability limit, and it is
a manifold-coverage limit, not a "not enough sources" limit.

This is not a clean pass or a clean fail: it is **conditional transferability**, which is
itself the scientific result. The practical recipe: text-only core, trained on benign that
includes the target domain, evaluated leave-one-source-out; report the domain-coverage
dependence as the central Threats-to-Validity finding.

## Gate decision

**STOP for review.** Recommendation for Parts B–E, if approved: deploy the **text-only** core;
frame the contribution as *conditional cross-source transferability with a domain-coverage
requirement* (F4 as the positive existence proof, zanbil/SR-BH-as-B* as the boundary), not as
a universal low-FP claim. Parts B–E (rule baseline, cross-tool, ablations+figure, LLM/human
family relabelling) are **not** run pending this decision.

Source-level disjointness asserted per fold (B*/A* excluded from train/calib; the few
trivially-shared signatures like `GET /` further excluded from the eval sets — counts in
`models/content_v3/partA_v2_results.json`). Deterministic, seed 42. Models/artifacts under
`models/content_v3/` (regenerable via `python -m services.content_detector.train_partA_v2`;
requires the ITA logs in the scratchpad or `CORPUS_ITA_DIR`).
