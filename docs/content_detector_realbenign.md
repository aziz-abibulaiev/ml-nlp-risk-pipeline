# TASK 06 Part A — real-benign training: the GATE result

Branch `feat/real-benign-validation` off `feat/content-detector` `d1b1a92`. This is the hard
gate: fix the B05 failure (TASK 05: 96% real-attack recall but 83% real-benign alert rate) by
adding real benign to training, and measure FPR on a **different, fully held-out real source**.
No autotune; fixed declared hyperparameters; regex never a feature or a training-data filter
(only an inspection aid in the manual review). Real benign goes in as-is, noise declared
(WAMM: ~9–10% of SR-BH "benign" are real attacks).

## Verdict: the naive fix FAILS the gate — but the failure is diagnostic and actionable.

At the **specified FPR-budget threshold** (set on the calibration slice), the held-out
real-benign alert rate is catastrophic in both symmetric configs:

| config | train benign | FPR test (held-out source) | alert-rate @1% | @5% | SR-BH real-attack recall @1% |
|---|---|---|---|---|---|
| **A1** | testbed + SR-BH-benign | zanbil | **90.3%** | 99.9% | **5.8%** |
| **A2** | testbed + zanbil | SR-BH benign | **99.7%** | 100% | 99.5% |

Neither config yields both an acceptable real-benign alert rate and real-attack recall. But
the two configs fail for **different, informative reasons**, and the mechanism is the result.

## A1 — adding SR-BH benign helps zanbil FPR, but destroys SR-BH attack recall

`p_content` mean scores under the A1 model: testbed benign 0.004, testbed attack 0.952,
**SR-BH benign 0.000, SR-BH attack 0.000**, zanbil 0.037.

Two things happened:

1. **Benign realism partially transferred.** In *absolute* terms the A1 model scores zanbil
   low, so the real-benign FPR on a third source collapses versus TASK 05:

   | abs. threshold | A1 zanbil alert | TASK 05 zanbil alert |
   |---|---|---|
   | 0.05 | **10.8%** | 99.8% |
   | 0.2 | **5.2%** | 91.1% |
   | 0.5 | **2.0%** | 76.8% |

   Adding one real benign source cut zanbil false-positives from ~80–99% to ~2–11%. That part
   of the hypothesis is real.

2. **But SR-BH attack recall went to 0%** (0.0% at every absolute threshold). SR-BH benign
   (in training) and SR-BH attacks (in test) are the **same honeypot source** — the attack is
   in the payload, but the request shape, paths, and headers are the same distribution. The
   model learned "SR-BH-like traffic = benign", so SR-BH attacks score ~0. ROC of
   zanbil-benign vs SR-BH-attack is **0.0195** — *inverted*: SR-BH attacks rank below zanbil
   benign.

3. **The FPR-budget threshold degenerates.** Because in-source calib benign (SR-BH) scores
   collapse to 0.000, the 1%-FPR threshold is ≈0, so any nonzero zanbil score is flagged →
   the 90.3% budget-threshold alert rate. The budget cannot be honored on a different source.

**Manual inspection (top-100 zanbil alerts, regex as aid): 0 regex-marked, 100 true FPs.**
Examples: `/admin`, `//xmlrpc.php`, Persian e-commerce filter URLs — ordinary browsing
flagged for being out-of-distribution, not attacks.

So A1 "fixes" zanbil FPR by memorising SR-BH-as-benign, which is useless: it cannot detect
SR-BH attacks, and you must never put the real-attack evaluation source's own benign into
training.

## A2 — SR-BH is wholly out-of-distribution to a zanbil-trained model

`p_content` flags **99.7%** of held-out SR-BH benign and 99.5% of SR-BH attacks alike; ROC on
SR-BH-benign vs SR-BH-attack is **0.508** (random). A zanbil-trained benign manifold does not
contain SR-BH, so the entire source scores as attack.

**Manual inspection (top-100 SR-BH-benign alerts): 93 regex-marked (plausible real attacks),
7 true FP.** The very highest-confidence "false positives" are the WAMM hidden attacks —
literal `<script>alert(1)</script>` payloads sitting in SR-BH "benign". Example rows:
`/blog/index.php/2020/'"<script>alert(1);</script>/page`. This validates the WAMM caveat: the
nominal alert-rate is an upper bound, and its top is real attacks. But the *mass* (99.7%) is
genuine over-flagging of OOD benign, not hidden attacks.

## Read: memorisation, not benign-realism generalisation

The detector's positive class is still **testbed attacks only**, so "attack" is defined
relative to whichever benign sources are in training. Consequences:
- A new benign source not in training is out-of-distribution → flagged (A2, and A1 at the
  degenerate budget threshold).
- If the real-attack test source's benign IS in training, recall on that source dies (A1).

Adding **one** real benign source is not enough to cover the benign manifold, and the A1
design (SR-BH benign in train, SR-BH attacks in test) is self-defeating. The absolute-score
improvement on zanbil shows the direction is right, but the fix as specified does not produce
a deployable model.

## What this implies for the path forward (for review, not executed)

1. **Multiple diverse real benign sources** in training, to cover the manifold — not one.
2. **Never** put the real-attack evaluation source's benign into training (drop A1's design).
3. **Calibrate the operating threshold on a held-out real benign source**, not the training
   source — the degenerate 0-threshold is a direct consequence of calibrating on in-training
   benign.
4. Likely **real attacks in training too** (e.g. ModSecurity-2025 attack corpus), so "attack"
   is not merely "unlike my benign".

## Gate decision

**STOP.** Per the task, if the fix does not make external alert-rate acceptable, nothing else
matters. It does not. Parts B–E (rule baseline, cross-tool, ablations + figure, LLM/human
family relabelling) are **not** run pending review of this result and a revised Part A design.
No model is promoted to "deployed".

Artifacts: `models/content_v2/{partA_results.json, absolute_threshold_diagnostic.json,
manual_inspect_A1.csv, manual_inspect_A2.csv}` (models joblib gitignored, regenerable via
`python -m services.content_detector.train_partA`). Deterministic, seed 42.
