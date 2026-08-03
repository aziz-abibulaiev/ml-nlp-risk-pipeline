# TASK 06 Parts B–F — baseline, honest recall, ablations, family, figure

Branch `feat/content-detector-eval` off `ccea856`. Final evidence for the paper on the
deployed **text-only in-domain** model (Part A v3: 0.42 % FP at 100 % recall). Every recall is
at the **deployed operating point** (threshold = 1 % FPR on zanbil benign) and labelled
**in-distribution**, **cross-tool**, or **in-domain-FP**. Regex is baseline / partition /
explanation only — never a model feature or label. Seed 42, deterministic.

## Part B — learned detector vs ModSecurity-style rule baseline (head to head)

Rule baseline: the detection-regex family flags if any rule matches the fixed-point-decoded
request. Same data, deployed operating point:

| metric | learned (ML) | regex baseline |
|---|---|---|
| late-zanbil false-positive rate | 0.42 % | **0.09 %** |
| SR-BH recall (in-distribution) | **100.0 %** | 45.8 % |
| SR-BH recall, **marker-free** (n=21,692) | **100.0 %** | **0.0 %** |
| testbed custom recall (cross-tool) | **99.9 %** | 27.8 % |
| custom recall, **marker-free** (n=2,233) | **99.9 %** | **0.0 %** |

**Reading — where ML wins and where it doesn't.** The regex baseline has a *lower* false
positive rate (0.09 % vs 0.42 %): rules only fire on explicit tokens, so they rarely false-
alarm. But the regex catches only ~46 % of SR-BH attacks and ~28 % of the cross-tool set,
and **0 % of marker-free attacks** — and marker-free is over half the real attacks
(21,692 of ~40k). The regex is also brittle to payload variants (it misses `UNION ALL SELECT`
because its rule is `union\s+select`). The learned detector catches ~100 % including the
entire marker-free class and the variants. So **ML beats rules decisively on recall/coverage,
at a modest FP cost** (0.33 pp). For a detector whose job is to catch attacks the rules miss,
that trade is clearly worth it; if minimal FP were the only goal, rules are cheaper. Stated
straight.

## Part C — honest attack generalization: cross-tool at the deployed threshold

Per-family cross-tool recall, threshold = 1 % FPR on zanbil benign (paired with the 0.42 %
in-domain FP so it is one operating point):

| family | ffuf → custom | custom → ffuf |
|---|---|---|
| sqli | 100.0 % | 100.0 % |
| xss | 99.3 % | 100.0 % |
| cmd_injection | **98.0 %** | 100.0 % |
| ssti | 100.0 % | 100.0 % |
| path_traversal† | 100.0 % | 100.0 % |
| scan† | **35.0 %** | 100.0 % |

† shared payloads dropped first; nosql/ssrf excluded (ffuf slice off-family).

**This overturns the TASK 05 worry.** There, cmd_injection cross-tool collapsed to 17 % — but
that was at a testbed-calibrated threshold. **At the deployed operating point (1 % FPR on
zanbil benign), cmd_injection is 98 % and ssti 100 %: injection families are NOT missed in
deployment.** Only `scan` stays tool-sensitive (35 % ffuf→custom; custom scan is path
enumeration, structurally different from the ffuf wordlist). **Caveat:** these probe models
train on zanbil benign, so part of the high cross-tool recall reflects testbed attacks being
out-of-distribution to zanbil, not purely attack detection — the cleaner same-domain
cross-tool generalization is TASK 05's (family-dependent). The honest deployment answer to
"are cmd/ssti missed?" is: no, they are flagged at the deployed point.

## Part D — ablations (in-domain, deployed point)

| model | zanbil FP | SR-BH recall (in-dist) | custom recall (cross-tool) |
|---|---|---|---|
| **text-only (deployed)** | **0.42 %** | 100.0 % | 99.9 % |
| fused (text+numeric) | 0.65 % | 100.0 % | 100.0 % |

The numeric channel **raises FP even in-domain** (0.42 → 0.65 %) for a negligible recall gain
(custom 99.9 → 100 %) — consistent with v2's cross-source finding. **Text-only is the correct
deployed model.**

Marked vs marker-free recall (text-only, from Part B): text catches marker-free attacks at
100 % (in-dist) / 99.9 % (cross-tool) where the regex gets 0 %. The content channel covers
both the marked and marker-free lexical structure; what it structurally cannot cover — attacks
with no request-content signal at all (behavioural: brute force, credential stuffing, rate) —
is out of scope for this per-request detector and belongs to the behavioural detector.

## Part E — LLM + human family relabelling of SR-BH attacks (caveated real family validation)

100 SR-BH confirmed attacks (binary label reliable) were relabelled by family by an LLM
(Claude) reading each fixed-point-decoded request under a recorded rubric
(`part_e_family.py` header). LLM family distribution: sqli 51, ssrf 14, cmd_injection 8,
path_traversal 8, other 8, ssti 6, scan 5 — SQLi-dominated, as expected for a WordPress
honeypot hit by mass scanners.

- **Agreement: LLM vs the detection regex (independent automated second annotator) = 93.0 %**
  (n=57 where the regex fired a family). **Independent HUMAN review is pending (author) — not
  claimed here.** This is an automated cross-check, honestly labelled, not the WAMM human
  concordance.
- **Family head (TASK 05 testbed-trained) on this real set: 20 % overall accuracy.** Per
  family: sqli P=1.00 R=0.29, scan P=0.07 R=1.00, everything else 0. **"Which attack" does not
  transfer to real traffic** — the family head learned testbed payload styles, not real
  WordPress-attack styles. Reported as **LLM-assigned, regex-cross-checked (93 %), N=100, human
  review pending — NOT clean ground truth.** Binary detection transfers; family attribution
  does not. This is the honest ceiling on the family head (consistent with D18: family is a
  downstream marker, not a detection capability).

## Part F — central figure

`docs/figures/main_result.pdf` (+ png), colourblind-safe (Okabe-Ito), vector. Two panels:
(a) held-out real-benign FP across training regimes on a log scale — testbed-only 83 % →
single-source 90 % → cross-source 99.5 % → **in-domain 0.42 %** (the ~237× drop); (b) the
deployed operating point — in-domain 0.42 % FP at 100 % / 99.9 % recall vs the unusable
cross-source point. A reader grasps "works in-domain, degrades cross-domain, at this operating
point" from the figure alone.

## Honest read — is the paper's claim supported?

**Yes, precisely.**
- *In-domain working detector*: yes — 0.42 % FP (≈0.29 % true after WAMM), 100 % in-dist and
  99.9 % cross-tool recall.
- *Honest cross-source boundary*: yes — v2 shows it degrades off-domain; the domain-coverage
  requirement is the Threats-to-Validity.
- *ML beats rules on marker-free*: yes — 100 % vs 0 %, decisively. The one honesty adjustment:
  ML beats rules on **recall/coverage**, not on FP (the regex has a slightly lower FP); the
  claim must be stated as coverage, not universal superiority.
- Two additional honest limits, both measured: the numeric channel hurts FP (text-only
  deployed), and family attribution does not transfer to real traffic (20 % — binary yes,
  family no).

## Config / determinism / disjointness

Fixed declared hyperparameters (`config.py`), text-only deployed model, seed 42, deterministic.
All recall at the 1 % zanbil-FPR deployed threshold. In-domain temporal split disjointness and
SR-BH attack train/test disjointness asserted in `train_partA_v3.build_indomain_data`. Artifacts
under `models/content_v4/` (`partsBCD_results.json`, `partE_family_results.json`,
`partE_llm_labels.csv`; joblibs gitignored) and `docs/figures/`.
