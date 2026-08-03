# TASK 08 Part B — adversarial robustness (static WAF-bypass evaluation)

Branch `feat/adversarial-eval`. Turns the declared "evasion" limitation into a measured
result. Static WAF-bypass repertoire (technique lineage: BWAFSQLi ACM TOSEM 2025, AdvSQLi
arXiv:2401.02615, WAFBooster arXiv:2501.14008) applied deterministically (seed 42) to
held-out **testbed custom-tool** attack payloads (never in training), per family. **No
adversarial training** — this is evaluation only. Text-only deployed model; fixed
hyperparameters; recall at the deployed operating point.

## Two detectors, because one confounds the answer

- **Deployed zanbil model** (threshold = 1 % FPR on zanbil benign, 0.0129): its recall is
  **100 % on every mutation**. But this is largely the OOD effect — zanbil benign scores ≈0,
  the threshold is ≈0.01, and *any* injection-like content (mutated or not) is unlike zanbil
  e-commerce traffic and clears it. It measures "does obfuscated content still look non-benign
  at deployment" (yes), not "does the attack pattern survive".
- **Attack-pattern probe** (threshold = 1 % FPR on **testbed** benign, 0.19): a text-only model
  trained to separate attacks from *same-domain* benign, so a mutation that breaks the pattern
  can actually evade it. **This is the honest evasion number.** Its baseline recall on
  held-out custom payloads is 66.6 % (the cross-tool gap from earlier tasks), and mutations
  move it from there.

## Degradation table (recall %, at the deployed operating points)

| mutation | deployed (zanbil) | **probe** (attack-pattern) | probe, **no decode** | regex baseline |
|---|---|---|---|---|
| original | 100.0 | 66.6 | — | 81.3 |
| case variation | 100.0 | 67.1 | 67.1 | 81.3 |
| inline comment `/**/` | 100.0 | **82.2** | 82.2 | **21.0** |
| versioned comment `/*!50000…*/` | 100.0 | 74.3 | 74.3 | 61.9 |
| whitespace substitution | 100.0 | **54.1** | 54.9 | 68.7 |
| URL encode | 100.0 | 66.6 | **54.5** | 81.3 |
| double URL encode | 100.0 | 66.6 | **45.0** | 81.3 |
| HTML-entity encode | 100.0 | 69.9 | 69.9 | **35.0** |
| equivalent syntax (`OR 1=1`→`OR 2>1`) | 100.0 | 70.5 | 70.5 | 66.7 |
| keyword→`CHAR()` | 100.0 | 75.2 | 75.2 | 66.7 |
| XSS tag/handler alternation | 100.0 | 87.5 | 87.5 | 100.0 |
| null / terminator | 100.0 | 67.4 | 67.4 | 84.6 |

## Findings

**1. The fixed-point decode is a concrete, measurable defense.** Compare probe *with* vs
*without* decode: URL-encode 66.6 % → **54.5 %**, double-URL-encode 66.6 % → **45.0 %**. With
the decode in the pipeline, encoding-based evasions are neutralised (recall stays at the 66.6 %
baseline); without it, they strip ~12–22 points. This is the D16 design decision paying off
against exactly the evasion class it was built for.

**2. Char n-grams survive most structural mutations.** Case variation (67 %), inline comments
(82 %), versioned comments (74 %), equivalent syntax (70 %), `CHAR()` keyword encoding (75 %),
XSS alternation (88 %) all hold at or above the 66.6 % baseline — several *rise*, because the
mutation adds attack-characteristic sub-tokens (e.g. `/**/`, `CHAR(`). Sub-token 3–5-grams are
robust to token-boundary tricks that defeat keyword regexes. (Note the pipeline is
case-sensitive — `lowercase=False` — yet case variation does not evade, because enough
sub-token n-grams are case-neutral.)

**3. Whitespace substitution is the most effective single evasion** (66.6 % → **54.1 %**):
replacing spaces with tabs / newlines / `/**/` changes many n-grams at once. This is the
residual structural weakness to name in Threats to Validity.

**4. ML beats the regex baseline under evasion, decisively on comment/entity tricks.** Inline
comments collapse the regex to **21 %** while the probe holds **82 %**; HTML-entity encoding
drops the regex to **35 %** vs probe **70 %**. The keyword regex is brittle to token-splitting
and encoding; the learned sub-token model is not. (On whitespace the two are comparable, and on
plain XSS alternation the regex still catches 100 % because its patterns are attribute-level.)

## Honest read / Threats to Validity framing

- **What it survives:** case, comment insertion (inline + versioned), equivalent SQL syntax,
  `CHAR()` encoding, XSS tag/handler alternation, and — thanks to the fixed-point decode —
  single and double URL encoding and HTML entities. These are the bulk of off-the-shelf WAF
  bypasses, and the learned detector is markedly more robust to them than the rule baseline.
- **What erodes it:** whitespace substitution (‑12 pts) and, *if the decode were removed*,
  heavy multi-pass encoding (‑22 pts) — i.e. the decode is load-bearing.
- **What this is NOT:** a static repertoire against a fixed model — **not an adaptive,
  white-box, or gradient-guided attacker** (AdvSQLi/WAFBooster-style search that optimises
  against the specific model). Those, and **adversarial training** as a defense, are explicit
  future work. We do not claim robustness to an adaptive adversary; we claim measured
  robustness to the standard obfuscation repertoire, better than rules, with the decode as a
  named, quantified mitigation and whitespace substitution as the named residual weakness.

## Config / determinism

Text-only; fixed declared hyperparameters (`config.py`); seed 42; deterministic mutations
(mapping mutated→original kept per payload). No adversarial training. Artifacts:
`models/content_v6/adversarial_results.json`. Base payloads: held-out testbed custom-tool
attacks (`data/corpus/phase2/labelled/testbed_phase2.parquet`).
