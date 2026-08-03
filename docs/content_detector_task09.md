# TASK 09 — WebLog-2025 (organic FP domain #2) + SEC EDGAR (pure-benign FP stress test)

Branch `feat/weblog-edgar-domains`. Biblio-US17 is auth-gated (TASK 07c), so this substitutes
two openly-downloadable real sources to (a) add a second organic-benign deployment domain and
(b) stress-test cross-domain false positives on a large pure-benign corpus. Same discipline as
TASK 06/07: **text-only** model, fixed declared hyperparameters (`config.py`), threshold on each
domain's **own calib** benign (never test), **regex only** as a WAMM inspection aid on the
**decoded** request (never a feature or a training filter), **real benign never rule-filtered**,
seed 42, deterministic, disjointness asserted. New artifacts under `models/content_v5c/`
(existing `models/` untouched).

## Fetch & parse outcomes (offline, reproducible)

| source | fetched | parsed | unique after dedup | labels |
|---|---|---|---|---|
| **WebLog-2025 Org Y** (Zenodo `10.5281/zenodo.20001206`, MIT) | `org-y.zip` 390 MB, 160 access-log files, 22 subdomains (multi-tenant WordPress hosting, CyberPanel/OpenLiteSpeed) | 7,272,920 lines, **5 unparsed** (7e-7) | **757,845 benign / 129 attack** | dataset's own YAML rules (30), applied to the raw access line incl. User-Agent |
| **SEC EDGAR** (Log File Data Sets, open) | 3 daily zips: `log20150601`, `log20160301`, `log20170502` | 1.2 M rows/day sampled | **1,661,635 benign** (pure) | benign-only (no attack labels) |

WebLog labels are **rule-derived** (like SR-BH / ModSec, *not* like human-verified Biblio LVL1):
the shipped `organization-y.yaml` matches `%`-wildcard substrings (AND within a rule) against the
log line. Reproduced faithfully; the resulting attack set is thin and noisy (see §A.2). EDGAR
URIs reconstructed as `/Archives/edgar/data/{cik}/{accession}{extention}`, `GET`, dedup on
`method|path`.

## Consolidated multi-source validation table (1 % FPR budget, threshold on own calib)

| deployment domain (in-domain benign) | benign nature | in-domain FP @1% (WAMM-corr) | @5% | recall SR-BH | recall testbed cross-tool | recall ModSec |
|---|---|---|---|---|---|---|
| **zanbil** (e-commerce, TASK 07) | organic, single stationary site | **0.45 % (~0.30 %)** | — | 100 % | 99.9 % | 92.9 % |
| **EDGAR** (SEC filings archive) | organic, single stationary structure | **0.61 % (0.61 %)** | 2.97 % | 100 % | 100 % | **100 %** |
| **WebLog Org Y** — **hash split** | organic, 22 WordPress sites (structural split) | **0.99 % (0.90 %)** | 4.97 % | 100 % | 100 % | 93.4 % |
| **WebLog Org Y** — **temporal split** | organic, 22 WordPress sites (temporal split) | **20.0 % (19.0 %)** | 60.3 % | 100 % | 100 % | 93.4 % |

zanbil and EDGAR — two clean single-distribution organic domains — corroborate the in-domain
finding (**0.45 % / 0.61 %**, both at 100 %/≈100 % recall across three real-attack sources).
WebLog corroborates it too **under a structural (hash) split (0.99 %)** but blows up to **20 %
under a temporal split** — and that gap is the finding of this task.

## §A — WebLog: in-domain works (0.99 %), but temporal drift on a multi-tenant host costs 20×

The decisive control is **hash vs temporal split** on the *same* WebLog benign, *same* fixed
attack corpus, *same* protocol:

- **Hash split** (test benign drawn from the same distribution as train — no time shift):
  **0.99 % FP @1 %**, 4.97 % @5 %. WordPress benign *is* separable from the WordPress-honeypot
  attack corpus when training benign is representative.
- **Temporal split** (train pre-2025-04-20, test after, incident day 04-29 in test):
  **20.0 % FP @1 %**, 60.3 % @5 %.

So the 20 % is **temporal distribution shift, not a structural CMS-token collision** — if it were
the latter, the hash split would fail too, and it does not. What shifts is *which sites* the
benign comes from: **Org Y is a multi-tenant hosting provider — 22 distinct WordPress sites**
(`/alatberat/…`, `/mekatronika/…`, `/mesin/…`, `/bispro/…`, `/wordpress2024/…`, `/home20/…`,
`/v9/…`, each its own themes, plugins, upload trees). A temporal split puts **tenants/pages that
came online later into the test set that were absent from training**, and their unfamiliar (but
benign) paths score as attacks. The top-100 temporal alerts are 95 % genuine WordPress traffic
(WAMM 5/100 plausible), and they are dominated by these later-appearing per-site subtrees:

```
/alatberat/wp-content/uploads/2020/09/868.jpg   /mekatronika/wp-content/uploads/2016/08/10.png
/wordpress2024/7.php   /home20/7.php   /v9/757.php/wp-json/   /wp-content/plugins/hide-my-wp/...
```

This is the **coverage principle in the time dimension**: in-domain training holds when the
deployment is a single coherent, stationary traffic distribution (zanbil, EDGAR — temporal FP
stays <1 %); it degrades when "the domain" is really an *aggregate of many sub-distributions that
onboard over time* (a multi-tenant host), because a temporal cut leaves later tenants uncovered.
The fix is representativeness, not more tuning: train on benign that covers all tenants (hash
split → 0.99 %), or retrain as sites onboard. It is **not** a CMS-collision with the honeypot
attack set, and it is **not** benign contamination (95/100 top temporal alerts are real benign).

### §A.2 — WebLog's rule-derived attack labels are behavioural / noise, not content-injection

After dedup to distinct request *content*, the 129 "attacks" break down as:

| rule id | unique | what they actually are | detector recall @1% |
|---|---|---|---|
| `file_inclusion` (113) | 113 | **mostly rule false-positives** — the `=https://` filter fires on legit WP `redirect_to=https://…` login redirects and `wp-json/oembed?url=…`; **1** real (`/?url=http://41.36.41.36/latest/meta-data/`, an SSRF/metadata probe) | 71.7 % |
| `bruteforce_login_web` (12) | 12 | **behavioural** — repeated `POST /wp-login.php` (4.58 M raw requests → 12 distinct contents); content-identical to a legitimate login, not per-request detectable | 100 % |
| `dir_scan` (4) | 4 | real planted webshells (`/1701.php`, `HelloDollyV2/1695.php`) dropped by `ShellBot`/`python`/`curl/8.5.0` UA | 100 % |

So the 75 % "WebLog attack recall" number is **not meaningful as content-detection**: `dir_scan`
webshells are caught (real, 4/4), `bruteforce_login_web` is behavioural (caught only because
`/wp-login.php` sits in the attack-associated token space, not because a single login is
detectable), and `file_inclusion` is ~99 % rule false-positives. WebLog's honest contribution is
its **757 k organic benign as an FP domain**, not attack recall — the same way SR-BH's honeypot
benign was too contaminated to serve as an FP benchmark, WebLog's rule-derived attacks are too
noisy/behavioural to serve as a content-recall benchmark.

## §B — EDGAR: pure-benign FP stress test

**(1) Cross-domain (deployed zanbil model on EDGAR benign).** At the deployed operating point
(threshold **0.0118**; model's own recomputed 1 % zanbil threshold 0.0129): the zanbil model
alerts on **100.0 %** of 1.66 M EDGAR benign URLs (100 % at either threshold). The same model
alerts on **97.4 %** of WebLog benign. **A model trained on one organic domain flags ~100 % of a
different organic domain as malicious** — the single hardest statement of the coverage principle
in the whole project, on a corpus that is provably pure-benign. (WAMM top-50 are all benign
archive URLs; incidentally the very top few expose real WordPress-scanner probes *inside* the
EDGAR logs — `…/wp-admin/admin-ajax.php` appended to filing paths — i.e. even "pure-benign" web
logs carry a trace of background scanning, the WAMM caveat again.)

**(2) EDGAR in-domain.** Cross-year temporal split (train ≈2015→2016, test 2016→2017 filings):
**0.61 % FP @1 %** (WAMM-corr 0.61 %, zero marked attacks in the top-100), 2.97 % @5 %, with
**100 % recall on SR-BH, testbed cross-tool, and ModSec**. A large, clean, homogeneous organic
benign domain in-domain-trains to <1 % FP and full cross-source recall — a second clean
confirmation alongside zanbil.

## Honest read

- **The in-domain-works claim now has two clean organic confirmations (zanbil 0.45 %, EDGAR
  0.61 %)** and a large pure-benign cross-domain failure (zanbil→EDGAR **100 %** alert) that
  makes the coverage principle undeniable.
- **WebLog is the instructive nuance**: in-domain FP is **0.99 % (hash) but 20 % (temporal)** on
  a multi-tenant WordPress host. The detector separates WordPress benign from a
  WordPress-*honeypot* attack corpus fine when training benign is representative — the failure is
  **temporal distribution shift from tenant heterogeneity** (new sites onboard over the capture
  window), not a CMS-token collision and not contamination (95/100 top alerts are genuine
  benign). This sharpens the deployability boundary in the time dimension: *in-domain training
  holds for a single stationary traffic distribution; a multi-tenant aggregate needs benign
  coverage of all tenants (or periodic retraining as sites onboard).*
- **Both new sources are real and openly reproducible**, unlike the gated Biblio-US17. Neither
  was padded or tuned toward a nicer number; the WebLog 20 % temporal result is reported as-is,
  alongside the 0.99 % hash control that explains it.

## Config / determinism / disjointness

Text-only; fixed declared hyperparameters (`config.py`); seed 42; deterministic. Per-domain
benign train/calib/test disjointness asserted in `multidomain.split_benign`; SR-BH attack
train/test disjoint. WebLog labels via the dataset's own `organization-y.yaml` applied to the raw
access line (`ingest_task09.py`); EDGAR URIs reconstructed from the official CSV schema. Artifacts:
`models/content_v5c/task09_results.json`, `text_only_weblog.joblib`, `text_only_edgar.joblib`,
`manual_inspect_{weblog,edgar_crossdomain,edgar_indomain}.csv` (joblibs gitignored). Parquets
under `data/corpus/phase2/external/{weblog2025,edgar}.parquet` (gitignored).
