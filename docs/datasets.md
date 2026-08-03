# Canonical dataset registry

Single source of truth for every dataset in the project — role, nature, coordinates, status.
Experiment scripts load from these paths (`services/content_detector/sources.py`,
`data/corpus/phase2/external/`). "nature" ∈ {real, testbed, semi-synthetic}. Status reflects
what is actually fetchable/used offline as of TASK 08.

## A.1 Real benign / deployment domains (in-domain-split capable)

| id | source | nature | body | size (dedup) | role | coordinates / status |
|---|---|---|---|---|---|---|
| `zanbil` | e-commerce nginx (Iran, 2019) | real | no | ~894k | **deploy domain #1 (done, v3/v5)** — 0.45% in-domain FP | Kaggle `eliasdabbas/web-server-access-logs` — **local (cached)** |
| `srbh` | WordPress honeypot (2020) | real | 2.9% | 90,917 benign / 287,710 attack | deploy domain #2 (FP **unmeasurable** — benign contaminated) + attacks | Harvard Dataverse `doi:10.7910/DVN/OGOIXX`; MD5 `173ec515308bdce5aec19cfd5b792596` — **local** |
| `biblio_us17` | Univ. Seville library, Drupal (2017) | real | no | 47.9M (42.5M benign + 328k attack) | intended deploy domain #3 + best-labelled attacks | DOI `10.12795/11441/148254` (idus.us.es) — **GATED: 401 auth-required, `canDownload`=0 for anonymous, CAPTCHA/login. NOT fetchable offline (TASK 07c).** |
| `weblog2025` | multi-tenant WordPress host (2025) | real | no | **757,845 benign / 129 attack** (dedup) | deploy domain #3 — FP **0.99% hash / 20% temporal** (multi-tenant drift, TASK 09) | Zenodo `10.5281/zenodo.20001206`, MIT — **local, parsed via `ingest_task09.py`** (Org Y, 22 sites; labels from shipped `organization-y.yaml`) |
| `edgar` | SEC EDGAR filings archive | real | no | **1,661,635 benign** (3 days, dedup) | deploy domain #4 + pure-benign FP stress test — in-domain **0.61%**, zanbil-model cross-domain **100% alert** (TASK 09) | SEC EDGAR Log File Data Sets (open) — **local, parsed via `ingest_task09.py`** |
| `nasa`,`clarknet`,`calgary` | Internet Traffic Archive (1990s) | real | no | 8.4k / 18.4k / 8.2k | benign diversity (LOSO only; too small/simple for a domain) | ita.ee.lbl.gov — **local** (`data/corpus/phase2/external/ita/`) |
| `csic` | CSIC 2010 | semi-synthetic | yes | ~9.6k benign / 16k attack (dedup) | legacy comparison / LOSO | Kaggle `ispangler/csic-2010-web-application-attacks` — **local** |
| `kimjmin` | tech-blog Apache (2022) | real | no | 10,713 | rejected 3rd domain (< 50k unique) | Kaggle `kimjmin/apache-web-log` — vetted, too small |
| `vishnu0399` | — | **synthetic** | — | — | rejected (Faker-generated, year-2037 ts) | Kaggle — vetted, not real |

## A.2 Real attack sources (recall)

| id | nature | labels | role | coordinates / status |
|---|---|---|---|---|
| `srbh_attacks` | real honeypot | binary reliable (CAPEC classes NOT — TASK 01) | recall | with `srbh` — local |
| `biblio_lvl1` | real library | human-verified (Snort+ModSec+Nemesida+expert), LVL1–4 | would be most reliable real-attack recall | **gated with biblio (unavailable)** |
| `modsec2025` | real WAF-flagged (2025) | CRS rule IDs | recall (attack-only) | Zenodo `10.5281/zenodo.17178461` (owasp.zip, 29.5 MB) — **local, parsed to 15,262 unique via `ingest_modsec.py`** |
| `testbed` | crAPI/VAmPI + real tools | provenance | recall + cross-tool | our honest corpus (`feat/honest-corpus` `17c7b69`) — local |

## A.3 Our testbed corpus

`feat/honest-corpus` (`17c7b69`): crAPI + VAmPI, provenance-labelled (labels from capture
provenance, never content), multi-tool. Benign 10,908 / attack 10,228; families per
`docs/owasp_coverage.md`. Cross-tool holdout capable (ffuf/sqlmap/custom + behavioural).

## A.4 Payload corpora (mutation base for adversarial eval)

| id | role | coordinates / status |
|---|---|---|
| `seclists` | base payloads (SQLi/XSS/LFI/cmd) | `~/.cache/corpus-tools/SecLists` — local (commit `e5e49caa…`) |
| `payloadbox` / `SunnyThakur25/Web-Application-Payloads-Dataset` | extra base payloads | GitHub — **not needed** (testbed custom-tool payloads + SecLists sufficed for TASK 08 Part B) |

## A.5 Explicitly NOT used (recorded for Threats to Validity)

- **CICIDS2017 / CSE-CIC-IDS2018 / UNSW-NB15** — flow-level, **no request payload** → the
  content/NLP channel is inapplicable in principle (not a workaround gap — a granularity
  mismatch).
- **ECML/PKDD 2007** — aged; legacy comparison only, not used.
- **Biblio-US17 / biblio_lvl1** — the best human-verified real dataset, but **auth-gated**
  (idUS login + CAPTCHA); could not be fetched offline. Recorded as an access limitation, not
  a substitution.
- **No public dataset provides real request bodies / API-JSON at scale** (PII constraints) →
  structural field-wide gap; the body channel is validated only on SR-BH (2.9% bodies) and the
  testbed. Declared in Threats to Validity.
