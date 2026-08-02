# SR-BH 2020 → pipeline feature space: mapping decisions

Companion to `jobs/retrain/build_ml_numeric_from_srbh.py`.
Every choice recorded here changes what a feature *means* across the two corpora, so this
document is the source for the paper's Threats to Validity section.

## Dataset

| | |
|---|---|
| Name | SR-BH 2020 multi-label web attack dataset |
| Citation | Sureda Riera T., Bermejo Higuera J.R., Bermejo Higuera J., Sicilia Montalvo J.A., Martínez Herráiz J.J. *A new multi-label dataset for Web attacks CAPEC classification using machine learning techniques.* Computers & Security, 2022. `10.1016/j.cose.2022.102788` |
| Distribution | Harvard Dataverse `doi:10.7910/DVN/OGOIXX`, CC0 1.0 |
| File | `data_capec_multilabel.csv`, 436,437,661 bytes, MD5 `173ec515308bdce5aec19cfd5b792596` |
| Capture | 12 days, July 2020; internet-exposed WordPress honeypot; ModSecurity 2.9.2 + CRS 3.3.0 in detection-only mode |
| Rows | 907,815 |

The MD5 is verified before any processing; the build aborts on mismatch.

---

## ⚠️ The CAPEC class labels are not reliable ground truth

**The binary label is trustworthy. The 13 CAPEC class columns are not.**

The dataset authors manually and semi-automatically reviewed the ModSecurity tagging,
correcting the normal/attack assignment where necessary. That review makes the *binary*
label human-corrected rather than purely silver, and it is independent of this project's
own `rules/base_rules.json` — which is exactly why SR-BH breaks the circularity of
evaluating rule-generated attacks against rule-derived silver labels.

The per-class assignment did not survive the same scrutiny. Measured on this file:

| Rows labelled `66 - SQL Injection` | n | share |
|---|---|---|
| contain an actual SQLi marker (`union select`, `or 1=1`, `information_schema`, …) | 106,868 | **42.7 %** |
| contain **no** SQLi marker | 143,443 | **57.3 %** |

The 57.3 % are genuine attacks of entirely different classes that CRS 3.3.0 attributed to
SQLi — server-side template injection (`{{ data.sizes.full.url }}`), path traversal
(`etc%252Fpasswd`, `c%253A%255C`), and SSI injection (`<!--#EXEC cmd="ls /"-->`).

**Consequence for use:** the `capec_*` columns are carried through for provenance and
exploratory work only. They must not be used as multi-class ground truth without an
explicit caveat in any table or figure derived from them. The binary label is fine, and it
is the only label this project's cross-dataset evaluation depends on.

Class coverage overlap with this pipeline's 16 attack types is also small: only SQLi,
command injection and (approximately) brute force correspond. XSS, SSRF, IDOR, NoSQL
injection, LDAP injection, data exposure and business logic are absent as SR-BH classes.

---

## Field mapping

The extractor `jobs/retrain/ml_http_features.py` was written against CSIC-shaped input.
SR-BH exposes different raw fields, so the mapping below is where the two corpora are
forced into one feature space.

| Extractor argument | SR-BH source | Notes |
|---|---|---|
| `url` | `normalize_url(request_http_request, host=request_host, default_host="srbh.local")` | `request_http_request` is the request target: path plus optional query string |
| `method` | `request_http_method` | Mapped through `METHOD_MAP`; unknown verbs → `http_method_code = 0` |
| `body_text` | `request_body` | Non-empty in 2.9 % of rows |
| `content_type` | `request_content_type` | Drives `is_json_payload` |
| `header_values` | values of the 11 request header columns, **empty values omitted** | See below |
| `label` | `0` if `000 - Normal == 1`, else `1` | |
| `source` | literal `"srbh2020"` | |

### Header mapping — the decision that matters

SR-BH stores headers as a **fixed set of 11 columns**, one per header, present for every
row whether or not the request carried that header. CSIC stores a different fixed set of
11 (`User-Agent`, `Pragma`, `Cache-Control`, `Accept`, `Accept-encoding`, `Accept-charset`,
`language`, `host`, `cookie`, `content-type`, `connection`).

**Decision: assemble a header dict from the 11 SR-BH columns, dropping entries whose value
is the empty string, and pass only the surviving values to the extractor.**

The alternative — passing all 11 unconditionally — would make `header_count` a constant 11
for every row in the corpus, destroying a feature that carries real variance in CSIC. The
extractor already filters falsy values internally (`ml_http_features.py:145-147`), so
omitting empties is also what keeps this path consistent with the CSIC builder.

Expected effects, to be stated in Threats to Validity:

| Feature | Effect | Why |
|---|---|---|
| `header_count` | **Bounded above by 11 and structurally different from CSIC.** Reflects which of SR-BH's 11 recorded headers were populated, not how many headers the client actually sent. Any header outside the recorded 11 is invisible. | SR-BH's capture schema, not the traffic, sets the ceiling |
| `header_value_total_len` | **Systematically lower than a true header blob.** Sums only the 11 recorded values. In CSIC benign it is effectively bimodal (2 distinct values); in SR-BH it varies with cookie and user-agent length. | Same truncation as above |
| `header_value_max_len` | **Constant 99 in CSIC benign, genuinely variable in SR-BH.** In SR-BH it is almost always the length of the user-agent or cookie string. | CSIC benign repeats a fixed header template; SR-BH is real traffic |

None of these three features is comparable across corpora in absolute terms. A model
trained on CSIC and scored on SR-BH will see them shift. That shift is a real property of
the transfer, not a bug in the mapping, and it must be reported rather than corrected away.

Header names are rewritten to wire form (`request_user_agent` → `user-agent`) so that the
text artifact reads like a header blob rather than like SR-BH column identifiers. This
affects only `text_headers`; the numeric extractor receives values only, never names.

### Fields deliberately not used

`timestamp`, `src_ip`, `src_port`, `dst_ip`, `dst_port`, `request_http_protocol`,
`response_http_protocol`, `response_http_status_code`, `response_http_status_message`,
`response_content_length`.

The 24 features are computed from the request alone. Response fields would leak the
server's reaction into a detector that is supposed to decide before the response exists,
and the honeypot's single source IP makes the network 5-tuple meaningless here.

---

## Text artifacts for the NLP channel

`srbh_text.parquet` is keyed to `srbh_features.parquet` by `row_id` (0-based, file order).

| Column | Content |
|---|---|
| `text_uri` | raw `request_http_request`, **unmodified** |
| `text_body` | raw `request_body`, **unmodified** |
| `text_headers` | `";".join(f"{name}:{value}")` over non-empty headers, wire names |
| `label` | binary label |

### Normalisation: why the artifact stores raw text

`services/nlp_infer/main.py::_normalize_text` (line 113) and `common/text_norm.py::norm_text`
apply exactly **one** `urllib.parse.unquote_plus` pass, then `html.unescape`, then
whitespace collapse and lowercasing. That normalisation happens **at inference time**, on
the concatenated blob, not on the stored corpus.

Storing already-decoded text would therefore double-decode at inference and produce a
different string than production sees. The artifact stores raw text; the NLP channel
normalises it itself, exactly as it does for live traffic.

`text_headers` uses the `";".join(f"{k}:{v}")` format because that is what
`services/preprocess/main.py::_normalize_headers` (line 370) emits into the `headers` text
field that `nlp_infer` consumes.

That format is **lossy and matched deliberately**: header values legitimately contain both
separators — `;` in `User-Agent` (`Mozilla/5.0 (X11; Linux x86_64)`) and `:` in `Referer`
(`http://...`) — so the blob cannot be split back into name/value pairs. This does not
affect detection, because the NLP channel treats the blob as an unstructured bag of
characters and never parses it. It is recorded here because it means `text_headers` must
not be used as a structured header source by any future consumer.

### Double encoding

SR-BH contains genuinely double-encoded targets — e.g.
`/blog/etc%252Fpasswd/my-account/edit-profile/...`, where `%252F` decodes to `%2F` and only
then to `/`. The counts `rows_with_pct_encoding_in_uri` and `rows_double_encoded_in_uri` in
`srbh_stats.json` measure how many rows still contain a `%XX` token after one
`unquote_plus` pass.

**The single-pass normalisation in `text_norm.py` does not fully decode these.** A rule
matching `etc/passwd` will not fire on `etc%2Fpasswd` after one pass. This is a real
detection gap in the NLP channel on real-world traffic, and it is worth reporting as a
finding rather than silently adding a second pass here — changing the number of passes in
the artifact would make offline evaluation disagree with production.

---

## Imputation

None. `featurize_http_request` is total: every ratio is guarded by `max(1, len(...))` or an
explicit empty-input branch, and every length is a non-negative integer. The build asserts
zero NaN and zero infinite values across the 24 feature columns and fails if either appears.

## Determinism

Seed 42 for the stratified 50k subsample. No other stochastic step. The input is read in
file order and `row_id` is the 0-based file position, so the artifacts are byte-stable
across runs.
