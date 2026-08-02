# Attack → OWASP → detector coverage manifest

Maps every `attack_family` in the corpus's controlled vocabulary to the OWASP category it
represents and to the detector expected to catch it. The third column is deliberately
honest: several families are generated **only to have ground truth**, not because the
per-request ML is expected to catch them.

Detector legend:
- **content** — per-request content detector (method + path + query + body + headers).
- **behavioural** — windowed detector over a session/source key (rate, error mix, auth
  failures, credential diversity, inter-arrival times).
- **context** — needs application authorization context ("who is allowed to see what").
  Neither ML component is expected to catch these from a single request or a rate window.
  Generated and labelled for ground truth and to measure the honest ceiling; this is the
  context/agent track (D12), not the per-request ML.

| attack_family | OWASP API Top 10 (2023) | OWASP Web Top 10 (2025) | Expected detector | Notes |
|---|---|---|---|---|
| `sqli` | API8 Security Misconfiguration (injection surface) | A03 Injection | **content** | Lexical/structural markers in query/body. Phase 1 family. |
| `xss` | — (web-centric) | A03 Injection | **content** | Script/markup in reflected params. |
| `cmd_injection` | API8 | A03 Injection | **content** | Shell metacharacters in params/body. |
| `nosql_injection` | API8 | A03 Injection | **content** | Operator injection (`$ne`, `$gt`) in JSON bodies. |
| `ldap_injection` | API8 | A03 Injection | **content** | LDAP filter metacharacters. |
| `ssti` | API8 | A03 Injection | **content** | Template syntax (`{{ }}`, `${ }`). |
| `path_traversal` | API8 | A01 Broken Access Control (path) | **content** | `../`, encoded traversal, absolute paths. |
| `ssrf` | **API7 Server-Side Request Forgery** | A10 SSRF | **content** | Internal/loopback URLs in request fields. |
| `scan` | API8 (probing, verbose errors, odd methods) | A05 Security Misconfiguration | **content** + **behavioural** | Per-request oddity (unknown paths, weird verbs) plus burst/coverage pattern over a window. |
| `brute_force` | **API2 Broken Authentication** | A07 Identification & Auth Failures | **behavioural** | Many auth attempts, one principal/source, high 4xx. Windowed. |
| `cred_stuffing` | **API2** | A07 | **behavioural** | Many distinct credentials, one source, high 4xx. Windowed. |
| `resource_exhaustion` | **API4 Unrestricted Resource Consumption** | — | **behavioural** | High rate and/or oversized payloads over a window. |
| `bola_idor` | **API1 Broken Object Level Authorization** | A01 Broken Access Control | **context** | A *valid* token accessing another user's object. The request looks legitimate; only per-user authorization context reveals it. **Per-request ML is not expected to catch this.** |
| `mass_assignment` | **API3 Broken Object Property Level Authorization** | A08 Software & Data Integrity | **context** | Extra privileged fields in an otherwise valid body. Needs the object's property model. **Not per-request ML.** |
| `business_logic_abuse` | **API6 Unrestricted Access to Sensitive Business Flows** | A04 Insecure Design | **context** | e.g. coupon reuse — each request is individually valid; abuse is only visible against business-flow state. **Not per-request ML.** |

## Honest scope statement

The per-request **content** detector is expected to catch the injection family, SSRF, and
the per-request face of scanning. The **behavioural** detector is expected to catch
brute-force, credential stuffing, resource exhaustion, and the burst face of scanning.

`bola_idor`, `mass_assignment`, and `business_logic_abuse` are **context** classes: the
individual HTTP transactions are indistinguishable from legitimate ones without knowing who
owns which object and what a flow is allowed to do. We generate and label them from the
attack script's *intent* (not from content) so that:
1. there is ground truth for the context/agent track (a later iteration, possibly a second
   paper), and
2. we can measure the honest ceiling of what per-request + windowed ML can achieve, by
   showing these classes sitting at chance for those detectors.

Claiming the per-request model detects these would be exactly the kind of inflated result
this corpus exists to avoid.

## OWASP categories intentionally out of scope

- **API9 Improper Inventory Management** and **API10 Unsafe Consumption of APIs** are not
  runtime per-request detections; they concern inventory hygiene and third-party API trust.
  No `attack_family` maps to them.
- **API5 Broken Function Level Authorization** is, like BOLA, an authorization-context
  problem; if generated later it joins the **context** track, not the ML detectors.
