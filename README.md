# ML/NLP risk pipeline for HTTP traffic

Code and evaluation for "Context-Aware ML/NLP Pipeline for Real-Time Anomaly Detection and
Risk Assessment in Cloud API Traffic" (Mach. Learn. Knowl. Extr. 2026, 8, 25).

The pipeline runs as Docker Compose services: preprocess, ml-inference, nlp-inference, fusion,
alerting and orchestrator, with Redis and OpenSearch.

## Requirements

- Docker with Compose v2 (12 GB of memory for the Docker VM is enough)
- Python 3.12 or newer
- A `.env` file:

  ```bash
  cp .env.example .env    # then set ORCH_TOKEN and OPENSEARCH_ADMIN_PASSWORD
  ```

- The Kaggle datasets in `~/.cache/kagglehub/datasets/`:

  | Kaggle dataset | File |
  |---|---|
  | `mahendradata/web-server-access-logs-labeled` | `labeled.csv` |
  | `moradrawashdeh/attack-simulation-lab` | `attack-simulation-alert.csv` |
  | `ispangler/csic-2010-web-application-attacks` | `csic_database.csv` |
  | `evg3n1j/httpparamsdataset` | `payload_full.csv` |
  | `cyberprince/web-application-payloads-dataset` | `WEB_APPLICATION_PAYLOADS.jsonl` |

  ```bash
  pip install kagglehub
  python -c "import kagglehub; [kagglehub.dataset_download(d + '/versions/1') for d in [
    'mahendradata/web-server-access-logs-labeled', 'moradrawashdeh/attack-simulation-lab',
    'ispangler/csic-2010-web-application-attacks', 'evg3n1j/httpparamsdataset',
    'cyberprince/web-application-payloads-dataset']]"
  ```

## Reproduce the results

```bash
make venv    # Python environment in .venv, once
make all     # about 2 hours on an Apple M3 Pro
```

`make all` builds the data splits, trains the ML and NLP models, scores a validation stream,
selects the thresholds and fusion weights on it, sends the mixed stream of 26,025 events once
with hysteresis and once without, computes the tables and figures and measures latency.
Existing runs are never overwritten. The steps can also be run one by one with `make data`,
`make images`, `make train`, `make validation`, `make freeze`, `make mixed`, `make tables` and
`make bench`.

The results are written to:

| File | Content |
|---|---|
| `runs/mixed/tables/table4.csv` | precision, recall, F1 and FPR at Medium+ and Low+, with 95% confidence intervals |
| `runs/mixed/tables/table4_distinct.csv` | the same with every distinct request counted once |
| `runs/mixed/tables/table5.csv`, `table5_by_group.csv` | risk levels by class and by attack type |
| `runs/mixed/tables/fig*.png` | figures |
| `runs/mixed/ablations.json` | other fusion weights, including the NLP branch alone |
| `runs/mixed/header_features.json` | attacks scored with the header features of benign events |
| `runs/mixed/run_info.json`, `runs/bench-replay/results.json` | latency and throughput |
| `models/ml/ml_test_eval.json`, `models/nlp/train_report.json` | held-out ML and NLP test metrics |
| `runs/validation/frozen_config.json` | the selected thresholds and fusion weights |
| `runs/mixed-no-hysteresis/` | the same tables without hysteresis |

To compare a new run with the reference outputs of the article:

```bash
make compare
```

## Results without re-running the pipeline

The reference outputs are in `runs/reference/`. The per-event outputs and the trained models
are attached to release v0.0.2 because of their size. Download the assets into `dist/`, then:

```bash
cd dist && shasum -a 256 -c SHA256SUMS.txt && cd ..
for r in validation mixed mixed-no-hysteresis; do gunzip -c dist/${r}_events.jsonl.gz > runs/reference/$r/events.jsonl; done
tar -xzf dist/models.tar.gz
shasum -a 256 -c runs/reference/validation/frozen_hashes.txt
make reference-tables
```

## Run the pipeline

```bash
docker compose up -d
```

The defaults of `docker-compose.yaml` are the frozen configuration
(`runs/reference/validation/frozen.env`). Send one event:

```bash
set -a; . ./.env; set +a
curl -s http://localhost:8000/process -H "X-Orch-Token: $ORCH_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"event": {"endpoint": "/api/items?id=1%27%20OR%201=1--", "method": "GET", "headers": {"User-Agent": "curl"}, "payload": "id=1%27%20OR%201=1--"}}'
```

`make down` stops the services.

## Notes

- All random choices use fixed seeds, 42 for the training subsamples and the NLP split and
  20261110 for the mixed stream and the attack generator.
- Hysteresis state is keyed by a hash that includes the send time to the second. A new run with
  hysteresis can therefore differ from the reference in a few benign events between Normal and Low.
- Neural network training can differ in the last digits on other CPU types. Latency depends on
  the hardware.
