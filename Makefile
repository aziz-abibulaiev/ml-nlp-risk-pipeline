# Evaluation pipeline, see README.md.
# Requires Docker with Compose v2, Python 3.12+, the Kaggle datasets in ~/.cache/kagglehub
# and a .env (see .env.example).
#
#   make venv              # host Python environment in .venv (once)
#   make all               # full chain: data, training, validation, freeze, mixed stream, tables, benchmark
#   make reference-tables  # regenerate tables and figures from the reference outputs in runs/reference
#   make compare           # compare the tables of a fresh run with the reference tables

# Host Python (`make venv` creates .venv)
PY      ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
RETRAIN ?= mlnlp-retrain
STACK    = opensearch os-init redis preprocess ml-inference nlp-inference fusion alerting orchestrator
SVC_BUILD = preprocess ml-inference nlp-inference fusion alerting orchestrator

# Run directories under runs/
VAL    ?= validation
MIXED  ?= mixed
NOHYST ?= mixed-no-hysteresis
BENCH  ?= bench-replay
REF     = reference

# Settings while the validation stream is scored. The freeze uses only the raw scores.
VAL_SETTINGS = ML_THRESHOLD=0.7 NLP_THRESHOLD=0.60 FUSION_ALPHA=0.6 FUSION_BETA=0.4 \
            FUSION_LEVELS=0.3,0.5,0.6,0.8 FUSION_NORMAL_THR=0.0

# .env and the frozen configuration
FROZEN = . ./.env; . runs/$(VAL)/frozen.env; MODELS_DIR=./models; IMPACT_POLICY_HOST=./policies/impact_policy.json

.PHONY: all venv data images train validation freeze mixed tables bench reference-tables compare down

all: data images train validation freeze mixed tables bench

venv:
	python3 -m venv .venv
	PIP_USER=0 .venv/bin/python -m pip install -r scripts/evaluation/requirements-host.txt

data:
	$(PY) scripts/evaluation/prepare_splits.py
	$(PY) scripts/evaluation/build_ml_features.py

images:
	docker build -f services/retrain/Dockerfile -t $(RETRAIN) .
	docker compose build $(SVC_BUILD)

# Training, ML calibration and evaluation on the held-out ML test split
train:
	docker run --rm -v "$$PWD":/w -w /w -e PYTHONHASHSEED=0 --entrypoint python $(RETRAIN) scripts/evaluation/train_ml.py
	docker run --rm -v "$$PWD":/w -w /w -e PYTHONHASHSEED=0 -e RULES_FILE=/w/rules/base_rules.json --entrypoint python $(RETRAIN) scripts/evaluation/train_nlp.py
	docker run --rm -v "$$PWD":/w -w /w --entrypoint python $(RETRAIN) scripts/evaluation/calibrate_ml.py
	docker run --rm -v "$$PWD":/w -w /w --entrypoint python $(RETRAIN) scripts/evaluation/eval_ml_test.py

validation:
	$(VAL_SETTINGS) docker compose up -d --force-recreate $(STACK)
	sleep 30
	set -a; . ./.env; set +a; $(VAL_SETTINGS) $(PY) scripts/evaluation/run_stream.py --set validation --run-id $(VAL) --concurrency 8

# Thresholds and fusion weights, then the SHA-256 of every frozen file
freeze:
	docker compose run --rm --no-deps -v "$$PWD":/w -w /w --entrypoint python fusion \
	  scripts/evaluation/select_thresholds.py --run $(VAL)
	shasum -a 256 runs/$(VAL)/frozen_config.json runs/$(VAL)/frozen.env policies/impact_policy.json \
	  models/ml/*.joblib models/ml/*.json models/nlp/*.joblib models/nlp/*.json > runs/$(VAL)/frozen_hashes.txt

# The mixed stream, with and without hysteresis
mixed:
	for H in 1 0; do \
	  set -a; $(FROZEN); HYST_ENABLED=$$H; set +a; \
	  docker compose up -d --force-recreate ml-inference nlp-inference fusion && sleep 20 && \
	  docker compose exec -T redis redis-cli FLUSHALL && \
	  RID=$(NOHYST); [ $$H = 1 ] && RID=$(MIXED); \
	  $(PY) scripts/evaluation/run_stream.py --set mixed --run-id $$RID || exit 1; \
	done

# Tables, ablations, header diagnostic and figures
tables:
	$(PY) scripts/evaluation/make_tables.py --run $(MIXED) --run $(NOHYST)
	set -a; $(FROZEN); set +a; \
	  docker compose run --rm --no-deps -v "$$PWD":/w -w /w --entrypoint python fusion \
	  scripts/evaluation/ablate.py --val $(VAL) --mixed $(MIXED)
	docker run --rm -v "$$PWD":/w -w /w --entrypoint python $(RETRAIN) \
	  scripts/evaluation/header_features.py score --mixed $(MIXED)
	set -a; $(FROZEN); set +a; \
	  docker compose run --rm --no-deps -v "$$PWD":/w -w /w --entrypoint python fusion \
	  scripts/evaluation/header_features.py decide --val $(VAL) --mixed $(MIXED)
	$(PY) scripts/evaluation/make_figures.py --run $(MIXED)

# Latency of the log-replay endpoint
bench:
	set -a; $(FROZEN); HYST_ENABLED=1; set +a; \
	  docker compose up -d --force-recreate orchestrator ml-inference nlp-inference fusion && sleep 30 && \
	  $(PY) scripts/evaluation/bench_replay.py --run-id $(BENCH)

# Tables and figures from the reference outputs
reference-tables:
	$(MAKE) tables VAL=$(REF)/validation MIXED=$(REF)/mixed NOHYST=$(REF)/mixed-no-hysteresis

# Fails if any table differs
compare:
	rc=0; \
	$(PY) scripts/evaluation/compare_runs.py --run $(MIXED) --ref $(REF)/mixed || rc=1; \
	$(PY) scripts/evaluation/compare_runs.py --run $(NOHYST) --ref $(REF)/mixed-no-hysteresis || rc=1; \
	exit $$rc

down:
	docker compose down
