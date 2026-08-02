#!/usr/bin/env bash
# Phase 2 entrypoint: full multi-tool corpus over VAmPI + crAPI, plus external ingestion.
# Generation only. Provenance-only labelling preserved. See docs/owasp_coverage.md.
#
# Design: benign runs first (distinct IPs), then all attack campaigns (distinct IPs per
# tool, so each tool has a distinct src_ip; families use >= 2 tools). Every attack tool runs
# in a container on corpus_net. crAPI is captured via a second proxy to its published port.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS="$REPO/corpus"
DATA="$REPO/data/corpus/phase2"
SECLISTS="${SECLISTS:-$HOME/.cache/corpus-tools/SecLists}"
NET="corpus_net"
PV="172.30.0.3"; PC="172.30.0.4"       # proxy IPs (vampi / crapi)
export RUN_ID="phase2_$(date -u +%Y%m%dT%H%M%SZ)"
PY="${PYTHON:-python3.13}"

mkdir -p "$DATA/raw" "$DATA/manifests" "$DATA/labelled" "$DATA/wordlists" "$DATA/splits" "$DATA/external"
: > "$DATA/raw/vampi_phase2.jsonl"; : > "$DATA/raw/crapi_phase2.jsonl"

echo "== build images =="
docker build -q -f "$CORPUS/Dockerfile.generator" -t corpus-generator "$CORPUS" >/dev/null
docker build -q -f "$CORPUS/Dockerfile.attacker"  -t corpus-attacker  "$CORPUS" >/dev/null

echo "== cap wordlists =="
head -600 "$SECLISTS/Fuzzing/Databases/SQLi/Generic-SQLi.txt" > "$DATA/wordlists/sqli.txt"
head -600 "$SECLISTS/Fuzzing/XSS/robot-friendly/XSS-Fuzzing.txt" > "$DATA/wordlists/xss.txt"
head -600 "$SECLISTS/Fuzzing/LFI/LFI-Jhaddix.txt" > "$DATA/wordlists/lfi.txt"
head -600 "$SECLISTS/Fuzzing/command-injection-commix.txt" > "$DATA/wordlists/cmd.txt"
head -600 "$SECLISTS/Discovery/Web-Content/common.txt" > "$DATA/wordlists/scan.txt"
cat "$SECLISTS/Fuzzing/template-engines-expression.txt" "$SECLISTS/Fuzzing/template-engines-special-vars.txt" 2>/dev/null > "$DATA/wordlists/ssti.txt"

echo "== bring up proxies (VAmPI + crAPI) =="
docker compose -f "$CORPUS/docker-compose.phase2.yaml" up -d

gen() {  local img="$1" ip="$2"; shift 2
  docker run --rm --network "$NET" --ip "$ip" \
    -v "$CORPUS":/corpus:ro -v "$DATA":/data -v "$SECLISTS":/seclists:ro \
    "$img" "$@"; }
atk() { gen corpus-attacker "$@"; }
ben() { gen corpus-generator "$@"; }

# readiness: talk to apps directly (not via proxy) so no infra flow is captured
gen corpus-generator 172.30.0.9 python -c "
import time,socket,urllib.request
for _ in range(60):
    try: urllib.request.urlopen('http://172.30.0.2:5000/',timeout=3); break
    except Exception: time.sleep(1)
for _ in range(60):
    try: socket.create_connection(('172.30.0.3',8080),3).close(); socket.create_connection(('172.30.0.4',8081),3).close(); break
    except Exception: time.sleep(1)
print('proxies ready')
"

echo "########## BENIGN (parallel, high volume: benign >= attack) ##########"
# seed VAmPI DB once directly (uncaptured); parallel workers use --skip-createdb
gen corpus-generator 172.30.0.9 python -c "import urllib.request; urllib.request.urlopen('http://172.30.0.2:5000/createdb',timeout=10); print('db seeded')"
bpids=()
for i in 0 1 2 3; do  # 4 VAmPI benign workers x 180 users
  ben 172.30.0.1$i python /corpus/benign/vampi_driver.py --target http://$PV:8080 \
    --manifest-dir /data/manifests --campaign-id benign_vampi_0$i --src-ip 172.30.0.1$i \
    --n-users 240 --seed $((42+i)) --skip-createdb & bpids+=($!)
done
for i in 6 7; do  # 2 crAPI benign workers x 55 users (distinct emails via campaign-id)
  ben 172.30.0.1$i python /corpus/benign/crapi_driver.py --target http://$PC:8081 \
    --manifest-dir /data/manifests --campaign-id benign_crapi_0$i --src-ip 172.30.0.1$i \
    --n-users 55 --seed $((42+i)) & bpids+=($!)
done
for p in "${bpids[@]}"; do wait "$p"; done

echo "########## CONTENT ATTACKS (VAmPI, >=2 tools/family) ##########"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
ffuf_run() { # family listfile point campaign
  atk 172.30.0.22 python /corpus/attacks/run_tool.py --family "$1" --tool ffuf --app vampi \
    --src-ip 172.30.0.22 --manifest-dir /data/manifests --campaign-id "$4" -- \
    ffuf -u "http://$PV:8080$3" -w "/data/wordlists/$2" -mc all -t 15 -s -H "User-Agent: $UA"; }
custom_run() { # family min campaign
  atk 172.30.0.24 python /corpus/attacks/custom_inject.py --target http://$PV:8080 \
    --family "$1" --manifest-dir /data/manifests --campaign-id "$3" --src-ip 172.30.0.24 --min-rows "$2" --seed 42; }

# sqli: sqlmap + ffuf(2 points) + custom
atk 172.30.0.21 python /corpus/attacks/run_sqlmap.py --target http://$PV:8080 \
  --manifest-dir /data/manifests --campaign-id sqli_vampi_sqlmap --src-ip 172.30.0.21 \
  --endpoint /users/v1/admin --level 5 --risk 3 --seed 42
ffuf_run sqli sqli.txt "/users/v1/FUZZ" sqli_vampi_ffuf_users
ffuf_run sqli sqli.txt "/books/v1/FUZZ" sqli_vampi_ffuf_books
custom_run sqli 150 sqli_vampi_custom
# xss
ffuf_run xss xss.txt "/users/v1/FUZZ" xss_vampi_ffuf
custom_run xss 150 xss_vampi_custom
# path_traversal
ffuf_run path_traversal lfi.txt "/users/v1/FUZZ" pt_vampi_ffuf
custom_run path_traversal 150 pt_vampi_custom
# cmd_injection
ffuf_run cmd_injection cmd.txt "/users/v1/FUZZ" cmd_vampi_ffuf
custom_run cmd_injection 150 cmd_vampi_custom
# ssti (small list -> ffuf 2 points + custom carries volume)
ffuf_run ssti ssti.txt "/users/v1/FUZZ" ssti_vampi_ffuf_users
ffuf_run ssti ssti.txt "/books/v1/FUZZ" ssti_vampi_ffuf_books
custom_run ssti 500 ssti_vampi_custom
# nosql (custom primary + ffuf small)
custom_run nosql_injection 500 nosql_vampi_custom
ffuf_run nosql_injection sqli.txt "/users/v1/FUZZ" nosql_vampi_ffuf  # placeholder list; labelled by campaign intent
# ssrf (custom primary + ffuf)
custom_run ssrf 500 ssrf_vampi_custom
ffuf_run ssrf lfi.txt "/users/v1/FUZZ" ssrf_vampi_ffuf
# scan: ffuf(dir discovery via SecLists) + custom scanner (path enum + verb tampering).
# nuclei was intended as the 2nd tool but its template provisioning was unreliable offline
# in this environment; the custom scanner is an independent, reproducible second tool.
ffuf_run scan scan.txt "/FUZZ" scan_vampi_ffuf
custom_run scan 500 scan_vampi_custom

echo "########## BEHAVIOURAL (VAmPI, many IPs -> many windows) ##########"
# 10 source IPs per family (>=50 windows each) at a modest rate (keeps row totals sane
# while each 55s session still spans ~6 ten-second windows -> ~60 windows/family)
pids=()
for i in $(seq 0 9); do
  atk 172.30.0.$((40+i)) python /corpus/attacks/behavioural.py --target http://$PV:8080 \
    --family brute_force --tool hydra --manifest-dir /data/manifests \
    --campaign-id bruteforce_$i --src-ip 172.30.0.$((40+i)) --duration 55 --rate 2 --seed 42 & pids+=($!)
done
for i in $(seq 0 9); do
  atk 172.30.0.$((50+i)) python /corpus/attacks/behavioural.py --target http://$PV:8080 \
    --family cred_stuffing --tool custom --manifest-dir /data/manifests \
    --campaign-id credstuff_$i --src-ip 172.30.0.$((50+i)) --duration 55 --rate 2 --seed 42 & pids+=($!)
done
for i in $(seq 0 9); do
  atk 172.30.0.$((60+i)) python /corpus/attacks/behavioural.py --target http://$PV:8080 \
    --family resource_exhaustion --tool custom --manifest-dir /data/manifests \
    --campaign-id resource_$i --src-ip 172.30.0.$((60+i)) --duration 55 --rate 2 --seed 42 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "########## CONTEXT ATTACKS (crAPI) ##########"
atk 172.30.0.30 python /corpus/attacks/crapi_context.py --target http://$PC:8081 \
  --family bola_idor --manifest-dir /data/manifests --campaign-id bola_crapi --src-ip 172.30.0.30 --n 200 --seed 42
atk 172.30.0.31 python /corpus/attacks/crapi_context.py --target http://$PC:8081 \
  --family mass_assignment --manifest-dir /data/manifests --campaign-id massassign_crapi --src-ip 172.30.0.31 --n 200 --seed 42
atk 172.30.0.32 python /corpus/attacks/crapi_context.py --target http://$PC:8081 \
  --family business_logic_abuse --manifest-dir /data/manifests --campaign-id bizlogic_crapi --src-ip 172.30.0.32 --n 200 --seed 42

echo "== tear down proxies =="
docker compose -f "$CORPUS/docker-compose.phase2.yaml" down >/dev/null 2>&1

echo "== label (provenance only) =="
$PY "$CORPUS/label/labeller.py" --capture "$DATA/raw/vampi_phase2.jsonl" \
  --manifest-dir "$DATA/manifests" --out "$DATA/labelled/vampi_phase2.parquet" --allow-unmatched
$PY "$CORPUS/label/labeller.py" --capture "$DATA/raw/crapi_phase2.jsonl" \
  --manifest-dir "$DATA/manifests" --out "$DATA/labelled/crapi_phase2.parquet" --allow-unmatched
$PY - "$DATA/labelled/vampi_phase2.parquet" "$DATA/labelled/crapi_phase2.parquet" "$DATA/labelled/testbed_phase2.parquet" <<'PYCAT'
import sys, pandas as pd
a, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
pd.concat([pd.read_parquet(a), pd.read_parquet(b)], ignore_index=True).to_parquet(out, index=False, compression="zstd")
print("merged testbed ->", out)
PYCAT

echo "== external ingestion =="
if [ -f "$DATA/external/zanbil.parquet.stats.json" ] && [ -z "${ZANBIL_FORCE:-}" ]; then
  echo "   zanbil already ingested (set ZANBIL_FORCE=1 to redo): $(cat "$DATA/external/zanbil.parquet.stats.json")"
else
  $PY "$CORPUS/ingest/zanbil_ingest.py" --out "$DATA/external/zanbil.parquet" ${ZANBIL_MAX:+--max-rows $ZANBIL_MAX}
fi
$PY "$CORPUS/ingest/srbh_ingest.py" --out "$DATA/external/srbh.parquet"

echo "== split by session =="
$PY "$CORPUS/split/split_corpus.py" --corpus "$DATA/labelled/testbed_phase2.parquet" --out-dir "$DATA/splits" --seed 42

echo "== verify + integrity + manifests =="
$PY "$CORPUS/verify_phase2.py" --testbed "$DATA/labelled/testbed_phase2.parquet" \
  --manifest-dir "$DATA/manifests" --labeller "$CORPUS/label/labeller.py" \
  --split-dir "$DATA/splits" --run-id "$RUN_ID" \
  --zanbil-stats "$DATA/external/zanbil.parquet.stats.json" \
  --srbh-stats "$DATA/external/srbh.parquet.stats.json" \
  --docs-out "$REPO/docs/corpus"

echo "Phase 2 complete."
