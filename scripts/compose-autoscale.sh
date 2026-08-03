#!/bin/sh
set -eu

PROJECT_DIR="${PROJECT_DIR:-/workspace}"
COMPOSE_FILE_PATH="${COMPOSE_FILE_PATH:-$PROJECT_DIR/docker-compose.yaml}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-ml-nlp-risk-pipeline}"

AUTOSCALE_INTERVAL_SEC="${AUTOSCALE_INTERVAL_SEC:-30}"
AUTOSCALE_CPU_UP_PCT="${AUTOSCALE_CPU_UP_PCT:-75}"
AUTOSCALE_CPU_DOWN_PCT="${AUTOSCALE_CPU_DOWN_PCT:-25}"
AUTOSCALE_COOLDOWN_CYCLES="${AUTOSCALE_COOLDOWN_CYCLES:-2}"

ML_MIN_REPLICAS="${ML_MIN_REPLICAS:-1}"
ML_MAX_REPLICAS="${ML_MAX_REPLICAS:-3}"
NLP_MIN_REPLICAS="${NLP_MIN_REPLICAS:-1}"
NLP_MAX_REPLICAS="${NLP_MAX_REPLICAS:-3}"

export COMPOSE_PROJECT_NAME
export COMPOSE_FILE="$COMPOSE_FILE_PATH"

if [ ! -f "$COMPOSE_FILE_PATH" ]; then
  echo "[autoscaler] compose file not found: $COMPOSE_FILE_PATH"
  exit 1
fi

ml_up_hits=0
ml_down_hits=0
nlp_up_hits=0
nlp_down_hits=0

log() {
  echo "[autoscaler] $*"
}

service_containers() {
  svc="$1"
  docker ps \
    --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}" \
    --filter "label=com.docker.compose.service=${svc}" \
    --format '{{.Names}}'
}

replicas_count() {
  svc="$1"
  n="$(service_containers "$svc" | wc -l | tr -d ' ')"
  if [ -z "$n" ]; then
    echo 0
  else
    echo "$n"
  fi
}

avg_cpu_pct() {
  svc="$1"
  names="$(service_containers "$svc" | tr '\n' ' ')"
  if [ -z "$names" ]; then
    echo 0
    return
  fi
  docker stats --no-stream --format '{{.CPUPerc}}' $names \
    | tr -d '%' \
    | awk '{sum+=$1;n+=1} END {if(n==0) print 0; else printf "%.2f", sum/n}'
}

scale_service() {
  svc="$1"
  target="$2"
  current="$(replicas_count "$svc")"
  if [ "$current" -eq "$target" ]; then
    return
  fi
  log "scaling ${svc}: ${current} -> ${target}"
  docker compose \
    --project-directory "$PROJECT_DIR" \
    -f "$COMPOSE_FILE_PATH" \
    up -d --no-deps --scale "${svc}=${target}" "${svc}" >/dev/null
}

float_ge() {
  a="$1"
  b="$2"
  awk -v x="$a" -v y="$b" 'BEGIN {exit !(x>=y)}'
}

float_le() {
  a="$1"
  b="$2"
  awk -v x="$a" -v y="$b" 'BEGIN {exit !(x<=y)}'
}

reconcile_service() {
  svc="$1"
  min_rep="$2"
  max_rep="$3"
  up_hits_ref="$4"
  down_hits_ref="$5"

  current="$(replicas_count "$svc")"
  if [ "$current" -lt "$min_rep" ]; then
    scale_service "$svc" "$min_rep"
    current="$min_rep"
  fi

  cpu="$(avg_cpu_pct "$svc")"
  log "service=${svc} replicas=${current} avg_cpu=${cpu}%"

  eval "up_hits=\${$up_hits_ref}"
  eval "down_hits=\${$down_hits_ref}"

  if float_ge "$cpu" "$AUTOSCALE_CPU_UP_PCT"; then
    up_hits=$((up_hits + 1))
    down_hits=0
  elif float_le "$cpu" "$AUTOSCALE_CPU_DOWN_PCT"; then
    down_hits=$((down_hits + 1))
    up_hits=0
  else
    up_hits=0
    down_hits=0
  fi

  if [ "$up_hits" -ge "$AUTOSCALE_COOLDOWN_CYCLES" ] && [ "$current" -lt "$max_rep" ]; then
    scale_service "$svc" $((current + 1))
    up_hits=0
    down_hits=0
  elif [ "$down_hits" -ge "$AUTOSCALE_COOLDOWN_CYCLES" ] && [ "$current" -gt "$min_rep" ]; then
    scale_service "$svc" $((current - 1))
    up_hits=0
    down_hits=0
  fi

  eval "$up_hits_ref=$up_hits"
  eval "$down_hits_ref=$down_hits"
}

if ! docker compose --project-directory "$PROJECT_DIR" -f "$COMPOSE_FILE_PATH" version >/dev/null 2>&1; then
  log "docker compose unavailable inside autoscaler container"
  exit 1
fi

log "started project=${COMPOSE_PROJECT_NAME} interval=${AUTOSCALE_INTERVAL_SEC}s up=${AUTOSCALE_CPU_UP_PCT}% down=${AUTOSCALE_CPU_DOWN_PCT}% cooldown=${AUTOSCALE_COOLDOWN_CYCLES}"

while true; do
  reconcile_service "ml-inference" "$ML_MIN_REPLICAS" "$ML_MAX_REPLICAS" "ml_up_hits" "ml_down_hits"
  reconcile_service "nlp-inference" "$NLP_MIN_REPLICAS" "$NLP_MAX_REPLICAS" "nlp_up_hits" "nlp_down_hits"
  sleep "$AUTOSCALE_INTERVAL_SEC"
done
