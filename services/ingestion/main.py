import os
import json
import uvicorn
import logging
import uuid
import httpx
import time

from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import ValidationError
from prometheus_client import Counter, Histogram, generate_latest

from common.schemas import ApiEvent
from common.utils import make_hash_id, getenv_str, getenv_int


SERVICE_NAME     = "ingestion-gateway"
LOG_LEVEL        = getenv_str("LOG_LEVEL", "INFO").upper()
ORCHESTRATOR_URL = getenv_str("ORCHESTRATOR_URL", "")      # e.g. http://orchestrator:8000/process
FORWARD_MODE     = getenv_str("FORWARD_MODE", "direct")    # "direct" | "none"
MAX_BODY_BYTES   = getenv_int("MAX_BODY_BYTES", 1048576)   # 1 MB
AUTH_TOKEN       = getenv_str("AUTH_TOKEN", "")
REQS = Counter("ingestion_requests_total", "Total ingestion requests", ["status"])
LATENCY = Histogram("ingestion_request_latency_seconds", "Request latency")


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","service":"'+SERVICE_NAME+'","msg":"%(message)s"}',
)
log = logging.getLogger(SERVICE_NAME)

app = FastAPI(title="ingestion-gateway", version="1.0.0")

@app.on_event("startup")
async def _log_cfg():
    log.info(
        f'config FORWARD_MODE="{FORWARD_MODE}" ORCHESTRATOR_URL="{ORCHESTRATOR_URL}" '
        f'MAX_BODY_BYTES="{MAX_BODY_BYTES}" LOG_LEVEL="{LOG_LEVEL}"'
    )


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    return {"ready": True, "version": app.version}

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain")

# Helpers
def ensure_trace_id(x_trace_id: Optional[str], event: ApiEvent) -> str:
    tid = (x_trace_id or "").strip()
    return tid if tid else f"id-{uuid.uuid4().hex[:8]}"

def sanitize_event_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    def _clip_str(s: str, n: int = 5000) -> str:
        return s if len(s) <= n else s[:n] + "...<truncated>"
    out = dict(d)
    headers = out.get("headers") or {}
    out["headers"] = {k: _clip_str(str(v), 2000) for k, v in headers.items()}
    payload = out.get("payload")
    try:
        s = json.dumps(payload, sort_keys=True)[:10000]
        out["payload"] = json.loads(s)
    except Exception:
        out["payload"] = {}
    return out


@app.post("/ingest")
async def ingest(
    req: Request,
    x_trace_id: Optional[str] = Header(default=None, convert_underscores=False),
    x_forwarded_for: Optional[str] = Header(default=None),
):
    start = time.perf_counter()
    status = "ok"
    try:
        log.debug(f'req_cfg FORWARD_MODE="{FORWARD_MODE}" ORCHESTRATOR_URL="{ORCHESTRATOR_URL}"')

        # 1) Обмеження розміру запиту
        body_bytes = await req.body()
        if len(body_bytes) > MAX_BODY_BYTES:
            status = "err"
            raise HTTPException(status_code=413, detail="Payload too large")

        # 2) Валідація структури
        try:
            body = json.loads(body_bytes.decode("utf-8"))
            event = ApiEvent(**body)
        except (json.JSONDecodeError, ValidationError) as e:
            status = "err"
            log.warning(f"invalid_event: {e}")
            raise HTTPException(status_code=400, detail="Invalid event schema")

        # 3) Обробка trace_id / ip
        hdr_trace = x_trace_id or req.headers.get("X-Trace-Id") or req.headers.get("x-trace-id")
        trace_id = ensure_trace_id(hdr_trace, event)
        if AUTH_TOKEN:
            token = req.headers.get("X-Auth-Token", "")
            if token != AUTH_TOKEN:
                status = "err"
                raise HTTPException(status_code=401, detail="Unauthorized")
        if not hdr_trace:
            log.info(f'generated_trace_id trace_id="{trace_id}"')
        if not event.source_ip and x_forwarded_for:
            event.source_ip = x_forwarded_for.split(",")[0].strip()

        # 4) Генеруємо стабільний hash_id
        event.hash_id = make_hash_id(event.ts, event.endpoint, event.user_id or "", event.payload or {})

        # 5) Лог події (безпечний)
        log.info(f'event_received trace_id="{trace_id}" hash_id="{event.hash_id}" endpoint="{event.endpoint}" method="{event.method}"')

        # 6) Режими доставки
        if FORWARD_MODE == "direct" and not ORCHESTRATOR_URL:
            status = "err"
            log.error("FORWARD_MODE=direct but ORCHESTRATOR_URL is empty")
            return JSONResponse(
                status_code=202,
                content={"status": "accepted", "hash_id": event.hash_id, "trace_id": trace_id, "note": "orchestrator url not configured"},
                headers={"X-Trace-Id": trace_id},
            )

        if FORWARD_MODE == "direct" and ORCHESTRATOR_URL:
            try:
                log.info(f'forward_to_orchestrator trace_id="{trace_id}" url="{ORCHESTRATOR_URL}"')
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(
                        ORCHESTRATOR_URL,
                        json={"event": event.model_dump()},
                        headers={"X-Trace-Id": trace_id},
                    )
                resp.raise_for_status()
                data = resp.json()
                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "accepted",
                        "hash_id": event.hash_id,
                        "trace_id": trace_id,
                        "result": data,
                    },
                    headers={"X-Trace-Id": trace_id},
                )
            except httpx.HTTPError as e:
                status = "err"
                log.error(f'orchestrator_error trace_id="{trace_id}" err="{e}"')
                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "accepted",
                        "hash_id": event.hash_id,
                        "trace_id": trace_id,
                        "note": "orchestrator unavailable"
                    },
                    headers={"X-Trace-Id": trace_id},
                )

        # режим none: просто ack
        return JSONResponse(
            status_code=202,
            content={"status": "accepted", "hash_id": event.hash_id, "trace_id": trace_id},
            headers={"X-Trace-Id": trace_id},
        )

    except HTTPException:
        raise
    except Exception as e:
        status = "err"
        log.exception('unhandled_in_ingest')
        raise
    finally:
        REQS.labels(status=status).inc()
        LATENCY.observe(time.perf_counter() - start)

# ──────────────────────────────────────────────────────────────────────────────
# Глобальний обробник непередбачених помилок
# ──────────────────────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    log.exception(f'unhandled_exception path="{request.url.path}"')
    trace_id = request.headers.get("X-Trace-Id") or request.headers.get("x-trace-id") or "-"
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "trace_id": trace_id}, headers={"X-Trace-Id": trace_id})
# ──────────────────────────────────────────────────────────────────────────────
# Запуск
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
    )