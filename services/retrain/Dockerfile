FROM python:3.13-slim AS builder

WORKDIR /app

ENV PYTHONPATH=/app

COPY services/retrain/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.13-slim AS runtime

WORKDIR /app

ENV PYTHONPATH=/app

COPY --from=builder /install /usr/local

COPY common /app/common

COPY jobs/retrain /app/jobs/retrain

COPY rules /app/rules
