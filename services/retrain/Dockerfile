FROM python:3.13-slim as BASE

WORKDIR /app

ENV PYTHONPATH=/app

COPY services/retrain/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common /app/common

COPY jobs/retrain /app/jobs/retrain

COPY rules /app/rules