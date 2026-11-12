FROM python:3.13.12-slim as BASE

WORKDIR /app

ENV PYTHONPATH=/app

COPY services/retrain/requirements.txt services/retrain/constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY common /app/common

COPY jobs/retrain /app/jobs/retrain

COPY rules /app/rules