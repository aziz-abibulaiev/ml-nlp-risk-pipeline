FROM python:3.13-slim AS builder

WORKDIR /app

COPY services/alerting/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY common /app/common
COPY services/alerting/main.py .
COPY policies/recommended_actions.json /app/policies/

ENV PORT=8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]