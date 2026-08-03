FROM python:3.13-slim AS builder

WORKDIR /app

COPY services/ml_infer/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.13-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local
COPY common /app/common
COPY services/ml_infer/main.py .
COPY models/ml/ml_config.json .

ENV PORT=8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
