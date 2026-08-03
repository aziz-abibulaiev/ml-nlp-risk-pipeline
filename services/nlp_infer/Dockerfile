FROM python:3.13-slim AS builder

WORKDIR /app

COPY services/nlp_infer/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.13-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local
COPY common /app/common
COPY services/nlp_infer/main.py .

ENV PORT=8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
