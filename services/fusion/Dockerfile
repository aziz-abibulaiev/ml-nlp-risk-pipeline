FROM python:3.13-slim AS builder

WORKDIR /app

COPY services/fusion/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.13-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local
COPY common /app/common
COPY services/fusion/main.py .

ENV PORT=8000

CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "30"]
